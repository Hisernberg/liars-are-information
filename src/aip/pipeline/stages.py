"""Stage definitions for the MX multi-stage agentic pipeline.

Three stages -- solver, critic, refiner -- over a single MedQA item. Each stage
is a prompt builder over the task and the *aggregated* output of the stages
before it. That dependency is the cascade channel: a stage's input is what the
previous stage's aggregator decided, so a fault contained at stage 2 never
reaches stage 3, and a fault that escapes does.

The prompts deliberately reuse the benchmark's own system prompt and answer
format so the extractor in ``aip.tasks.medqa`` applies unchanged at every stage.
A stage that answered in a different format would need its own extractor, and a
second extractor is a second place for silent scoring drift.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The three stages, in order. ``needs`` names which earlier stage outputs the
#: prompt consumes, and is what the runner uses to decide what can be cached
#: once and what must be regenerated per arm.
STAGE_ORDER = ("solver", "critic", "refiner")

SOLVER_USER = (
    "{question}\n\n"
    "{choices_block}\n\n"
    "Think briefly. Then give your final answer on its own last line in exactly "
    "this format:\n"
    "Answer: <letter>\n"
    "where <letter> is one of {letters}."
)

CRITIC_USER = (
    "{question}\n\n"
    "{choices_block}\n\n"
    "A first responder proposed: {prior_answer}\n\n"
    "Critique that proposal. Say briefly whether it is right or wrong and why. "
    "Then give your own final answer on its own last line in exactly this "
    "format:\n"
    "Answer: <letter>\n"
    "where <letter> is one of {letters}."
)

REFINER_USER = (
    "{question}\n\n"
    "{choices_block}\n\n"
    "A first responder proposed: {solver_answer}\n"
    "A reviewer then concluded: {critic_answer}\n\n"
    "Weigh both and decide. Give your final answer on its own last line in "
    "exactly this format:\n"
    "Answer: <letter>\n"
    "where <letter> is one of {letters}."
)


@dataclass(frozen=True)
class Stage:
    """One pipeline stage."""

    name: str
    user_template: str
    #: Which earlier stages' aggregate answers this stage's prompt consumes.
    needs: tuple[str, ...]

    def build_user(self, fields: dict[str, object], prior: dict[str, str]) -> str:
        """Render this stage's user message.

        ``prior`` maps an earlier stage name to its aggregated answer. A missing
        dependency is an error rather than an empty string: a refiner prompt
        silently missing the critic's verdict would still generate, still score,
        and quietly measure a two-stage pipeline.
        """
        missing = [s for s in self.needs if not prior.get(s)]
        if missing:
            raise ValueError(
                f"stage {self.name!r} needs aggregated output from {missing}, "
                f"got keys {sorted(prior)}"
            )
        merged = dict(fields)
        if "solver" in self.needs:
            merged["solver_answer"] = prior["solver"]
            merged["prior_answer"] = prior["solver"]
        if "critic" in self.needs:
            merged["critic_answer"] = prior["critic"]
        return self.user_template.format(**merged)


STAGES: dict[str, Stage] = {
    "solver": Stage("solver", SOLVER_USER, ()),
    "critic": Stage("critic", CRITIC_USER, ("solver",)),
    "refiner": Stage("refiner", REFINER_USER, ("solver", "critic")),
}


def stage(name: str) -> Stage:
    try:
        return STAGES[name]
    except KeyError:
        raise KeyError(f"unknown stage {name!r}; known: {sorted(STAGES)}") from None
