"""Probe tests for the MX pipeline harness.

These run without a GPU. They exist to prove the wiring before any model is
loaded, because the MX failure modes are the ones this project has already hit
once each: a stage silently missing its dependency, a liar that does not
actually differ from the honest replica it replaced, a tie-break that depends on
replica ordering, and telemetry that reports a decision the aggregator never
made.
"""

from __future__ import annotations

import numpy as np
import pytest

from aip.pipeline.arms import (
    DISCARD,
    INVERT,
    TRUST,
    StageDecision,
    build_arms,
    plurality,
    to_broadcasts,
)
from aip.pipeline.liars import attractor_rider, fixed_lie, inject
from aip.pipeline.stages import STAGE_ORDER, stage

LABELS = ["A", "B", "C", "D"]


class TestStages:
    def test_stage_order_is_the_pipeline(self) -> None:
        assert STAGE_ORDER == ("solver", "critic", "refiner")

    def test_solver_needs_nothing(self) -> None:
        assert stage("solver").needs == ()

    def test_later_stages_declare_their_dependencies(self) -> None:
        assert stage("critic").needs == ("solver",)
        assert stage("refiner").needs == ("solver", "critic")

    def test_missing_dependency_raises_rather_than_rendering(self) -> None:
        """A refiner prompt missing the critic verdict would still generate."""
        fields = {"question": "q", "choices_block": "A) x", "letters": "A, B"}
        with pytest.raises(ValueError, match="needs aggregated output"):
            stage("refiner").build_user(fields, {"solver": "A"})

    def test_dependency_reaches_the_prompt(self) -> None:
        fields = {"question": "q", "choices_block": "A) x", "letters": "A, B"}
        text = stage("refiner").build_user(fields, {"solver": "A", "critic": "B"})
        assert "A first responder proposed: A" in text
        assert "A reviewer then concluded: B" in text


class TestPlurality:
    def test_counts_the_anchor_as_a_vote(self) -> None:
        assert plurality(["A", "B"], "B", LABELS) == "B"

    def test_tie_break_does_not_depend_on_replica_order(self) -> None:
        a = plurality(["A", "B"], None, LABELS)
        b = plurality(["B", "A"], None, LABELS)
        assert a == b == "A"

    def test_all_abstentions_give_none(self) -> None:
        assert plurality([None, None], None, LABELS) is None


class TestLiars:
    def test_l1_rides_the_shared_wrong_answer(self) -> None:
        rng = np.random.default_rng(0)
        res = attractor_rider(["B", "B"], gold="A", label_space=LABELS, rng=rng)
        assert res.answer == "B" and not res.fallback

    def test_l1_falls_back_when_there_is_no_attractor(self) -> None:
        """Both honest replicas right: nothing to ride, and it must say so."""
        rng = np.random.default_rng(0)
        res = attractor_rider(["A", "A"], gold="A", label_space=LABELS, rng=rng)
        assert res.answer != "A" and res.fallback

    def test_l1_tie_break_is_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        one = attractor_rider(["B", "C"], "A", LABELS, rng).answer
        two = attractor_rider(["C", "B"], "A", LABELS, rng).answer
        assert one == two

    def test_l2_ignores_the_honest_replicas(self) -> None:
        """The uninformed liar must not depend on what the others said."""
        assert fixed_lie("A", LABELS).answer == fixed_lie("A", LABELS).answer
        assert fixed_lie("A", LABELS).answer != "A"

    def test_l1_and_l2_differ_where_the_attractor_is_not_the_fixed_lie(self) -> None:
        """If they never differ, MX-H4 is measuring one adversary twice."""
        rng = np.random.default_rng(0)
        l1 = attractor_rider(["C", "C"], "A", LABELS, rng).answer
        l2 = fixed_lie("A", LABELS).answer
        assert l1 == "C" and l2 == "B" and l1 != l2

    def test_inject_replaces_exactly_one_slot(self) -> None:
        rng = np.random.default_rng(0)
        out, res = inject(["A", "C", "C"], 0, "L1", "A", LABELS, rng)
        assert out[1:] == ["C", "C"]
        assert out[0] == "C" == res.answer

    def test_liar_does_not_read_the_replica_it_replaced(self) -> None:
        """Slot 0 held 'D'; the liar must ride the others' 'C', not see 'D'."""
        rng = np.random.default_rng(0)
        out, _ = inject(["D", "C", "C"], 0, "L1", "A", LABELS, rng)
        assert out[0] == "C"

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown liar variant"):
            inject(["A"], 0, "L9", "A", LABELS, np.random.default_rng(0))


class TestArms:
    def test_five_arms_with_the_registered_shapes(self) -> None:
        arms = build_arms("m14", ("m14", "phi4", "l3b"), "m14")
        assert set(arms) == {"P0", "P1", "P2", "P3", "P4"}
        assert arms["P0"].k == 1
        assert all(arms[a].k == 3 for a in ("P1", "P2", "P3", "P4"))
        assert not arms["P1"].is_cross_family
        assert arms["P3"].is_cross_family

    def test_rule_pairs_share_a_roster(self) -> None:
        """P1/P2 and P3/P4 must differ only in the rule, or the contrast is confounded."""
        arms = build_arms("m14", ("m14", "phi4", "l3b"), "m14")
        assert arms["P1"].replicas == arms["P2"].replicas
        assert arms["P3"].replicas == arms["P4"].replicas
        assert arms["P1"].rule != arms["P2"].rule
        assert arms["P3"].rule != arms["P4"].rule

    def test_anchor_is_present_and_is_the_self_id(self) -> None:
        bs = to_broadcasts(["A", "B"], ("m", "m"), "C", "t1")
        assert len(bs) == 3
        assert bs[-1].answer == "C"
        assert bs[-1].agent_id == 2


class TestTelemetry:
    def test_false_inversion_counts_only_honest_channels(self) -> None:
        d = StageDecision(
            task_id="t", stage="critic", arm="P2", output="A", anchor="A",
            replica_answers=("A", "B", "B"), replica_models=("m",) * 3,
            decisions=(TRUST, INVERT, INVERT), liar_slot=1,
        )
        assert d.honest_slots() == (0, 2)
        assert d.honest_inverts() == 1

    def test_no_decisions_logged_means_no_false_inversions(self) -> None:
        d = StageDecision("t", "critic", "P1", "A", "A", ("A", "B"), ("m", "m"))
        assert d.honest_inverts() == 0

    def test_decision_vocabulary_matches_the_aggregator(self) -> None:
        """Re-exported, not redefined -- a parallel vocabulary would drift."""
        from aip.aggregation import aip as gate

        assert (INVERT, TRUST, DISCARD) == (gate.INVERT, gate.TRUST, gate.DISCARD)


class TestGateWiring:
    """The AIP gate is needs_fit. An unfitted one still returns an answer.

    That is the trap: it returns a degenerate answer with no channel statistics
    and no logged decisions, and nothing in the call signature says so. These
    tests make the difference visible.
    """

    @staticmethod
    def _gate(labels):
        from aip.aggregation.aip import AIPAggregator, InversionThresholds
        from aip.aggregation.base import ParityConfig

        return AIPAggregator("medqa", "gated", InversionThresholds.load(),
                             ParityConfig(0.1), labels)

    @staticmethod
    def _stream():
        answers = {f"t{i}": (["A", "A", "B"] if i % 3 else ["B", "B", "C"])
                   for i in range(30)}
        anchors = {f"t{i}": ("A" if i % 3 else "B") for i in range(30)}
        return answers, anchors

    def test_unfitted_gate_logs_no_decisions(self) -> None:
        from aip.pipeline.runner import aggregate_stage

        arms = build_arms("m", ("m", "p", "l"), "m")
        answers, anchors = self._stream()
        d = aggregate_stage(arms["P2"], "t1", answers["t1"], anchors["t1"],
                            LABELS, self._gate(LABELS))
        assert d.decisions == ()

    def test_fitted_gate_logs_one_decision_per_replica(self) -> None:
        from aip.pipeline.runner import aggregate_stage, fit_stage

        arms = build_arms("m", ("m", "p", "l"), "m")
        answers, anchors = self._stream()
        gate = self._gate(LABELS)
        fit_stage(gate, arms["P2"], answers, anchors)
        d = aggregate_stage(arms["P2"], "t1", answers["t1"], anchors["t1"],
                            LABELS, gate)
        assert len(d.decisions) == arms["P2"].k
        assert all(x in {INVERT, TRUST, DISCARD} for x in d.decisions)

    def test_aip_arm_without_an_aggregator_raises(self) -> None:
        from aip.pipeline.runner import aggregate_stage

        arms = build_arms("m", ("m", "p", "l"), "m")
        with pytest.raises(ValueError, match="needs an AIP aggregator"):
            aggregate_stage(arms["P2"], "t", ["A", "A", "B"], "A", LABELS, None)

    def test_observation_anchor_is_the_receiver(self) -> None:
        """The gate's tests are receiver-anchored; the anchor must be self_id."""
        from aip.pipeline.runner import stage_observations

        arms = build_arms("m", ("m", "p", "l"), "m")
        obs = stage_observations(arms["P2"], {"t": ["A", "B", "C"]}, {"t": "D"})
        o = obs[0][0]
        assert o.self_id == 3
        assert o.own.answer == "D"
        assert len(o.peers) == 3


class TestGenerationCache:
    """An incomplete generation must be a cache miss, never an abstention.

    Rows are written before generation and filled in after, so a run that dies
    mid-flight leaves `answer=None` on disk. Served back, those score as an
    agent that declined to answer -- indistinguishable in the output from a real
    abstention, and wrong in every arm downstream. This cost one manual cache
    deletion during Phase 1; the guard replaces the need to remember.
    """

    def test_incomplete_row_is_a_miss(self, tmp_path) -> None:
        from aip.pipeline.runner import GenerationCache

        c = GenerationCache(tmp_path / "c.parquet")
        c.put("k1", stage="solver", model="m", task_id="t", sample_index=0,
              prompt="p", answer=None, raw="")
        assert c.get("k1") is None

    def test_complete_row_is_served(self, tmp_path) -> None:
        from aip.pipeline.runner import GenerationCache

        c = GenerationCache(tmp_path / "c.parquet")
        c.put("k1", stage="solver", model="m", task_id="t", sample_index=0,
              prompt="p", answer="A", raw="Answer: A")
        assert c.get("k1")["answer"] == "A"

    def test_incomplete_rows_do_not_survive_a_round_trip(self, tmp_path) -> None:
        from aip.pipeline.runner import GenerationCache

        path = tmp_path / "c.parquet"
        c = GenerationCache(path)
        c.put("good", stage="s", model="m", task_id="t", sample_index=0,
              prompt="p", answer="A", raw="")
        c.put("bad", stage="s", model="m", task_id="t", sample_index=1,
              prompt="p", answer=None, raw="")
        c.save()
        reloaded = GenerationCache(path).load()
        assert reloaded.get("good") is not None
        assert reloaded.get("bad") is None
        assert len(reloaded) == 1


class TestExtractionRoundTrip:
    """Extraction is the one path the synthetic probes never touch.

    The Phase 1 run died on `Extraction.answer`, which does not exist -- the
    field is `.value`. The aggregation probes could not catch it because they
    feed answers in directly. This closes that gap.
    """

    def test_extraction_field_is_value_and_round_trips(self) -> None:
        from aip.tasks import get_benchmark

        bench = get_benchmark("medqa")
        ex = bench.extract_answer("Reasoning about the case.\nAnswer: C")
        assert hasattr(ex, "value") and not hasattr(ex, "answer")
        assert bench.normalize(ex.value) == "C"

    def test_unparseable_completion_yields_none(self) -> None:
        from aip.tasks import get_benchmark

        bench = get_benchmark("medqa")
        assert bench.extract_answer("I decline to answer.").value is None

    def test_every_stage_answer_format_is_extractable(self) -> None:
        """All three stages ask for the same answer format on purpose."""
        from aip.pipeline.stages import STAGE_ORDER, stage
        from aip.tasks import get_benchmark

        bench = get_benchmark("medqa")
        for name in STAGE_ORDER:
            tmpl = stage(name).user_template
            assert "Answer: <letter>" in tmpl, f"{name} asks for a different format"
        assert bench.normalize(bench.extract_answer("x\nAnswer: B").value) == "B"

    def test_record_generation_uses_the_real_dataclass_fields(self) -> None:
        """Exercises the field names that killed three GPU runs."""
        from aip.models.inference import GenerationResult, TokenLogprob
        from aip.pipeline.runner import record_generation
        from aip.tasks import get_benchmark

        bench = get_benchmark("medqa")
        result = GenerationResult(
            text="Weighing the options.\nAnswer: D",
            tokens=[TokenLogprob(text="x", logprob=-0.1, start=0, end=1)] * 7,
            finish_reason="stop", n_prompt_tokens=42,
        )
        row: dict = {}
        record_generation(row, result, bench)
        assert row["answer"] == "D"
        assert row["n_tokens"] == 7
        assert row["finish_reason"] == "stop"
        assert row["raw"].endswith("Answer: D")

    def test_record_generation_handles_an_unparseable_completion(self) -> None:
        from aip.models.inference import GenerationResult
        from aip.pipeline.runner import record_generation
        from aip.tasks import get_benchmark

        row: dict = {}
        record_generation(row, GenerationResult(text="no answer here", tokens=[],
                                                finish_reason="length",
                                                n_prompt_tokens=1),
                          get_benchmark("medqa"))
        assert row["answer"] is None and row["n_tokens"] == 0


class TestTruncationGuard:
    """A truncated completion must not yield an answer the model never gave.

    Found live in Phase 1: phi4_mini_reasoning, capped at 512 tokens, was cut off
    mid-sentence arguing about option A, and the shared extractor's
    leading-letter fallback returned C. 113 of 464 generations were affected, at
    a 100% parse rate that hid the problem entirely. The extractor is not at
    fault -- its fallbacks are correct for untruncated completions from models
    that ignore the instructed format. The fault was a global 512-token cap
    overriding each model's registry budget.
    """

    @staticmethod
    def _result(text: str, finish: str):
        from aip.models.inference import GenerationResult, TokenLogprob

        return GenerationResult(text=text,
                                tokens=[TokenLogprob(text="x", logprob=-0.1,
                                                     start=0, end=1)] * 5,
                                finish_reason=finish, n_prompt_tokens=10)

    def test_truncated_without_commitment_is_an_abstention(self) -> None:
        from aip.pipeline.runner import record_generation
        from aip.tasks import get_benchmark

        text = ("Now, looking at the options:\n\nA. Disclose the error. This "
                "aligns with honesty. C. Reassign the case, which would be")
        row: dict = {}
        record_generation(row, self._result(text, "length"), get_benchmark("medqa"))
        assert row["answer"] is None
        assert row["answer_discarded"] == "truncated_without_commitment"
        assert row["truncated"] is True and row["committed_answer"] is False

    def test_truncated_but_committed_is_kept(self) -> None:
        """Commitment then truncation of trailing text is still a real answer."""
        from aip.pipeline.runner import record_generation
        from aip.tasks import get_benchmark

        row: dict = {}
        record_generation(row, self._result("Reasoning.\nAnswer: B\nfurther tex",
                                            "length"), get_benchmark("medqa"))
        assert row["answer"] == "B" and row["answer_discarded"] is None

    def test_untruncated_keeps_the_extractor_fallbacks(self) -> None:
        """A model that ignores the format but finished is still extractable."""
        from aip.pipeline.runner import record_generation
        from aip.tasks import get_benchmark

        row: dict = {}
        record_generation(row, self._result("I think the answer is **D**.", "stop"),
                          get_benchmark("medqa"))
        assert row["answer"] == "D"
        assert row["truncated"] is False

    def test_the_exact_observed_failure_no_longer_manufactures_an_answer(self) -> None:
        from aip.pipeline.runner import record_generation
        from aip.tasks import get_benchmark

        observed = ("\nNow, looking at the options:\n\nA. Disclose the error to "
                    "the patient and put it in the operative report. This aligns "
                    "with the principle of honesty and transparency. The operative "
                    "report should accurately reflect what happened during the "
                    "surgery, including any complications or errors, even if they "
                    "were subsequently repaired. The patient should be informed "
                    "because they are a patient, and it's part of")
        row: dict = {}
        record_generation(row, self._result(observed, "length"),
                          get_benchmark("medqa"))
        assert row["answer"] is None, "the C that was manufactured here is back"
