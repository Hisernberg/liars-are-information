"""Attack wrapper interface shared by all adversary classes.

Seven classes, but only four need new inference. The other three are exact
offline transforms or compositions of cached outputs, which is not a shortcut:
falsified_confidence *must* leave the answer and logprob untouched for its
prediction to be testable, and burst and the bandit are schedulers over outputs
that already exist.

Adversarial generations go to their own cache namespace,
``data/cache_adversarial/{attack}/{benchmark}/{model}.parquet``. The Phase A
honest caches are read-only, forever.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from aip.tasks.base import Benchmark, ChatMessage
from aip.types import TaskItem

#: Extra context an attack may need beyond the task itself (rushing uses this to
#: pass the honest broadcasts it is allowed to condition on).
AttackContext = dict[str, Any]


@dataclass(frozen=True)
class Attack:
    """One adversarial behaviour."""

    name: str
    needs_inference: bool
    description: str
    build_messages: Callable[[Benchmark, TaskItem, AttackContext], list[ChatMessage]] | None = None
    #: True if the attack is expected to produce a *coherent* channel -- the same
    #: wrong answer from independent adversaries -- and therefore be invertible.
    expected_invertible: bool = True
    #: Extra generated tokens this attack needs relative to an honest answer.
    token_multiplier: float = 1.0
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def format_peer_answers(peers: Sequence[tuple[str, str | None]]) -> str:
    """Render honest broadcasts for the rushing adversary to condition on."""
    lines = []
    for i, (_model, answer) in enumerate(peers, start=1):
        lines.append(f"  Agent {i}: {answer if answer is not None else '(no answer)'}")
    return "\n".join(lines) if lines else "  (no answers received yet)"
