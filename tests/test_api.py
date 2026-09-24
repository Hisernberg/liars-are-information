"""TrustLayer: the drop-in API reproduces RACE and stays causal."""

from __future__ import annotations

import numpy as np

from lai.api import TrustLayer

LABELS = ("A", "B", "C", "D")


def _round(rng, liars=("mallory", "trudy", "eve")):
    gold = LABELS[rng.integers(4)]
    wrong = [x for x in LABELS if x != gold]
    answers = {name: (gold if rng.random() < 0.75 else str(rng.choice(wrong)))
               for name in ("me", "alice", "bob", "carol")}
    for name in liars:
        answers[name] = str(rng.choice(wrong))  # never the truth, never coordinated
    return gold, answers


def test_trust_layer_learns_to_invert_string_named_liars():
    rng = np.random.default_rng(0)
    layer = TrustLayer("me", LABELS)
    for q in range(80):
        _, answers = _round(rng)
        layer.observe(q, answers)
    by_name = {p.peer: p for p in layer.trust()}
    assert {by_name[n].decision for n in ("mallory", "trudy", "eve")} == {"INVERT"}
    assert {by_name[n].decision for n in ("alice", "bob", "carol")} == {"TRUST"}
    hits = own = 0
    for _ in range(200):
        gold, answers = _round(rng)
        hits += layer.decide(answers).answer == gold
        own += answers["me"] == gold
    assert hits > own + 20


def test_decide_is_causal_and_falls_back_to_self():
    layer = TrustLayer("me", LABELS)
    d = layer.decide({"me": "B", "x": "C"})
    assert d.answer == "B" and d.n_history == 0
    layer.observe(0, {"me": "B", "x": "C"})
    before = layer.decide({"me": "A", "x": "D"}).posterior
    # Deciding does not change history; only observe does.
    assert len(layer) == 1 and layer.decide({"me": "A", "x": "D"}).posterior == before


def test_forgetting_and_explain():
    rng = np.random.default_rng(1)
    layer = TrustLayer("me", LABELS, forgetting=0.95)
    for q in range(60):
        layer.observe(q, _round(rng)[1])
    text = layer.decide(_round(rng)[1]).explain()
    assert "inverting 3" in text and "unlabeled questions" in text
