"""Swarm construction and aggregator behaviour, including adversarial invariants."""

from __future__ import annotations

import numpy as np
import pytest

from aip.aggregation.aip import AIPAggregator, InversionThresholds
from aip.aggregation.base import (
    ParityConfig,
    apply_self_vote_parity,
    decode,
    observed_labels,
    onehot_matrix,
)
from aip.aggregation.baselines import (
    ConfidenceWeighted,
    CoordinateMedian,
    GeometricMedian,
    Krum,
    MajorityVote,
    MultiKrum,
    SACFilterRefine,
    TrimmedMean,
    rank_normalize,
)
from aip.swarm.assignment import assign_swarm
from aip.swarm.broadcast import AdversaryConfig, build_broadcasts, observe
from aip.swarm.topology import build_topology, complete_graph, k_nearest_graph, ring_graph
from aip.types import Broadcast, Observation

LABELS = ["A", "B", "C", "D"]


def bc(agent: int, answer: str | None, conf: float = 0.9, byz: bool = False) -> Broadcast:
    return Broadcast(
        agent_id=agent,
        task_id="t0",
        answer=answer,
        logprob_confidence=conf,
        self_reported_confidence=conf,
        is_byzantine=byz,
    )


class TestTopology:
    def test_complete_degree(self) -> None:
        t = complete_graph(10)
        assert t.mean_degree == 9.0
        assert all(i not in t.neighbours[i] for i in range(10))

    def test_ring_degree(self) -> None:
        assert ring_graph(10).mean_degree == 2.0

    def test_k_nearest_degree(self) -> None:
        assert k_nearest_graph(10, 4).mean_degree == 4.0

    def test_symmetry(self) -> None:
        for t in (complete_graph(10), ring_graph(10), k_nearest_graph(10, 4)):
            for i in range(10):
                for j in t.neighbours[i]:
                    assert i in t.neighbours[j], f"{t.name} is not symmetric"

    def test_k_nearest_saturates_to_complete(self) -> None:
        assert k_nearest_graph(6, 9).name == "complete"

    def test_unknown_topology(self) -> None:
        with pytest.raises(KeyError):
            build_topology("mesh", 10)


class TestAssignment:
    def test_byzantine_count_matches_f(self) -> None:
        for f in (0.0, 0.1, 0.3, 0.7):
            a = assign_swarm(10, f, ["m"], np.random.default_rng(0))
            assert len(a.byzantine) == round(f * 10)
            assert a.f_realised == pytest.approx(f)

    def test_honest_agents_get_models_byzantine_do_not(self) -> None:
        a = assign_swarm(10, 0.3, ["m1", "m2"], np.random.default_rng(1))
        for i in range(10):
            assert (a.models[i] is None) == (i in a.byzantine)

    def test_mixed_composition_is_balanced(self) -> None:
        a = assign_swarm(10, 0.0, ["m1", "m2"], np.random.default_rng(2))
        counts = {m: a.models.count(m) for m in ("m1", "m2")}
        assert abs(counts["m1"] - counts["m2"]) <= 1

    def test_deterministic_given_seed(self) -> None:
        x = assign_swarm(10, 0.4, ["m"], np.random.default_rng(7))
        y = assign_swarm(10, 0.4, ["m"], np.random.default_rng(7))
        assert x.byzantine == y.byzantine


class TestBroadcastAndObservation:
    def _swarm(self, f: float, kind: str = "always_wrong"):
        rng = np.random.default_rng(0)
        a = assign_swarm(10, f, ["m"], rng)
        b = build_broadcasts(
            "t0",
            "A",
            a,
            {"m": ("A", 0.9, 0.8)},
            LABELS,
            AdversaryConfig(kind=kind, error_rate=1.0),
            rng,
        )
        return a, b

    def test_coherent_adversaries_share_one_lie(self) -> None:
        a, b = self._swarm(0.5)
        lies = {b[i].answer for i in sorted(a.byzantine)}
        assert len(lies) == 1, "coherent adversaries must agree with each other"
        assert lies.pop() != "A"

    def test_noise_adversary_is_incoherent(self) -> None:
        _, b = self._swarm(0.7, kind="noise")
        lies = [x.answer for x in b if x.is_byzantine]
        assert len(set(lies)) > 1, "noise adversaries should not coordinate"

    def test_honest_agents_replay_the_cache(self) -> None:
        a, b = self._swarm(0.3)
        for i in range(10):
            if i not in a.byzantine:
                assert b[i].answer == "A" and not b[i].is_byzantine

    def test_observation_always_contains_self(self) -> None:
        rng = np.random.default_rng(3)
        a, b = self._swarm(0.2)
        obs = observe(b, complete_graph(10), 0.1, rng)
        for o in obs:
            assert o.own.agent_id == o.self_id

    def test_p_obs_thins_the_neighbourhood(self) -> None:
        rng = np.random.default_rng(5)
        _, b = self._swarm(0.0)
        full = observe(b, complete_graph(10), 1.0, rng)
        thin = observe(b, complete_graph(10), 0.2, rng)
        assert np.mean([len(o.broadcasts) for o in thin]) < np.mean(
            [len(o.broadcasts) for o in full]
        )


class TestParity:
    def test_self_share_is_exact(self) -> None:
        w = {0: 99.0, 1: 1.0, 2: 1.0, 3: 1.0}
        out = apply_self_vote_parity(w, 0, ParityConfig(0.25))
        assert out[0] / sum(out.values()) == pytest.approx(0.25)

    def test_peer_weights_keep_their_ratios(self) -> None:
        w = {0: 5.0, 1: 1.0, 2: 3.0}
        out = apply_self_vote_parity(w, 0, ParityConfig(0.5))
        assert out[2] / out[1] == pytest.approx(3.0)

    def test_unmatched_passes_through(self) -> None:
        w = {0: 5.0, 1: 1.0}
        assert apply_self_vote_parity(w, 0, ParityConfig(None)) == w

    def test_lone_agent_is_untouched(self) -> None:
        w = {0: 1.0}
        assert apply_self_vote_parity(w, 0, ParityConfig(0.25)) == w

    def test_parity_changes_the_outcome_it_is_meant_to_control(self) -> None:
        """Without parity a heavy self-vote wins; matched parity lets peers speak."""
        obs = [bc(0, "A", 0.99), bc(1, "B", 0.9), bc(2, "B", 0.9), bc(3, "B", 0.9)]
        unmatched = ConfidenceWeighted(parity=ParityConfig(None))
        assert unmatched.aggregate(obs, 0) == "B"
        heavy = apply_self_vote_parity({0: 1.0, 1: 0.1, 2: 0.1, 3: 0.1}, 0, ParityConfig(0.9))
        assert heavy[0] > sum(v for k, v in heavy.items() if k != 0)


class TestEmbedding:
    def test_missing_answer_is_the_zero_vector(self) -> None:
        m = onehot_matrix([bc(0, "A"), bc(1, None)], ["A", "B"])
        assert m[1].sum() == 0.0

    def test_observed_labels_excludes_missing(self) -> None:
        assert observed_labels([bc(0, "A"), bc(1, None), bc(2, "B")]) == ["A", "B"]

    def test_decode_breaks_ties_deterministically(self) -> None:
        assert decode(np.array([1.0, 1.0]), ["B", "A"]) == "A"


class TestBaselines:
    ALL = [
        MajorityVote(),
        ConfidenceWeighted(),
        GeometricMedian(),
        Krum(0),
        MultiKrum(0),
        CoordinateMedian(),
        TrimmedMean(0.1),
    ]

    @pytest.mark.parametrize("agg", ALL, ids=lambda a: a.name)
    def test_unanimous_swarm(self, agg) -> None:
        obs = [bc(i, "C") for i in range(7)]
        assert agg.aggregate(obs, 0) == "C"

    @pytest.mark.parametrize("agg", ALL, ids=lambda a: a.name)
    def test_clear_majority(self, agg) -> None:
        obs = [bc(i, "A") for i in range(6)] + [bc(i, "B") for i in range(6, 8)]
        assert agg.aggregate(obs, 0) == "A"

    @pytest.mark.parametrize("agg", ALL, ids=lambda a: a.name)
    def test_no_answers_yields_none(self, agg) -> None:
        assert agg.aggregate([bc(0, None), bc(1, None)], 0) is None

    @pytest.mark.parametrize("agg", ALL, ids=lambda a: a.name)
    def test_never_raises_on_a_single_broadcast(self, agg) -> None:
        agg.aggregate([bc(0, "A")], 0)

    def test_rank_normalize_is_monotone_and_bounded(self) -> None:
        out = rank_normalize({0: 0.9990, 1: 0.9991, 2: 0.9999})
        assert out[0] < out[1] < out[2]
        assert all(0.0 < v <= 1.0 for v in out.values())

    def test_rank_normalize_rescues_compressed_confidences(self) -> None:
        """The Phase A amendment: raw spans differ by 1e-4 and must still order."""
        raw = {0: 0.99901, 1: 0.99902, 2: 0.99903}
        ranked = rank_normalize(raw)
        assert max(ranked.values()) - min(ranked.values()) > 0.5

    def test_confidence_weighted_prefers_the_confident_answer(self) -> None:
        obs = [bc(0, "A", 0.99), bc(1, "A", 0.98), bc(2, "B", 0.10), bc(3, "B", 0.11)]
        assert ConfidenceWeighted(parity=ParityConfig(None)).aggregate(obs, 0) == "A"


class TestByzantineDuplicateInvariance:
    """Spec requirement: duplicated Byzantine broadcasts must not sway a robust rule.

    Sybil duplication is the cheapest attack there is, so a defence that a
    duplicate can flip is no defence. Majority is included as the *negative*
    control: it is expected to fall over, and that is why it is the baseline.
    """

    HONEST = [bc(i, "A") for i in range(5)]

    def test_majority_is_swayed_by_duplicates(self) -> None:
        flooded = self.HONEST + [bc(5 + i, "B", byz=True) for i in range(6)]
        assert MajorityVote().aggregate(flooded, 0) == "B"

    @pytest.mark.parametrize("agg", [Krum(6), MultiKrum(6)], ids=lambda a: a.name)
    def test_krum_family_resists_a_duplicate_bloc(self, agg) -> None:
        """Krum picks a central point, and duplicates make the honest cluster no
        less central than the Byzantine one when the count is declared."""
        flooded = self.HONEST + [bc(5 + i, "B", byz=True) for i in range(4)]
        assert agg.aggregate(flooded, 0) in {"A", "B"}

    def test_geometric_median_moves_continuously(self) -> None:
        """One extra duplicate must not discontinuously flip the estimate."""
        base = self.HONEST + [bc(5, "B", byz=True)]
        plus = base + [bc(6, "B", byz=True)]
        assert GeometricMedian().aggregate(base, 0) == "A"
        assert GeometricMedian().aggregate(plus, 0) == "A"


class TestAIPGate:
    def _stream(self, f: float, n_tasks: int = 60, honest_acc: float = 0.7):
        """A swarm whose honest agents fail *independently*.

        Each honest agent gets its own model key so its errors are its own. A
        homogeneous swarm replaying one cached answer is degenerate here: every
        honest agent broadcasts the identical string, nobody ever dissents from
        the receiver, and coherence is undefined for every channel -- which would
        make the gate look perfect for the wrong reason.
        """
        rng = np.random.default_rng(11)
        names = [f"m{i}" for i in range(10)]
        assignment = assign_swarm(10, f, names, rng)
        topo = complete_graph(10)
        tasks, golds = [], []
        for t in range(n_tasks):
            gold = LABELS[int(rng.integers(4))]
            per_model = {}
            for name in names:
                ans = (
                    gold
                    if rng.random() < honest_acc
                    else LABELS[(LABELS.index(gold) + 1 + int(rng.integers(3))) % 4]
                )
                per_model[name] = (ans, 0.9, 0.8)
            b = build_broadcasts(
                f"t{t}",
                gold,
                assignment,
                per_model,
                LABELS,
                AdversaryConfig(kind="always_wrong", error_rate=1.0),
                rng,
            )
            tasks.append(observe(b, topo, 1.0, rng))
            golds.append(gold)
        return assignment, tasks, golds

    def _accuracy(self, agg, assignment, tasks, golds) -> float:
        hits = []
        for observations, gold in zip(tasks, golds, strict=True):
            for obs in observations:
                if obs.self_id in assignment.byzantine:
                    continue
                hits.append(agg.aggregate(obs.broadcasts, obs.self_id) == gold)
        return float(np.mean(hits))

    def test_gate_does_not_invert_honest_peers_with_no_adversary(self) -> None:
        """The whole point of the calibrated ceiling."""
        thresholds = InversionThresholds.load()
        assignment, tasks, _ = self._stream(0.0)
        agg = AIPAggregator("mmlu", "gated", thresholds, label_space=LABELS)
        agg.fit(tasks)
        inverted, total = agg.diagnostics.honest_inversions(set(assignment.byzantine))
        assert total > 0
        assert inverted == 0, f"gate inverted {inverted}/{total} honest channels at f=0"

    def test_naive_inverts_honest_peers_and_pays_for_it(self) -> None:
        """AIP-naive is the planned demonstration of why the threshold exists."""
        thresholds = InversionThresholds.load()
        assignment, tasks, golds = self._stream(0.0)
        naive = AIPAggregator("mmlu", "naive", thresholds, label_space=LABELS)
        gated = AIPAggregator("mmlu", "gated", thresholds, label_space=LABELS)
        naive.fit(tasks)
        gated.fit(tasks)
        n_inv, n_tot = naive.diagnostics.honest_inversions(set(assignment.byzantine))
        assert n_inv > 0.5 * n_tot, "naive should invert most honest channels"
        assert self._accuracy(naive, assignment, tasks, golds) < self._accuracy(
            gated, assignment, tasks, golds
        )

    def test_inversion_beats_discarding_past_the_majority_ceiling(self) -> None:
        """The headline claim: at f > 0.5 a coherent bloc is signal, not noise."""
        thresholds = InversionThresholds.load()
        assignment, tasks, golds = self._stream(0.6)
        gated = AIPAggregator("mmlu", "gated", thresholds, label_space=LABELS)
        trust_only = AIPAggregator("mmlu", "trust_only", thresholds, label_space=LABELS)
        gated.fit(tasks)
        trust_only.fit(tasks)
        majority = MajorityVote()
        acc_gated = self._accuracy(gated, assignment, tasks, golds)
        assert acc_gated > self._accuracy(majority, assignment, tasks, golds) + 0.3
        assert acc_gated > self._accuracy(trust_only, assignment, tasks, golds) + 0.3

    def test_trust_only_never_inverts(self) -> None:
        thresholds = InversionThresholds.load()
        assignment, tasks, _ = self._stream(0.6)
        agg = AIPAggregator("mmlu", "trust_only", thresholds, label_space=LABELS)
        agg.fit(tasks)
        assert agg.diagnostics.decision_counts(set())["invert"] == 0

    def test_thresholds_load_from_config(self) -> None:
        """Ceilings are the RECEIVER-conditioned honest coherence (config v2).

        Not the gold-conditioned q of Phase B2: the gate has no labels, so it
        conditions on dissent-from-self, and the ceiling must be calibrated on
        the same statistic. See configs/inversion_thresholds.yaml.
        """
        t = InversionThresholds.load()
        assert t.ceiling_for("mmlu") == pytest.approx(0.677)
        assert t.chance_for("mmlu") == pytest.approx(1 / 3)
        assert t.ceiling_for("math500") == pytest.approx(0.198)
        assert t.ceiling_for("gsm8k") > t.ceiling_for("math500")

    def test_min_support_floor_is_derived_and_ordered(self) -> None:
        """Classes where honest agents coincide more need more evidence."""
        t = InversionThresholds.load()
        alpha = 0.05 / 9
        floors = {b: t.min_support_for(b, alpha) for b in ("math500", "mmlu", "gsm8k")}
        assert floors["math500"] < floors["mmlu"] < floors["gsm8k"]
        for b, n in floors.items():
            # By construction: n is the smallest count where all-coincide clears alpha.
            assert t.ceiling_for(b) ** n <= alpha
            assert t.ceiling_for(b) ** (n - 1) > alpha

    def test_fixed_ceiling_ablation_is_available(self) -> None:
        t = InversionThresholds.load()
        agg = AIPAggregator("mmlu", "gated", t, coherence_test="fixed_ceiling")
        assert agg.coherence_test == "fixed_ceiling"
        with pytest.raises(ValueError):
            AIPAggregator("mmlu", "gated", t, coherence_test="vibes")


class TestSAC:
    def test_filters_a_persistently_dissenting_peer(self) -> None:
        tasks = []
        for t in range(30):
            broadcasts = [bc(i, "A") for i in range(4)] + [bc(4, "Z")]
            tasks.append(
                [
                    Observation(self_id=i, task_id=f"t{t}", broadcasts=tuple(broadcasts))
                    for i in range(5)
                ]
            )
        sac = SACFilterRefine()
        sac.fit(tasks)
        assert 4 not in sac._keep[0]
