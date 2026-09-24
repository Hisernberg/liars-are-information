import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import dgx_low_resource as run  # noqa: E402


def test_history_validation_test_do_not_overlap_and_cover_all_tasks():
    ids = [f"t{i}" for i in range(200)]
    split = run.split_indices(ids)
    assert {k: len(v) for k, v in split.items()} == {"history": 80, "validation": 40, "test": 80}
    assert len(set(sum(split.values(), []))) == 200
    reversed_ids = ids[::-1]
    reversed_split = run.split_indices(reversed_ids)
    assert {ids[i] for i in split["test"]} == {reversed_ids[i] for i in reversed_split["test"]}


def test_increasing_corruption_preserves_unaffected_model_identities():
    models = [f"m{i}" for i in range(7)]
    cache = run.pc.BenchmarkCache("mmlu", ["t"], ["A"],
        {(m, 0): [("A", .8, .8)] for m in models})
    world = dict(seed=73, benchmark="mmlu", composition="frozen_mix",
        attack="always_wrong", coherence_p=1., p_obs=1., f=0.)
    base, _, _ = run.build_world(cache, models, world)
    previous = set()
    for f in [.3, .5, .7]:
        assignment, _, _ = run.build_world(cache, models, dict(world, f=f))
        assert previous <= assignment.byzantine
        for i in assignment.honest:
            assert assignment.models[i] == base.models[i]
        previous = set(assignment.byzantine)


def test_visibility_does_not_change_broadcasts_or_corruption():
    models = ["a", "b"]
    cache = run.pc.BenchmarkCache("mmlu", ["x", "y"], ["A", "B"],
        {(m, 0): [("A", .8, .8), ("C", .8, .8)] for m in models})
    world = dict(seed=73, benchmark="mmlu", composition="frozen_mix",
        attack="gate_aware", coherence_p=.5, p_obs=1., f=.5)
    first, _, raw1 = run.build_world(cache, models, world)
    second, _, raw2 = run.build_world(cache, models, dict(world, p_obs=.5))
    assert first == second
    assert raw1 == raw2


def test_attack_selection_cannot_choose_using_test_accuracy():
    rows = []
    for p in [0., 1.]:
        for split in ["validation", "test"]:
            for method in ["aip_gated", "self_only", "aip_trust_only"]:
                # Opposite ranking between validation and test.
                accuracy = p if split == "validation" else 1. - p
                rows.append(dict(study="adaptive", benchmark="mmlu", f=.5, seed=1,
                    coherence_p=p, split=split, method=method, task_id=split, accuracy=accuracy))
    selected, _ = run.select_attacks(pd.DataFrame(rows))
    assert (selected.selected_p == 0.).all()
    assert (selected.test_accuracy == 1.).all()


def test_math_equivalence_is_scored_and_never_used_as_a_lie():
    assert run.score_answer("math500", "0.5", r"\frac{1}{2}")
    models = ["a", "b"]
    cache = run.pc.BenchmarkCache("math500", ["t"], [r"\frac{1}{2}"],
        {("a", 0): [("0.5", .8, .8)], ("b", 0): [("2", .8, .8)]})
    for seed in range(8):
        world = dict(seed=seed, benchmark="math500", composition="frozen_mix",
            attack="gate_aware", coherence_p=.5, p_obs=1., f=.5)
        assignment, _, raw = run.build_world(cache, models, world)
        for i in assignment.byzantine:
            assert not run.score_answer("math500", raw[0][i].answer, cache.gold[0])
        for i in assignment.honest:
            assert raw[0][i].answer in ["0.5", "2"]
