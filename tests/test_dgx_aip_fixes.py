import os
import subprocess
import sys
from dataclasses import replace

import numpy as np

from aip.aggregation.aip import AIPAggregator
from aip.types import Broadcast, Observation


def test_disjoint_peer_exposures_do_not_create_coherence():
    tasks = []
    for t in range(40):
        tid = str(t)
        own = Broadcast(0, tid, 'A', .8, .8, False)
        peer = Broadcast(1 if t < 20 else 2, tid, 'B', .8, .8, False)
        tasks.append((Observation(0, tid, (own, peer)),))
    agg = AIPAggregator('mmlu')
    agg.fit(tasks)
    for peer in [1, 2]:
        stats = agg.diagnostics.channels[0][peer]
        assert stats.n_observed == 20
        assert stats.joint_dissents == 0
        assert np.isnan(stats.coherence)


def test_refit_clears_previous_receiver_history():
    b = Broadcast(0, 'old', 'A', .8, .8, False)
    agg = AIPAggregator('mmlu')
    agg.fit([(Observation(0, 'old', (b,)),)])
    b2 = replace(b, agent_id=4, task_id='new')
    agg.fit([(Observation(4, 'new', (b2,)),)])
    assert set(agg.diagnostics.channels) == {4}


def test_threshold_randomization_is_stable_across_python_hash_seeds():
    script = "from aip.aggregation.aip import AIPAggregator; a=AIPAggregator('mmlu',randomized_threshold=.3,threshold_seed=17); a._draw_ceiling_jitter(4); print(a._effective_ceiling())"
    vals = [subprocess.check_output([sys.executable, '-c', script], env=dict(os.environ, PYTHONHASHSEED=str(seed)), text=True) for seed in [1, 99]]
    assert vals[0] == vals[1]


def test_randomized_ceiling_is_a_valid_probability():
    a = AIPAggregator('boolq', randomized_threshold=.3)
    a._ceiling_jitter = 1.3
    assert a._effective_ceiling() == 1.0
