#!/usr/bin/env python3
"""Plugging RACE into a multi-agent loop with the TrustLayer API.

The swarm streams MedQA questions. Seven honest agents are real open-weight
models (cached answers); three saboteurs are real LLMs that were prompted to
mislead after seeing the honest answers (cached ``rushing`` attack). One honest
agent (Phi-4-mini) runs a TrustLayer. No labels are ever passed to it.

    PYTHONPATH=src python examples/trust_layer_demo.py
"""

import numpy as np

from lai.api import TrustLayer
from lai.data import FROZEN_MODELS, load_benchmark, score

data = load_benchmark("medqa")
honest = list(FROZEN_MODELS)
saboteurs = {"saboteur-1": ("rushing", "llama32_3b"), "saboteur-2": ("rushing", "ministral3_14b"),
             "saboteur-3": ("semantic_negation", "ministral3_14b")}
me = "phi4_mini_reasoning"
layer = TrustLayer(me, data.label_space, forgetting=0.98)

hits = {"TrustLayer (RACE)": 0, "majority vote": 0, "trust only myself": 0}
order = np.random.default_rng(0).permutation(len(data.task_ids))
for step, t in enumerate(order, 1):
    answers = {m: data.honest[m][t] for m in honest}
    answers |= {name: data.adversarial[src][t] for name, src in saboteurs.items()}
    decision = layer.decide(answers)            # uses only earlier questions
    layer.observe(data.task_ids[t], answers)    # then commits this one
    votes = [a for a in answers.values() if a is not None]
    majority = max(sorted(set(votes)), key=votes.count)
    gold = data.gold[t]
    hits["TrustLayer (RACE)"] += score("medqa", decision.answer, gold)
    hits["majority vote"] += score("medqa", majority, gold)
    hits["trust only myself"] += score("medqa", answers[me], gold)
    if step in (1, 10, 50, 200):
        print(f"\nafter {step:3d} questions: {decision.explain()}")

print("\npeer trust after the stream (most distrusted first):")
for p in layer.trust():
    print(f"  {p.peer:<22} â={p.accuracy:.2f}  λ={p.weight:+.2f}  {p.decision}")
print("\naccuracy over the stream:", {k: f"{v / len(order):.1%}" for k, v in hits.items()})
