---
license: mit
language:
  - en
pretty_name: "Liars Are Information v3 — RACE: receiver-anchored channel estimation for Byzantine-robust multi-agent LLMs"
size_categories:
  - 100K<n<1M
task_categories:
  - question-answering
tags:
  - multi-agent
  - llm-agents
  - byzantine-fault-tolerance
  - adversarial-robustness
  - deception
  - crowdsourcing
  - dawid-skene
  - research-artifact
configs:
  - config_name: cache_greedy
    description: "Honest greedy answers of 9 models on 6 benchmarks (X3-repaired)."
    data_files:
      - split: train
        path: "data/cache/*/*.parquet"
  - config_name: cache_adversarial
    description: "Answers written by LLMs prompted to deceive (4 prompts x 2 attacker models)."
    data_files:
      - split: train
        path: "data/cache_adversarial/*/*/*.parquet"
  - config_name: live_swarm
    description: "Fresh CPU inference by six small open models in honest, saboteur and debate roles."
    data_files:
      - split: train
        path: "data/live_cache/raw/*/*/*.parquet"
  - config_name: results_main
    description: "E1 per-task results (world x method x split x task)."
    data_files:
      - split: train
        path: "results/main/per_task.parquet"
  - config_name: results_llm
    description: "E3 replayed real-LLM deceivers, per-task results."
    data_files:
      - split: train
        path: "results/llm/per_task.parquet"
  - config_name: results_budget
    description: "E9 information budget: RACE, RACE-D, known-channel and liar-removal oracles, per task."
    data_files:
      - split: train
        path: "results/ext/per_task.parquet"
  - config_name: channel_estimates
    description: "E7 label-free channel estimates vs evaluator-side truth, per receiver-peer channel."
    data_files:
      - split: train
        path: "results/extra/channel_estimates.parquet"
---

# Liars Are Information v3.1: RACE

**Receiver-Anchored Channel Estimation for Byzantine-robust aggregation in multi-agent LLM systems.**

Each honest agent fits a latent-truth model to its own **unlabeled** history, anchored on the one thing it knows: that it is itself honest. Every peer then gets a signed weight. Peers whose answers depend on the truth positively are trusted. Peers whose answers depend on it negatively, such as liars who never tell the truth, are *inverted*. Peers whose answers are independent of the truth are discarded. The approach has no breakdown point at f = ½. It also sidesteps the coherence impossibility that defeated the original AIP gate.

Code, figures, videos and manuscript: see `README.md` in this release and the GitHub repository.

| Accuracy (%) | Receiver alone | Majority | AIP | **RACE** |
|---|---:|---:|---:|---:|
| Coherent liars, f = 0.9 | {{mainSelfNine}} | {{mainMajNine}} | {{mainAipNine}} | **{{mainRaceNine}}** |
| Gate-aware liars, f = 0.7 | {{zooCamoSelfSeven}} | {{zooGateMajSeven}} | {{zooGateAipSeven}} | **{{zooGateRaceSeven}}** |
| Real LLM deceivers, f = 0.7 | {{llmSelfSeven}} | {{llmMajSeven}} | {{llmAipSeven}} | **{{llmRaceSeven}}** |
| Independent liars, f = 0.7 | {{extSelfIndepSeven}} | {{extMajIndepSeven}} | {{extAipIndepSeven}} | **{{extRaceIndepSeven}}** |

Against independent liars at f = 0.7, RACE ({{extRaceIndepSeven}}%) also beats an oracle that knows who lies and removes them ({{extRemovalIndepSeven}}%). The liars carry information, and RACE reads it without labels (study E9).

**Live swarm (E8).** Six small open models answered 240 MMLU and BoolQ questions live on a CPU, as honest agents, covert saboteurs and debaters (2,880 answers, `data/live_cache/`). On MMLU, RACE lifts every honest agent by {{liveMmluGainMin}}–{{liveMmluGainMax}} points. On binary BoolQ the frozen v3.0 fell below the receiver alone ({{liveBoolqLlmVthreeFive}}% vs {{liveBoolqLlmSelfFive}}% at f = 0.5), because small models answer yes/no questions with an option bias that a symmetric channel cannot represent. RACE v3.1 uses class-conditional channels on binary questions and scores {{liveBoolqLlmRaceFive}}%. It was adopted after this observation, and both versions are reported.

<p align="center"><img src="media/gif/film_1_protocol.gif" width="760" alt="Agents broadcast answers; each honest agent decides with learned trust links"></p>

**Videos.** `media/film_liars_are_information.mp4` is the explainer film. It shows agents broadcasting answers, each honest agent's trust links, the swarm's trust matrix learning over 90 unlabeled questions, and one decision in slow motion. `media/swarm_*.mp4` follow a single receiver in three attack scenarios.

## Provenance

* Honest and adversarial caches: Dhruv Jyoti Das, [`Dhruv1000/Liars_Are_Information`](https://huggingface.co/datasets/Dhruv1000/Liars_Are_Information), post-X3 repair.
* Fixed AIP code: the Sep-12 release in [`Nabidnur/Liars_Are_Information-bucket`](https://huggingface.co/buckets/Nabidnur/Liars_Are_Information-bucket).
* v3 additions: RACE, the theory, the audit (`docs/AUDIT.md`), studies E1–E9 including the live swarm and the information budget, the TrustLayer API, figures, the explainer film, videos and the manuscript.

Benchmark items and model outputs remain under their upstream licenses; ARC and BoolQ are CC BY-SA.
