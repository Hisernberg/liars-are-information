# Page cut: 17 → 12 — EXECUTED

Applied 2026-09-11. The plan below records what was done and what it cost, so a
reviewer response or a longer-limit venue can reverse specific items.

## Result

**12 pages**, IEEE two-column, references included, no appendix. 0 undefined
references, 0 undefined citations, 0 overfull boxes, 0 non-font LaTeX warnings.
All 35 claim ids still cited. Sections 13 → 10.

## What moved, per tier

| tier | action | kept |
|---|---|---|
| **0** | Appendix A dissolved. Proposition 1's statement, proof sketch and TV bound stay in the main text (§8); only the formal setup, the exchangeability condition and the finite-$n$ statement moved to `docs/proposition1_note.md`. The evasion figure moved into §8 rather than leaving with the appendix. | proof sketch, TV bound, all four resolution qualifications |
| **1** | §11 prior-art-revisited dissolved into §2 and §9; §8 real-LLM adversaries folded into §6; §7 protocol demoted to §5 Law 4; §13 reproducibility compressed. | every claim; E2a, E3 and X1 were briefly lost in a bad range edit and restored from git |
| **2** | Figures 8 → 3. Deleted `pobs_constraint`, `gating_ablation`, `invertibility_spectrum`, `burst_windowed`; `correlation_heatmap` moved to the artifact. Tables 4 → 3: the attack-class table became prose. | `money_figure`, `evasion_band` (the no-interpolation grid), `sybil_accuracy` (the non-separation that tests E2b) — each the sole evidence for a claim |
| **3** | Prose 12,531 → ~8,900 words across every section. | — |

## Amendments applied as instructed

- **Refutations.** Flat-in-$f$ (H1) and the withdrawn $f=0.7$ anchor stay in
  **prose** in §9; the other eight are in Table~\ref{tab:refutations}, one row
  each with magnitude and floor. The difference was recovered from §1 and §2,
  which lost ~450 and ~560 words respectively.
- **Proposition 1.** Sketch and TV bound in the main text, as required.

## What the last page cost

The final page was pure bibliography. Removing it took four figures rather than
three: `burst_windowed` went because B1's evidence is fully numeric in the text
(pooled $0.470 \to 0.965$, stationary cost $\leq 0.003$), whereas the three
surviving figures have no textual substitute.

## Reversal order if the limit turns out to be larger

1. `+0.3 pp` restore `burst_windowed.pdf`
2. `+0.3 pp` restore `invertibility_spectrum.pdf`
3. `+0.3 pp` restore `correlation_heatmap.pdf`
4. `+0.6 pp` un-tabulate the eight refutations
5. `+1.7 pp` restore Appendix A from `docs/proposition1_note.md`
