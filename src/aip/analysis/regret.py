"""Minimax-regret analysis over the Byzantine fraction.

A method's *regret* at a given ``f`` is how far it falls short of the best method
available at that ``f``. Its **minimax regret** is the worst such shortfall across
the whole range. This is the right summary for a defence that has to be chosen
before ``f`` is known: it asks "how bad can this choice be?" rather than "how good
is it on average?", and a method that wins at low ``f`` while collapsing at high
``f`` is penalised for the collapse rather than rescued by the wins.

Oracle methods are excluded by default. Krum, Multi-Krum and trimmed mean are
handed the true Byzantine count, which a deployed agent does not have, so
including them would mean comparing a deployable choice against one nobody can
actually make.
"""

from __future__ import annotations

import pandas as pd


def regret_table(
    frame: pd.DataFrame,
    value: str = "accuracy_point",
    group: tuple[str, ...] = ("benchmark", "f"),
    method_col: str = "method",
    deployable_only: bool = True,
    oracle_col: str = "is_oracle",
) -> pd.DataFrame:
    """Per-method regret at each ``f``, plus the minimax summary."""
    df = frame.copy()
    if deployable_only and oracle_col in df.columns:
        df = df[~df[oracle_col].astype(bool)]
    if df.empty:
        return pd.DataFrame()

    best = df.groupby(list(group))[value].max().rename("best")
    joined = df.join(best, on=list(group))
    joined["regret"] = joined["best"] - joined[value]
    return joined


def minimax_summary(
    frame: pd.DataFrame, method_col: str = "method", by: tuple[str, ...] = ("benchmark",)
) -> pd.DataFrame:
    """Worst-case and mean regret per method, sorted by worst case."""
    keys = [*by, method_col] if by else [method_col]
    out = (
        frame.groupby(keys)["regret"]
        .agg(minimax_regret="max", mean_regret="mean", n_cells="size")
        .reset_index()
        .sort_values([*by, "minimax_regret"] if by else ["minimax_regret"])
    )
    return out


def worst_case_f(frame: pd.DataFrame, method: str, benchmark: str | None = None) -> float:
    """The ``f`` at which a method incurs its maximum regret."""
    sub = frame[frame.method == method]
    if benchmark is not None:
        sub = sub[sub.benchmark == benchmark]
    if sub.empty:
        return float("nan")
    return float(sub.loc[sub["regret"].idxmax(), "f"])
