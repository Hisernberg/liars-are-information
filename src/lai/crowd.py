"""Classical label-free crowdsourcing aggregators, as baselines for RACE.

All four estimate annotator reliability without labels, from the receiver's own
history, exactly the information RACE gets. None has a receiver anchor: each
assumes, like the crowdsourcing literature, that the *majority* of annotators
is better than chance. That is precisely the assumption a liar majority breaks.

* ``IWMV``  Iterative weighted majority voting (Li and Yu, 2014): alternate a
  weighted vote with per-annotator weights ``K * accuracy - 1``.
* ``MACE``  Multi-Annotator Competence Estimation (Hovy et al., 2013): each
  annotator either answers truthfully or "spams" from its own label
  distribution; EM over competence and spam distributions.
* ``GLAD``  Generative model of Labels, Abilities and Difficulties (Whitehill et
  al., 2009): P(correct) = sigmoid(ability_j * inverse_difficulty_t), errors
  uniform; EM with gradient M-steps. Unseen (test) questions get average difficulty.
* ``KOS``   Karger, Oh and Shah (2014) iterative message passing, binary
  questions only; worker reliabilities from the history graph weight test votes.

Every class follows the ``Aggregator`` interface of ``aip.aggregation.base``:
``fit`` on history observations, ``aggregate(broadcasts, self_id)`` on a new
question. Ties go to the receiver's own answer, as for RACE.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aip.aggregation.base import Aggregator
from aip.types import Broadcast, Observation


def _rows(tasks: Sequence[Sequence[Observation]]) -> dict[int, list[dict[int, str | None]]]:
    out: dict[int, list[dict[int, str | None]]] = {}
    for observations in tasks:
        for obs in observations:
            out.setdefault(obs.self_id, []).append({b.agent_id: b.answer for b in obs.broadcasts})
    return out


class _Crowd(Aggregator):
    needs_fit = True
    is_oracle = False

    def __init__(self, label_space: Sequence[str] | None = None, max_iter: int = 50) -> None:
        self.label_space = tuple(sorted(label_space)) if label_space else None
        self.max_iter = int(max_iter)
        self.params: dict[int, dict] = {}

    # candidate sets: closed label space, or the distinct answers heard on the task
    def _cands(self, row: dict[int, str | None]) -> list[str]:
        if self.label_space:
            return list(self.label_space)
        return sorted({a for a in row.values() if a is not None})

    def _encode(self, rows):
        agents = sorted({a for row in rows for a in row})
        col = {a: j for j, a in enumerate(agents)}
        reports = np.full((len(rows), len(agents)), -1, dtype=int)
        k = np.zeros(len(rows), dtype=int)
        for t, row in enumerate(rows):
            cands = self._cands(row)
            idx = {c: i for i, c in enumerate(cands)}
            k[t] = len(cands)
            for a, ans in row.items():
                if ans is not None and ans in idx:
                    reports[t, col[a]] = idx[ans]
        return agents, reports, k

    def fit(self, tasks: Sequence[Sequence[Observation]]) -> None:
        self.params = {}
        for receiver, rows in _rows(tasks).items():
            self.params[receiver] = self._fit(rows)

    def _fit(self, rows) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _scores(self, params: dict, row: dict[int, str | None], cands: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def aggregate(self, observations: Sequence[Broadcast], self_id: int) -> str | None:
        row = {b.agent_id: b.answer for b in observations}
        cands = self._cands(row)
        own = row.get(self_id)
        params = self.params.get(self_id)
        if not cands:
            return None
        if params is None:
            return own
        s = self._scores(params, row, cands)
        best = np.flatnonzero(np.isclose(s, s.max()))
        if own in cands and cands.index(own) in best:
            return own
        return cands[int(best[0])]


class IWMV(_Crowd):
    """Iterative weighted majority voting (Li and Yu, 2014)."""

    name = "iwmv"

    def _fit(self, rows) -> dict:
        agents, reports, k = self._encode(rows)
        n_t, n_j = reports.shape
        v = np.ones(n_j)
        acc = np.full(n_j, 0.5)
        for _ in range(self.max_iter):
            votes = np.zeros((n_t, max(int(k.max()), 1)))
            for j in range(n_j):
                ok = reports[:, j] >= 0
                votes[np.flatnonzero(ok), reports[ok, j]] += v[j]
            yhat = votes.argmax(axis=1)
            seen = reports >= 0
            agree = (reports == yhat[:, None]) & seen
            new_acc = (agree.sum(axis=0) + 1.0) / (seen.sum(axis=0) + 2.0)
            v_new = np.mean(k[k > 0]) * new_acc - 1.0 if np.any(k > 0) else new_acc
            if np.allclose(v_new, v, atol=1e-6):
                acc, v = new_acc, v_new
                break
            acc, v = new_acc, v_new
        return {"agents": {a: j for j, a in enumerate(agents)}, "v": v, "acc": acc}

    def _scores(self, params, row, cands):
        s = np.zeros(len(cands))
        for a, ans in row.items():
            j = params["agents"].get(a)
            if j is not None and ans in cands:
                s[cands.index(ans)] += params["v"][j]
        return s


class MACE(_Crowd):
    """Multi-Annotator Competence Estimation (Hovy et al., 2013), EM with light smoothing."""

    name = "mace"

    def _fit(self, rows) -> dict:
        agents, reports, k = self._encode(rows)
        n_t, n_j = reports.shape
        kmax = max(int(k.max()), 2)
        closed = self.label_space is not None
        theta = np.full(n_j, 0.5)                    # spamming probability
        xi = np.full((n_j, kmax), 1.0 / kmax)        # spam distribution (closed label spaces)
        valid = np.arange(kmax)[None, :] < k[:, None]
        q = np.zeros((n_t, kmax))
        for _ in range(self.max_iter):
            # E-step: posterior over the truth
            logp = np.where(valid, 0.0, -np.inf)
            for j in range(n_j):
                ok = np.flatnonzero(reports[:, j] >= 0)
                if not len(ok):
                    continue
                r = reports[ok, j]
                spam = xi[j, r] if closed else 1.0 / k[ok]
                lik = theta[j] * spam[:, None] * np.ones((1, kmax))
                lik[np.arange(len(ok)), r] += 1.0 - theta[j]
                logp[ok] += np.log(np.clip(lik, 1e-12, None))
            logp -= logp.max(axis=1, keepdims=True)
            q_new = np.where(valid, np.exp(logp), 0.0)
            q_new /= q_new.sum(axis=1, keepdims=True)
            # M-step: competence and spam distribution
            for j in range(n_j):
                ok = np.flatnonzero(reports[:, j] >= 0)
                if not len(ok):
                    continue
                r = reports[ok, j]
                spam = xi[j, r] if closed else 1.0 / k[ok]
                p_true = q_new[ok, r]  # posterior that the reported label is the truth
                # expected responsibility of spamming: certain if the label is wrong, else Bayes
                s_resp = p_true * theta[j] * spam / (theta[j] * spam + 1 - theta[j] + 1e-12) + (1 - p_true)
                theta[j] = (s_resp.sum() + 0.5) / (len(ok) + 1.0)
                if closed:
                    counts = np.bincount(r, weights=s_resp, minlength=kmax) + 0.5
                    xi[j] = counts / counts.sum()
            if np.max(np.abs(q_new - q)) < 1e-6:
                q = q_new
                break
            q = q_new
        return {"agents": {a: j for j, a in enumerate(agents)}, "theta": theta, "xi": xi, "closed": closed}

    def _scores(self, params, row, cands):
        k = len(cands)
        s = np.zeros(k)
        for a, ans in row.items():
            j = params["agents"].get(a)
            if j is None or ans not in cands:
                continue
            r = cands.index(ans)
            th = params["theta"][j]
            spam = params["xi"][j, r] if params["closed"] and r < params["xi"].shape[1] else 1.0 / k
            lik = np.full(k, th * spam)
            lik[r] += 1 - th
            s += np.log(np.clip(lik, 1e-12, None))
        return s


class GLAD(_Crowd):
    """Whitehill et al. (2009): abilities and difficulties, uniform errors."""

    name = "glad"

    def __init__(self, label_space=None, max_iter: int = 30, lr: float = 0.1, inner: int = 10) -> None:
        super().__init__(label_space, max_iter)
        self.lr = float(lr)
        self.inner = int(inner)

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _fit(self, rows) -> dict:
        agents, reports, k = self._encode(rows)
        n_t, n_j = reports.shape
        kmax = max(int(k.max()), 2)
        valid = np.arange(kmax)[None, :] < k[:, None]
        alpha = np.ones(n_j)
        logbeta = np.zeros(n_t)
        seen = reports >= 0
        for _ in range(self.max_iter):
            p = self._sig(alpha[None, :] * np.exp(logbeta)[:, None])  # (T, J) P(correct)
            logq = np.where(valid, 0.0, -np.inf)
            for j in range(n_j):
                ok = np.flatnonzero(seen[:, j])
                r = reports[ok, j]
                wrong = np.log(np.clip((1 - p[ok, j]) / np.maximum(k[ok] - 1, 1), 1e-12, None))
                lik = np.repeat(wrong[:, None], kmax, axis=1)
                lik[np.arange(len(ok)), r] = np.log(np.clip(p[ok, j], 1e-12, None))
                logq[ok] += lik
            logq -= logq.max(axis=1, keepdims=True)
            q = np.where(valid, np.exp(logq), 0.0)
            q /= q.sum(axis=1, keepdims=True)
            # expected correctness of each report under q
            c = np.zeros_like(p)
            tt, jj = np.nonzero(seen)
            c[tt, jj] = q[tt, reports[tt, jj]]
            for _ in range(self.inner):  # gradient ascent on the expected log-likelihood
                beta = np.exp(logbeta)
                p = self._sig(alpha[None, :] * beta[:, None])
                g = (c - p) * seen
                alpha += self.lr * (g * beta[:, None]).sum(axis=0) / np.maximum(seen.sum(axis=0), 1) - 0.01 * (alpha - 1)
                logbeta += self.lr * (g * alpha[None, :] * beta[:, None]).sum(axis=1) / np.maximum(seen.sum(axis=1), 1) \
                    - 0.01 * logbeta
        return {"agents": {a: j for j, a in enumerate(agents)}, "alpha": alpha}

    def _scores(self, params, row, cands):
        k = len(cands)
        s = np.zeros(k)
        for a, ans in row.items():
            j = params["agents"].get(a)
            if j is None or ans not in cands:
                continue
            p = float(np.clip(self._sig(params["alpha"][j]), 1e-6, 1 - 1e-6))  # average difficulty
            lik = np.full(k, np.log((1 - p) / max(k - 1, 1)))
            lik[cands.index(ans)] = np.log(p)
            s += lik
        return s


class KOS(_Crowd):
    """Karger, Oh and Shah (2014) message passing, for binary questions only."""

    name = "kos"

    def __init__(self, label_space=None, max_iter: int = 20) -> None:
        if not label_space or len(set(label_space)) != 2:
            raise ValueError("KOS is defined for binary questions")
        super().__init__(label_space, max_iter)

    def _fit(self, rows) -> dict:
        agents, reports, _ = self._encode(rows)
        a = np.where(reports >= 0, 2 * reports - 1, 0).astype(float)  # +1 / -1 / 0 (missing)
        mask = a != 0
        rng = np.random.default_rng(0)
        y = np.where(mask, rng.normal(1.0, 1.0, a.shape), 0.0)  # worker -> task messages
        for _ in range(self.max_iter):
            tot = (a * y).sum(axis=1, keepdims=True)
            x = np.where(mask, tot - a * y, 0.0)                  # task -> worker, excluding the recipient
            tot_w = (a * x).sum(axis=0, keepdims=True)
            y = np.where(mask, tot_w - a * x, 0.0)
            norm = np.abs(y).max()
            if norm > 0:
                y /= norm
        tot = (a * y).sum(axis=1, keepdims=True)
        x = np.where(mask, tot - a * y, 0.0)
        rel = (a * x).sum(axis=0)
        rel = rel / (np.abs(rel).max() + 1e-12)
        return {"agents": {ag: j for j, ag in enumerate(agents)}, "rel": rel}

    def _scores(self, params, row, cands):
        s = 0.0
        for a, ans in row.items():
            j = params["agents"].get(a)
            if j is None or ans not in cands:
                continue
            s += params["rel"][j] * (1 if cands.index(ans) == 1 else -1)
        return np.array([-s, s])
