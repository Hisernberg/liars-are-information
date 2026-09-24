"""Communication topologies: complete, ring, k-nearest.

A topology says who *can* hear whom. Partial observability (``p_obs``) then
thins what is actually received on a given task; the two are deliberately
separate, because a sparse graph and a lossy channel degrade a swarm in
different ways and the sweep varies them independently.

Agents are arranged on a cycle so that ring and k-nearest are well defined, and
an agent is never its own neighbour -- observing yourself is handled by the
observation builder, which must always include the self-vote.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topology:
    """Static neighbour sets. ``neighbours[i]`` excludes ``i``."""

    name: str
    n_agents: int
    neighbours: tuple[frozenset[int], ...]
    param: int | None = None

    def degree(self, agent: int) -> int:
        return len(self.neighbours[agent])

    @property
    def mean_degree(self) -> float:
        return sum(len(s) for s in self.neighbours) / self.n_agents


def complete_graph(n_agents: int) -> Topology:
    """Everyone hears everyone. The default for the main sweep."""
    return Topology(
        name="complete",
        n_agents=n_agents,
        neighbours=tuple(frozenset(j for j in range(n_agents) if j != i) for i in range(n_agents)),
    )


def ring_graph(n_agents: int) -> Topology:
    """Each agent hears its two cycle neighbours -- the sparsest connected case."""
    if n_agents < 3:
        return complete_graph(n_agents)
    return Topology(
        name="ring",
        n_agents=n_agents,
        neighbours=tuple(
            frozenset({(i - 1) % n_agents, (i + 1) % n_agents}) for i in range(n_agents)
        ),
        param=2,
    )


def k_nearest_graph(n_agents: int, k: int) -> Topology:
    """Each agent hears the ``k`` nearest agents on the cycle, balanced either side."""
    if k >= n_agents - 1:
        return complete_graph(n_agents)
    if k < 1:
        raise ValueError("k must be at least 1")
    neighbours = []
    for i in range(n_agents):
        picked: set[int] = set()
        offset = 1
        while len(picked) < k:
            for delta in (offset, -offset):
                if len(picked) >= k:
                    break
                picked.add((i + delta) % n_agents)
            offset += 1
        neighbours.append(frozenset(picked - {i}))
    return Topology(name="k_nearest", n_agents=n_agents, neighbours=tuple(neighbours), param=k)


def build_topology(name: str, n_agents: int, k: int | None = None) -> Topology:
    key = name.strip().lower()
    if key == "complete":
        return complete_graph(n_agents)
    if key == "ring":
        return ring_graph(n_agents)
    if key in ("k_nearest", "knn", "k-nearest"):
        if k is None:
            raise ValueError("k_nearest topology requires k")
        return k_nearest_graph(n_agents, k)
    raise KeyError(f"unknown topology {name!r}; known: complete, ring, k_nearest")
