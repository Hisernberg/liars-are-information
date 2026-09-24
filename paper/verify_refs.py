#!/usr/bin/env python3
"""Verify every bibliography entry against Crossref (DOI) or arXiv (abs page).

Each entry is (key, query title, expected year, kind, fallback id). An entry is
kept only if the retrieved title matches the query closely; the verified
metadata is written to ``references_verified.json`` and ``references.bib``.
"""

from __future__ import annotations

import difflib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

CANDIDATES = [
    ("dawid1979", "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm", "crossref", None),
    ("lamport1982", "The Byzantine Generals Problem", "crossref", None),
    ("blanchard2017", "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent", "arxiv", "1703.02757"),
    ("yin2018", "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", "arxiv", "1803.01498"),
    ("karger2014", "Budget-Optimal Task Allocation for Reliable Crowdsourcing Systems", "crossref", None),
    ("zhang2016", "Spectral Methods meet EM: A Provably Optimal Algorithm for Crowdsourcing", "arxiv", "1406.3824"),
    ("raykar2010", "Learning From Crowds", "crossref_jmlr", None),
    ("allman2009", "Identifiability of parameters in latent structure models with many observed variables", "crossref", None),
    ("kruskal1977", "Three-way arrays: rank and uniqueness of trilinear decompositions, with application to arithmetic complexity and statistics", "crossref", None),
    ("blackwell1953", "Equivalent Comparisons of Experiments", "crossref", None),
    ("parisi2014", "Ranking and combining multiple predictors without labeled data", "crossref", None),
    ("prelec2017", "A solution to the single-question crowd wisdom problem", "crossref", None),
    ("ladha1992", "The Condorcet Jury Theorem, Free Speech, and Correlated Votes", "crossref", None),
    ("du2024debate", "Improving Factuality and Reasoning in Language Models through Multiagent Debate", "arxiv", "2305.14325"),
    ("wang2023sc", "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "arxiv", "2203.11171"),
    ("amayuelas2024", "MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate", "arxiv", "2406.14711"),
    ("huang2024resilience", "On the Resilience of Multi-Agent Systems with Malicious Agents", "arxiv", "2408.00989"),
    ("kim2025correlated", "Correlated Errors in Large Language Models", "arxiv", "2506.07962"),
    ("kohli2026", "Nine Judges, Two Effective Votes: Correlated Errors Undermine LLM Evaluation Panels", "arxiv", "2605.29800"),
    ("jiang2025hivemind", "Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)", "arxiv", "2510.22954"),
    ("verga2024juries", "Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models", "arxiv", "2404.18796"),
    ("zheng2023judge", "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "arxiv", "2306.05685"),
    ("hendrycks2021mmlu", "Measuring Massive Multitask Language Understanding", "arxiv", "2009.03300"),
    ("clark2019boolq", "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions", "arxiv", "1905.10044"),
    ("cobbe2021gsm8k", "Training Verifiers to Solve Math Word Problems", "arxiv", "2110.14168"),
    ("hendrycks2021math", "Measuring Mathematical Problem Solving With the MATH Dataset", "arxiv", "2103.03874"),
    ("lightman2023verify", "Let's Verify Step by Step", "arxiv", "2305.20050"),
    ("jin2021medqa", "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams", "arxiv", "2009.13081"),
    ("clark2018arc", "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", "arxiv", "1803.05457"),
    ("chen2024blockagents", "BlockAgents: Towards Byzantine-Robust LLM-Based Multi-Agent Coordination via Blockchain", "arxiv", "2401.07007"),
    ("dempster1977", "Maximum Likelihood from Incomplete Data Via the EM Algorithm", "crossref", None),
    ("hoeffding1963", "Probability Inequalities for Sums of Bounded Random Variables", "crossref", None),
    ("holm1979", "A Simple Sequentially Rejective Multiple Test Procedure", "crossref", None),
    ("guerraoui2018", "The Hidden Vulnerability of Distributed Learning in Byzantium", "arxiv", "1802.07927"),
    ("greshake2023", "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection", "arxiv", "2302.12173"),
    ("park2023generative", "Generative Agents: Interactive Simulacra of Human Behavior", "arxiv", "2304.03442"),
    ("hubinger2024sleeper", "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training", "arxiv", "2401.05566"),
    ("park2024deception", "AI Deception: A Survey of Examples, Risks, and Potential Solutions", "arxiv", "2308.14752"),
    ("qwen2024", "Qwen2.5 Technical Report", "arxiv", "2412.15115"),
    ("allal2025smollm2", "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model", "arxiv", "2502.02737"),
    ("olmo2024", "2 OLMo 2 Furious", "arxiv", "2501.00656"),
    ("gemma3", "Gemma 3 Technical Report", "arxiv", "2503.19786"),
    ("llama3", "The Llama 3 Herd of Models", "arxiv", "2407.21783"),
]


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "lai-refcheck/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def sim(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def crossref(title: str) -> dict | None:
    data = json.loads(get("https://api.crossref.org/works?rows=5&query.bibliographic=" + urllib.parse.quote(title)))
    best = None
    for it in data["message"]["items"]:
        t = (it.get("title") or [""])[0]
        s = sim(t, title)
        if best is None or s > best[0]:
            best = (s, it)
    if not best or best[0] < 0.85:
        return None
    it = best[1]
    return dict(title=it["title"][0], doi=it.get("DOI"), venue=(it.get("container-title") or [""])[0],
                year=it.get("issued", {}).get("date-parts", [[None]])[0][0],
                authors=[f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in it.get("author", [])],
                volume=it.get("volume"), pages=it.get("page"), score=round(best[0], 3), source="crossref")


def arxiv(aid: str, title: str) -> dict | None:
    page = get(f"https://arxiv.org/abs/{aid}")
    t = re.search(r'<meta name="citation_title" content="([^"]+)"', page)
    if not t:
        return None
    t = html.unescape(t.group(1))
    if sim(t, title) < 0.85:
        return None
    authors = [html.unescape(a) for a in re.findall(r'<meta name="citation_author" content="([^"]+)"', page)]
    date = re.search(r'<meta name="citation_date" content="(\d{4})', page)
    return dict(title=t, arxiv=aid, year=int(date.group(1)) if date else None, authors=authors,
                venue=f"arXiv:{aid}", score=round(sim(t, title), 3), source="arxiv")


def bibtex(key: str, m: dict) -> str:
    authors = " and ".join(m["authors"][:12]) + (" and others" if len(m["authors"]) > 12 else "")
    fields = {"title": "{" + m["title"] + "}", "author": authors, "year": str(m["year"])}
    if m["source"] == "arxiv":
        fields |= {"journal": f"arXiv preprint arXiv:{m['arxiv']}", "eprint": m["arxiv"], "archivePrefix": "arXiv"}
        kind = "article"
    else:
        fields |= {"journal": m["venue"], "doi": m["doi"]}
        if m.get("volume"):
            fields["volume"] = m["volume"]
        if m.get("pages"):
            fields["pages"] = m["pages"].replace("-", "--")
        kind = "article"
    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {v}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}\n"


def main() -> None:
    verified, failed = {}, []
    for key, title, kind, aid in CANDIDATES:
        try:
            m = arxiv(aid, title) if kind == "arxiv" else crossref(title)
        except Exception as exc:  # network hiccup: record, do not invent
            m = None
            print("ERR", key, exc)
        time.sleep(0.4)
        if m:
            verified[key] = m
            print("OK  ", key, m["year"], m["title"][:70])
        else:
            failed.append(key)
            print("FAIL", key)
    (HERE / "references_verified.json").write_text(json.dumps(verified, indent=1))
    (HERE / "references.bib").write_text("".join(bibtex(k, m) for k, m in verified.items()))
    print("failed:", failed)


if __name__ == "__main__":
    main()
