"""Build-integrity checks for the manuscript.

Three invariants, each of which has already caught a real error in this project:

1. **No hand-typed numbers.** Every reported quantity must be a macro from the
   generated ``numbers.tex``. A bare decimal anywhere else is a build failure.
   The seed-noise floor was hand-typed as 0.030 in an earlier report when the
   data said 0.225; this check exists so that cannot reach the paper.
2. **No undefined macros.** Every ``\\num...`` the manuscript uses must exist in
   ``numbers.tex``, so a renamed quantity fails loudly rather than rendering
   blank.
3. **Claims coverage.** Every claim id in the ledger is either mapped to a
   section in ``claims_map.md`` or explicitly listed as unused.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
RESULTS = Path(__file__).resolve().parents[1] / "results"
NUMBERS = PAPER / "numbers.tex"

#: Two or more digits after a decimal point: a reported measurement, not a
#: structural constant like f=0.5 or a section number.
BARE_DECIMAL = re.compile(r"(?<!\d)\d+\.\d{2,}")
MACRO_USE = re.compile(r"\\(num[A-Za-z]+)")
MACRO_DEF = re.compile(r"\\newcommand\{\\(num[A-Za-z]+)\}")

pytestmark = pytest.mark.skipif(not PAPER.exists(), reason="paper/ not present")


def tex_sources() -> list[Path]:
    return sorted(p for p in PAPER.rglob("*.tex") if p.name != "numbers.tex")


def test_numbers_file_exists() -> None:
    assert NUMBERS.exists(), "run scripts/make_numbers.py"


def test_no_hand_typed_numbers() -> None:
    """A bare decimal outside numbers.tex is a build failure."""
    offenders: list[str] = []
    for path in tex_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.split("%", 1)[0]  # ignore LaTeX comments
            for match in BARE_DECIMAL.finditer(stripped):
                offenders.append(f"{path.relative_to(PAPER)}:{lineno}: {match.group(0)!r}")
    assert not offenders, (
        "hand-typed numbers found; use a macro from numbers.tex instead:\n  "
        + "\n  ".join(offenders)
    )


def test_every_macro_used_is_defined() -> None:
    defined = set(MACRO_DEF.findall(NUMBERS.read_text(encoding="utf-8")))
    missing: dict[str, list[str]] = {}
    for path in tex_sources():
        for macro in MACRO_USE.findall(path.read_text(encoding="utf-8")):
            if macro not in defined:
                missing.setdefault(macro, []).append(str(path.relative_to(PAPER)))
    assert not missing, f"undefined number macros: {missing}"


def test_numbers_are_generated_not_edited() -> None:
    head = NUMBERS.read_text(encoding="utf-8")[:400]
    assert "GENERATED FILE" in head and "make_numbers.py" in head


class TestClaimsCoverage:
    def _claim_ids(self) -> set[str]:
        text = (RESULTS / "claims.md").read_text(encoding="utf-8")
        ids = set()
        for line in text.splitlines():
            if line.startswith("| ") and not line.startswith("| id"):
                # Strip markdown emphasis: the ledger bolds ids for readability
                # (`| **H2** |`), and an id is the same id with or without it.
                first = line.split("|")[1].strip().strip("*").strip()
                if first and not set(first) <= {"-", " "}:
                    ids.add(first)
        return ids

    def test_claims_map_exists(self) -> None:
        assert (PAPER / "claims_map.md").exists()

    def test_every_claim_is_mapped_or_declared_unused(self) -> None:
        mapped = (PAPER / "claims_map.md").read_text(encoding="utf-8")
        unaccounted = [c for c in self._claim_ids() if c not in mapped]
        assert not unaccounted, (
            f"claims in the ledger with no entry in claims_map.md: {sorted(unaccounted)}"
        )

    def test_evasion_anchor_is_the_tie(self) -> None:
        """The evasion story is anchored at f = 0.5 and must stay there.

        At f = 0.7 the Byzantine bloc is an outright majority, so every
        coherence level drives accuracy to zero and an evasion effect cannot be
        separated from numerical dominance. A 0.92 attacker gain read off that
        row was reported once and withdrawn; this keeps it withdrawn.
        """
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from aip.analysis.paper_numbers import EVASION_ANCHOR_F

        assert EVASION_ANCHOR_F == 0.5
        numbers = NUMBERS.read_text(encoding="utf-8")
        assert "\\newcommand{\\numEvasionAnchorF}{0.5}" in numbers, (
            "numbers.tex does not carry the f = 0.5 evasion anchor"
        )

    def test_no_claim_is_pending(self) -> None:
        """Task 6's contract: the ledger carries verdicts, not open questions.

        A "pending re-verification" row is a claim the manuscript may cite while
        nobody has checked it against the measurement floor. The audit that
        closed the last five (`scripts/task6_audit_pending.py`) overturned three
        of them, so the state is not allowed back without an explicit verdict.
        """
        text = (RESULTS / "claims.md").read_text(encoding="utf-8")
        # Inline code spans hold file names -- `pending_claims_audit.md` is the
        # report that closed the section, not a pending claim.
        prose = re.sub(r"`[^`]*`", "", text).upper()
        offenders = [ln.strip() for ln in prose.splitlines() if "PENDING" in ln]
        assert not offenders, (
            "results/claims.md has a pending claim again; re-audit it against the "
            "measurement floor before it can be cited:\n  " + "\n  ".join(offenders)
        )

    def test_pending_audit_report_is_present(self) -> None:
        report = RESULTS / "pending_claims_audit.md"
        assert report.exists(), "run scripts/task6_audit_pending.py"
        body = report.read_text(encoding="utf-8")
        for claim in ("## H2", "## F1", "## B1", "## G2", "## M2"):
            assert claim in body, f"audit report is missing {claim}"

    def test_retired_claim_ids_are_not_cited(self) -> None:
        """`H2` and `M2` were split; citing the bare id hides which half."""
        retired = {"H2", "M2"}
        cited: set[str] = set()
        pattern = re.compile(r"\\textbf\{\[([^\]]+)\]\}")
        for path in tex_sources():
            for group in pattern.findall(path.read_text(encoding="utf-8")):
                cited.update(part.strip() for part in group.split(","))
        assert not (cited & retired), (
            f"manuscript cites retired claim ids {sorted(cited & retired)}; "
            "use the split ids from results/claims.md"
        )

    def test_paper_claim_ids_exist_in_ledger(self) -> None:
        """A claim id cited in the manuscript must exist in the ledger."""
        known = self._claim_ids()
        cited: set[str] = set()
        pattern = re.compile(r"\\textbf\{\[([^\]]+)\]\}")
        for path in tex_sources():
            for group in pattern.findall(path.read_text(encoding="utf-8")):
                cited.update(part.strip() for part in group.split(","))
        unknown = sorted(c for c in cited if c and c not in known)
        assert not unknown, f"manuscript cites claim ids absent from the ledger: {unknown}"


def test_every_benchmark_has_a_latex_tag() -> None:
    """A benchmark with no macro tag fails the whole numbers build.

    This exact gap -- `KeyError: 'medqa'` -- killed a batch run four hours in,
    after every expensive phase had already succeeded, because the tag map was
    the NINTH place in this project carrying a hardcoded three-benchmark
    assumption. Checking it here turns a late runtime crash into a test failure.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from aip.analysis.paper_numbers import BENCHMARK_TAGS
    from aip.tasks import roster

    missing = [b for b in roster.benchmarks() if b not in BENCHMARK_TAGS]
    assert not missing, f"benchmarks with no LaTeX macro tag: {missing}"


class TestFigureProvenance:
    """A figure the manuscript cites must have been produced by this repository.

    Seven of the manuscript's figures spent this entire regeneration as L4-era
    PDFs from the initial upload, because the generators defaulted to `figures/`
    while LaTeX read `paper/figures/`. Nothing caught it: the files existed and
    the references resolved. There is now one figure directory, and every
    generator stamps `paper/figures/MANIFEST.json`.
    """

    def _cited(self) -> set[str]:
        pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
        cited: set[str] = set()
        for path in tex_sources():
            cited.update(pattern.findall(path.read_text(encoding="utf-8")))
        return cited

    def test_every_cited_figure_exists(self) -> None:
        missing = sorted(f for f in self._cited() if not (PAPER / "figures" / f).exists())
        assert not missing, f"figures cited by the manuscript but absent: {missing}"

    def test_every_cited_figure_is_stamped(self) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from aip.analysis.provenance import load

        entries = load(PAPER / "figures")
        assert entries, "paper/figures/MANIFEST.json is missing; rerun the generators"
        unstamped = sorted(f for f in self._cited() if f not in entries)
        assert not unstamped, (
            "figures with no provenance entry -- they may predate this hardware "
            f"regime (RULE 1): {unstamped}"
        )
        for name, entry in entries.items():
            assert entry.get("script", "").startswith("scripts/"), (
                f"{name} claims an unknown generator: {entry.get('script')!r}"
            )

    def test_no_second_figure_directory(self) -> None:
        stray = PAPER.parent / "figures"
        assert not stray.exists(), (
            "a second figures/ directory is back; the generators and LaTeX must "
            "read the same one or they drift"
        )
