"""
Ablation evaluation: does retrieval grounding reduce hallucination?

    python scripts/evaluate_ablation.py --runs 20

Generates a grounded report (Chain of Evidence + retrieved FATF typology) and an
ungrounded baseline (scores only) for the same transaction, then measures how
much of each report is traceable to evidence that was actually supplied.

WHY THESE METRICS
-----------------
"Hallucination" is only meaningful if it is checkable. Asking a language model
to grade another model's output introduces the same failure mode it is meant to
detect, so every measure here is a deterministic string or numeric check
against the exact evidence that went into the prompt.

  numeric fidelity      Every figure in a forensic report should be one that was
                        supplied. Each number in the report is matched against
                        the scores, percentages and thresholds present in the
                        prompt. Anything unmatched was invented.

  typology grounding    The grounded arm is given one FATF typology and told to
                        cite its ID. The baseline is given none, so any FATF
                        identifier it produces is fabricated outright.

  pattern support       Named laundering patterns (hub-and-spoke, smurfing,
                        layering …) asserted without appearing in the retrieved
                        typology text are unsupported claims.

  missing-modality      Rule 5 requires an unavailable modality be flagged, not
  handling              estimated. Inventing a score for a model that did not
                        respond is the most consequential hallucination in this
                        system, because an investigator cannot tell it happened.

Lower fabrication counts and higher fidelity in the grounded arm is the result
the architecture predicts. The script reports whatever it finds, including a
null or negative result.

SCOPE
-----
This measures traceability to supplied evidence. It does not measure whether a
report is well written, useful, or correct about the world — only whether every
claim in it can be sourced. That is the specific property Chain of Evidence
prompting is designed to enforce, and the property a regulator would test.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402

# Laundering patterns a report might assert. Checked against the retrieved
# typology text: asserting one that is not there is an unsupported claim.
KNOWN_PATTERNS = [
    "hub-and-spoke", "hub and spoke", "smurfing", "structuring", "layering",
    "placement", "integration", "account takeover", "mule", "money mule",
    "shell company", "trade-based", "round-tripping", "cuckoo smurfing",
    "wire stripping", "funnel account",
]

# A FATF identifier in any of the shapes the corpus and prompts use.
TYPOLOGY_ID_RE = re.compile(r"\b(?:FATF[-_ ]?\d+|TY[-_]\d+[A-Z_]*)\b", re.IGNORECASE)

# Numbers as they appear in prose: 0.8734, 87.34%, 87%, 1,234.56
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|\b\d*\.\d+%?")

MODALITY_WORDS = {
    "graph": ["graph", "network", "graphsage", "relational", "topology"],
    "behavioral": ["behaviour", "behavior", "vae", "dsaa", "baseline"],
    "temporal": ["temporal", "timing", "tcn", "burstiness", "velocity"],
}

# Phrases that satisfy rule 5 — an explicit acknowledgement of missing data.
UNAVAILABLE_MARKERS = [
    "data unavailable", "unavailable", "timed out", "timeout", "not available",
    "did not respond", "no data", "missing",
]


@dataclass
class ArmResult:
    """Measurements for one report."""

    arm: str
    numbers_total: int = 0
    numbers_grounded: int = 0
    numbers_fabricated: int = 0
    fabricated_examples: list = field(default_factory=list)

    typology_ids_cited: list = field(default_factory=list)
    typology_ids_fabricated: list = field(default_factory=list)

    patterns_asserted: list = field(default_factory=list)
    patterns_unsupported: list = field(default_factory=list)

    missing_modalities: list = field(default_factory=list)
    missing_flagged: list = field(default_factory=list)
    missing_fabricated: list = field(default_factory=list)

    report_chars: int = 0

    @property
    def numeric_fidelity(self) -> float | None:
        if self.numbers_total == 0:
            return None
        return self.numbers_grounded / self.numbers_total


def _normalise_number(token: str) -> list[float]:
    """
    A token may legitimately mean more than one value: '87%' matches evidence
    stored as either 0.87 or 87. Both readings are returned so a correct
    citation is never counted as fabricated on a formatting technicality.
    """
    raw = token.strip().rstrip("%").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return []
    if token.strip().endswith("%"):
        return [value, value / 100.0]
    return [value, value * 100.0]


def _build_evidence_values(case: dict) -> set[float]:
    """Every numeric value the prompt actually supplied for this arm."""
    values: set[float] = set()

    def add(v):
        if v is None:
            return
        values.add(round(float(v), 4))
        values.add(round(float(v) * 100, 2))

    add(case.get("confidence_score"))
    for key in ("graph_score", "behavioral_score", "temporal_score"):
        if case.get(f"{key.split('_')[0]}_available", True):
            add(case.get(key))
    add(case.get("similarity_score"))

    # Thresholds stated verbatim in the prompt template, and the modality count.
    for literal in (0.80, 0.50, 3, 1, 2, 5):
        add(literal)

    # Figures inside the retrieved typology are supplied evidence too. Quoting a
    # threshold from the source document is exactly the grounding behaviour we
    # are testing for, so counting it as invented would penalise the arm for
    # working. The baseline receives no document and so gets no such values.
    document = case.get("typology_document")
    if document:
        for token in NUMBER_RE.findall(document):
            for candidate in _normalise_number(token):
                values.add(round(candidate, 4))

    return values


def _strip_identifiers(report: str) -> str:
    """
    Remove identifier tokens before extracting numeric claims.

    The digits inside FATF-004 or EVAL_007 are part of a name, not a figure the
    report is asserting. Left in place they are counted as fabricated numbers —
    and because rule 3 requires the typology ID be cited, the grounded arm was
    penalised several times per report for following its own instructions. That
    inverted the result: the first run showed the baseline as more faithful,
    which was an artefact of this, not a property of the system.
    """
    cleaned = TYPOLOGY_ID_RE.sub(" ", report)
    cleaned = re.sub(r"\b[A-Z]{2,}[-_]\d+[A-Z_-]*\b", " ", cleaned)  # EVAL_007, TX_12
    cleaned = re.sub(r"\bC\d{6,}\b", " ", cleaned)                    # account numbers
    cleaned = re.sub(r"\bSECTION\s+\d\b", " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _measure_numbers(report: str, evidence: set[float], result: ArmResult) -> None:
    for token in NUMBER_RE.findall(_strip_identifiers(report)):
        candidates = _normalise_number(token)
        if not candidates:
            continue

        # Section numbers and years are structural, not claims.
        if token.strip() in {"1", "2", "3", "4", "5"} or re.fullmatch(r"20\d\d", token.strip()):
            continue

        result.numbers_total += 1
        # Tolerance absorbs rounding: a prompt shows 0.8734, prose may say 87.3%.
        matched = any(
            any(abs(c - e) <= max(0.02, abs(e) * 0.01) for e in evidence)
            for c in candidates
        )
        if matched:
            result.numbers_grounded += 1
        else:
            result.numbers_fabricated += 1
            if len(result.fabricated_examples) < 8:
                result.fabricated_examples.append(token.strip())


def _measure_typology(report: str, permitted_id: str | None, result: ArmResult) -> None:
    for match in TYPOLOGY_ID_RE.findall(report):
        cited = match.upper().replace("_", "-").replace(" ", "-")
        result.typology_ids_cited.append(match)
        if permitted_id is None:
            # Baseline was given no typology: any identifier is invented.
            result.typology_ids_fabricated.append(match)
        else:
            allowed = permitted_id.upper().replace("_", "-")
            if cited not in allowed and allowed not in cited:
                result.typology_ids_fabricated.append(match)


def _measure_patterns(report: str, typology_text: str | None, result: ArmResult) -> None:
    lowered = report.lower()
    source = (typology_text or "").lower()
    for pattern in KNOWN_PATTERNS:
        if pattern not in lowered:
            continue
        result.patterns_asserted.append(pattern)
        if pattern not in source:
            result.patterns_unsupported.append(pattern)


def _measure_missing_modalities(report: str, case: dict, result: ArmResult) -> None:
    lowered = report.lower()
    for modality in ("graph", "behavioral", "temporal"):
        if case.get(f"{modality}_available", True):
            continue
        result.missing_modalities.append(modality)

        # Look at the sentences mentioning this modality and decide whether the
        # report acknowledged the gap or filled it in.
        sentences = [
            s for s in re.split(r"[.\n]", lowered)
            if any(w in s for w in MODALITY_WORDS[modality])
        ]
        context = " ".join(sentences)

        if any(marker in context for marker in UNAVAILABLE_MARKERS):
            result.missing_flagged.append(modality)
        elif NUMBER_RE.search(context):
            # A figure attached to a model that never responded.
            result.missing_fabricated.append(modality)


def evaluate_report(report: str, case: dict, arm: str, grounded: bool) -> ArmResult:
    result = ArmResult(arm=arm, report_chars=len(report))

    _measure_numbers(report, _build_evidence_values(case), result)
    _measure_typology(report, case.get("typology_id") if grounded else None, result)
    _measure_patterns(report, case.get("typology_document") if grounded else None, result)
    _measure_missing_modalities(report, case, result)

    return result


async def run_case(scenario: str, index: int) -> dict | None:
    """Run one transaction through both arms and measure each."""
    from backend.fusion_engine import MetaClassifier
    from backend.llm.forensic_reporter import ForensicReporter, create_llm_backend
    from backend.mock_scores import FraudScenario, generate_mock_scores
    from backend.rag.knowledge_base import FATFKnowledgeBase
    from backend.rag.prompt_builder import (
        UpstreamContext,
        build_baseline_prompt,
        build_chain_of_evidence_prompt,
    )
    from backend.rag.retriever import FATFRetriever

    kb = _shared["kb"]
    retriever = _shared["retriever"]
    classifier = _shared["classifier"]
    reporter = _shared["reporter"]

    mock = generate_mock_scores(scenario=FraudScenario(scenario))
    fusion = classifier.fuse(
        graph_score=mock.graph_score,
        behavioral_score=mock.behavioral_score,
        temporal_score=mock.temporal_score,
    )

    retrievals = retriever.retrieve(
        graph_score=fusion.graph_score,
        behavioral_score=fusion.behavioral_score,
        temporal_score=fusion.temporal_score,
        confidence_score=fusion.confidence_score,
    )
    if not retrievals:
        return None
    top = retrievals[0]

    transaction_id = f"EVAL_{index:03d}"
    case = {
        "transaction_id": transaction_id,
        "scenario": scenario,
        "confidence_score": fusion.confidence_score,
        "graph_score": fusion.graph_score,
        "behavioral_score": fusion.behavioral_score,
        "temporal_score": fusion.temporal_score,
        "graph_available": fusion.graph_available,
        "behavioral_available": fusion.behavioral_available,
        "temporal_available": fusion.temporal_available,
        "similarity_score": top.similarity_score,
        "typology_id": top.typology_id,
        "typology_name": top.typology_name,
        "typology_document": top.document,
    }

    grounded_prompt = build_chain_of_evidence_prompt(
        transaction_id=transaction_id,
        graph_score=fusion.graph_score,
        behavioral_score=fusion.behavioral_score,
        temporal_score=fusion.temporal_score,
        confidence_score=fusion.confidence_score,
        graph_available=fusion.graph_available,
        behavioral_available=fusion.behavioral_available,
        temporal_available=fusion.temporal_available,
        retrieval=top,
        upstream_context=UpstreamContext(),
    )
    baseline_prompt = build_baseline_prompt(
        transaction_id=transaction_id,
        graph_score=fusion.graph_score,
        behavioral_score=fusion.behavioral_score,
        temporal_score=fusion.temporal_score,
        confidence_score=fusion.confidence_score,
    )

    try:
        grounded_text = await _generate_with_retry(reporter, grounded_prompt)
        baseline_text = await _generate_with_retry(reporter, baseline_prompt)
    except QuotaExhausted:
        raise
    except Exception as e:
        print(f"generation failed: {type(e).__name__}: {e}")
        return None

    return {
        "case": {k: v for k, v in case.items() if k != "typology_document"},
        "grounded": asdict(evaluate_report(grounded_text, case, "grounded", grounded=True)),
        "baseline": asdict(evaluate_report(baseline_text, case, "baseline", grounded=False)),
        "reports": {"grounded": grounded_text, "baseline": baseline_text},
    }


class QuotaExhausted(Exception):
    """The daily request allowance is spent; no amount of waiting helps today."""


def _retry_after_seconds(message: str) -> float | None:
    match = re.search(r"retry in ([\d.]+)s", message, re.IGNORECASE)
    return float(match.group(1)) if match else None


async def _generate_with_retry(reporter, prompt, attempts: int = 3):
    """
    Generate, backing off on a rate limit.

    The Gemini free tier allows 20 generate_content calls per day and throttles
    per minute. A per-minute throttle is worth waiting out; a daily cap is not,
    so the two are distinguished and the daily one stops the run rather than
    sleeping pointlessly.
    """
    last: Exception | None = None

    for attempt in range(attempts):
        try:
            return await asyncio.to_thread(reporter.generate_report, prompt)
        except Exception as e:
            last = e
            text = str(e)
            if "429" not in text and "ResourceExhausted" not in type(e).__name__:
                raise

            if "per_day" in text or "free_tier_requests" in text:
                raise QuotaExhausted(
                    "Daily Gemini quota exhausted (free tier allows 20 calls/day, "
                    "and each case uses 2). Results so far have been saved — "
                    "re-run tomorrow, or use a paid key to evaluate more cases."
                ) from e

            wait = _retry_after_seconds(text) or (5 * (attempt + 1))
            if attempt < attempts - 1:
                print(f"rate limited, waiting {wait:.0f}s… ", end="", flush=True)
                await asyncio.sleep(wait + 1)

    raise last if last else RuntimeError("generation failed")


_shared: dict = {}


def _init_components() -> bool:
    from backend.fusion_engine import MetaClassifier
    from backend.llm.forensic_reporter import ForensicReporter, create_llm_backend
    from backend.rag.knowledge_base import FATFKnowledgeBase
    from backend.rag.retriever import FATFRetriever

    kb = FATFKnowledgeBase(
        chroma_db_path=config.get("paths", "chroma_db"),
        fatf_data_path=config.get("paths", "fatf_data"),
    )
    kb.initialize()

    _shared["kb"] = kb
    _shared["retriever"] = FATFRetriever(
        collection=kb.get_collection(), embedder=kb.get_embedder(), top_k=1
    )

    classifier = MetaClassifier(model_save_path=config.get("paths", "meta_classifier"))
    classifier.initialize()
    _shared["classifier"] = classifier

    try:
        _shared["reporter"] = ForensicReporter(backend=create_llm_backend())
    except ValueError as e:
        print(f"No language model configured: {e}")
        return False
    return True


def summarise(results: list[dict]) -> dict:
    def collect(arm: str, field_name: str) -> list:
        return [r[arm][field_name] for r in results]

    def mean_or_none(values: list) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.mean(clean) if clean else None

    summary = {}
    for arm in ("grounded", "baseline"):
        fidelities = []
        for r in results:
            total = r[arm]["numbers_total"]
            if total:
                fidelities.append(r[arm]["numbers_grounded"] / total)

        summary[arm] = {
            "cases": len(results),
            "numeric_fidelity_mean": mean_or_none(fidelities),
            "fabricated_numbers_total": sum(collect(arm, "numbers_fabricated")),
            "fabricated_numbers_per_report": mean_or_none(collect(arm, "numbers_fabricated")),
            "fabricated_typology_ids_total": sum(
                len(x) for x in collect(arm, "typology_ids_fabricated")
            ),
            "reports_citing_fabricated_typology": sum(
                1 for x in collect(arm, "typology_ids_fabricated") if x
            ),
            "unsupported_patterns_total": sum(
                len(x) for x in collect(arm, "patterns_unsupported")
            ),
            "reports_with_unsupported_patterns": sum(
                1 for x in collect(arm, "patterns_unsupported") if x
            ),
            "missing_modality_instances": sum(len(x) for x in collect(arm, "missing_modalities")),
            "missing_flagged": sum(len(x) for x in collect(arm, "missing_flagged")),
            "missing_fabricated": sum(len(x) for x in collect(arm, "missing_fabricated")),
            "mean_report_chars": mean_or_none(collect(arm, "report_chars")),
        }

    return summary


def render(summary: dict, results: list[dict]) -> str:
    g, b = summary["grounded"], summary["baseline"]

    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"

    def num(v):
        return "—" if v is None else f"{v:.2f}"

    small_sample = g["cases"] < 10

    lines = [
        "# Ablation: retrieval grounding vs ungrounded generation",
        "",
        f"Cases: {g['cases']}  ·  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        f"Model: {config.get('llm', 'gemini_model')}",
        "",
    ]

    if small_sample:
        lines += [
            f"> **{g['cases']} cases is a small sample.** The direction is worth",
            "> reporting; the magnitudes are not stable. The Gemini free tier",
            "> allows 20 generations a day and each case uses two, so a larger",
            "> run needs either several days or a paid key.",
            "",
        ]

    lines += [
        "Every measure is a deterministic check against the evidence supplied in",
        "the prompt. No language model grades another model's output.",
        "",
        "| Measure | Grounded | Baseline | |",
        "|---|---:|---:|---|",
    ]

    def row(label, gv, bv, better_when_lower=True, fmt=num):
        if gv is None or bv is None:
            verdict = ""
        elif gv == bv:
            verdict = "no difference"
        else:
            improved = (gv < bv) if better_when_lower else (gv > bv)
            verdict = "grounded better" if improved else "**baseline better**"
        return f"| {label} | {fmt(gv)} | {fmt(bv)} | {verdict} |"

    lines += [
        row("Numeric fidelity (higher better)", g["numeric_fidelity_mean"],
            b["numeric_fidelity_mean"], better_when_lower=False, fmt=pct),
        row("Fabricated figures per report", g["fabricated_numbers_per_report"],
            b["fabricated_numbers_per_report"]),
        row("Reports citing a fabricated typology ID",
            g["reports_citing_fabricated_typology"], b["reports_citing_fabricated_typology"],
            fmt=lambda v: str(int(v)) if v is not None else "—"),
        row("Reports asserting unsupported patterns",
            g["reports_with_unsupported_patterns"], b["reports_with_unsupported_patterns"],
            fmt=lambda v: str(int(v)) if v is not None else "—"),
    ]

    lines += [
        "",
        "## Unavailable modalities",
        "",
        "When a model does not respond, the report must say so rather than",
        "estimate. Inventing a score here is the most serious failure in this",
        "system: an investigator cannot tell it happened.",
        "",
        "| | Grounded | Baseline |",
        "|---|---:|---:|",
        f"| Instances of a missing modality | {g['missing_modality_instances']} | {b['missing_modality_instances']} |",
        f"| Correctly flagged as unavailable | {g['missing_flagged']} | {b['missing_flagged']} |",
        f"| Score invented for it | {g['missing_fabricated']} | {b['missing_fabricated']} |",
        "",
    ]

    worst = sorted(results, key=lambda r: r["baseline"]["numbers_fabricated"], reverse=True)[:3]
    if worst and worst[0]["baseline"]["numbers_fabricated"] > 0:
        lines += ["## Examples of ungrounded figures in the baseline", ""]
        for r in worst:
            ex = r["baseline"]["fabricated_examples"]
            if not ex:
                continue
            lines.append(
                f"- `{r['case']['transaction_id']}` ({r['case']['scenario']}): "
                + ", ".join(f"`{e}`" for e in ex[:6])
            )
        lines.append("")

    lines += [
        "## Reading this",
        "",
        "Numeric fidelity is the share of figures in a report that match a value",
        "supplied in its prompt. A fabricated typology ID means the report cited",
        "an FATF identifier it was never given. An unsupported pattern is a named",
        "laundering technique asserted without appearing in the retrieved",
        "typology text.",
        "",
        "This measures traceability, not writing quality or real-world",
        "correctness — only whether each claim can be sourced. That is the",
        "property Chain of Evidence prompting enforces, and the one a regulator",
        "would test.",
        "",
    ]
    return "\n".join(lines)


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=int, default=12, help="cases to evaluate")
    p.add_argument("--out", type=Path, default=Path("data/evaluation"))
    p.add_argument(
        "--scenarios",
        default="mule_network,layering,smurfing,account_takeover,velocity_fraud,legitimate",
    )
    args = p.parse_args()

    print("Initialising components…")
    if not _init_components():
        return 1

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    print(f"Evaluating {args.runs} cases across {len(scenarios)} scenarios.")
    print("Two reports per case, so this takes a while.\n")

    results = []
    stopped_early = None

    for i in range(args.runs):
        scenario = scenarios[i % len(scenarios)]
        print(f"  [{i + 1}/{args.runs}] {scenario}… ", end="", flush=True)
        try:
            result = await run_case(scenario, i + 1)
        except QuotaExhausted as e:
            print("\n\n" + str(e))
            stopped_early = str(e)
            break

        if result:
            results.append(result)
            gf = result["grounded"]["numbers_fabricated"]
            bf = result["baseline"]["numbers_fabricated"]
            print(f"fabricated figures: grounded {gf}, baseline {bf}")
        else:
            print("skipped")

    if not results:
        print("\nNo cases completed. Check that the language model is reachable.")
        return 1

    if stopped_early:
        print(f"\nSaving the {len(results)} case(s) completed before stopping.\n")

    summary = summarise(results)
    report = render(summary, results)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    (args.out / f"ablation-{stamp}.md").write_text(report, encoding="utf-8")
    (args.out / f"ablation-{stamp}.json").write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2, default=str),
        encoding="utf-8",
    )
    (args.out / "ablation-latest.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nWritten to {args.out}/ablation-{stamp}.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
