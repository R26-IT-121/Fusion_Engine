"""
Chain of Evidence Prompt Builder
Constructs a strict, grounded prompt that forces the LLM to generate forensic
narratives anchored exclusively in the provided numerical scores and retrieved
FATF typology — eliminating free-form hallucination.

When upstream APIs provide rich forensic signals (M1 anomaly fingerprints,
M2 subgraph patterns, M3 burstiness), those are injected as additional
grounded evidence — still under the same anti-hallucination rules.
"""

from dataclasses import dataclass
from typing import Optional

from backend.rag.retriever import RetrievalResult


@dataclass
class UpstreamContext:
    """Rich forensic signals from upstream model APIs — injected into LLM prompt."""
    behavioral_signal_summary: Optional[str] = None   # M1 VAE fraud_signal_summary
    graph_signal_summary: Optional[str] = None         # M2 GraphSAGE subgraph pattern
    temporal_signal_summary: Optional[str] = None      # M3 TCN burstiness / predecessor


@dataclass
class ForensicPromptPackage:
    system_prompt: str
    user_prompt: str


def build_chain_of_evidence_prompt(
    transaction_id: str,
    graph_score: float,
    behavioral_score: float,
    temporal_score: float,
    confidence_score: float,
    graph_available: bool,
    behavioral_available: bool,
    temporal_available: bool,
    retrieval: RetrievalResult,
    upstream_context: Optional[UpstreamContext] = None,
    classification: Optional[str] = None,
) -> ForensicPromptPackage:
    """
    Builds the two-part prompt (system + user) for the LLM forensic analyst.
    The system prompt hard-codes the Chain of Evidence constraint rules.
    The user prompt injects the case-specific numerical evidence, retrieved
    FATF typology, and any rich signals from upstream models.
    """

    system_prompt = """You are a Senior Financial Forensic Analyst at a regulatory compliance unit.
Your task is to generate a structured, legally admissible Case Investigation Report.

MANDATORY CHAIN OF EVIDENCE RULES — you MUST follow all of these without exception:
1. You may ONLY cite the numerical scores provided in the CASE DATA section below. Do NOT invent, estimate, or approximate any other figures.
2. You may ONLY reference fraud patterns that are explicitly documented in the RETRIEVED FATF TYPOLOGY section below. Do NOT introduce fraud patterns not present in that section.
3. You MUST cite the FATF Typology ID (e.g., FATF-002) when referencing any crime pattern.
4. You MUST structure your report using the exact five-section format specified in the user message.
5. You MUST flag any modality score that was unavailable as "DATA UNAVAILABLE — modality timed out" rather than estimating it.
6. If UPSTREAM FORENSIC SIGNALS are provided, you MUST reference them in Section 2 as additional grounded evidence. Do NOT paraphrase them — quote them directly.
7. Do NOT use vague language such as "possibly", "might", or "could indicate". Use definitive analytical language grounded in the provided evidence.
8. Do NOT add recommendations, disclaimers, or commentary outside the five-section report format.
9. Your report must be suitable for submission to a regulatory body and withstand legal scrutiny."""

    # Build modality status block
    modality_lines = []
    if graph_available:
        modality_lines.append(
            f"  - Graph Network Analysis Score:    {graph_score:.4f} ({graph_score:.1%})"
        )
    else:
        modality_lines.append(
            "  - Graph Network Analysis Score:    DATA UNAVAILABLE — modality timed out"
        )

    if behavioral_available:
        modality_lines.append(
            f"  - Behavioral Anomaly Score:        {behavioral_score:.4f} ({behavioral_score:.1%})"
        )
    else:
        modality_lines.append(
            "  - Behavioral Anomaly Score:        DATA UNAVAILABLE — modality timed out"
        )

    if temporal_available:
        modality_lines.append(
            f"  - Temporal Pattern Analysis Score: {temporal_score:.4f} ({temporal_score:.1%})"
        )
    else:
        modality_lines.append(
            "  - Temporal Pattern Analysis Score: DATA UNAVAILABLE — modality timed out"
        )

    modality_block = "\n".join(modality_lines)
    available_count = sum([graph_available, behavioral_available, temporal_available])
    missing_note = (
        ""
        if available_count == 3
        else f"\n  NOTE: {3 - available_count} modality/modalities unavailable. Confidence score computed on {available_count}/3 inputs."
    )

    # Build upstream forensic signals block (only if any signals were returned)
    upstream_block = ""
    if upstream_context is not None:
        signals = []
        if upstream_context.behavioral_signal_summary:
            signals.append(
                f"  [VAE/DSAA — Behavioral Module]\n  {upstream_context.behavioral_signal_summary}"
            )
        if upstream_context.graph_signal_summary:
            signals.append(
                f"  [GraphSAGE — Network Intelligence Module]\n  {upstream_context.graph_signal_summary}"
            )
        if upstream_context.temporal_signal_summary:
            signals.append(
                f"  [TCN/TSCFD — Temporal Analysis Module]\n  {upstream_context.temporal_signal_summary}"
            )
        if signals:
            upstream_block = (
                "\n\n══════════════════════════════════════════════════════\n"
                "UPSTREAM FORENSIC SIGNALS (cite directly in Section 2)\n"
                "══════════════════════════════════════════════════════\n"
                + "\n\n".join(signals)
            )

    # The system has already classified this transaction; handing the model the
    # answer removes a hallucination surface (it cannot disagree with our own
    # banding) and removes the instruction text it used to echo verbatim into
    # the finished report — "Classification: LOW — derive from confidence score".
    classification_line = classification or "[CRITICAL / HIGH / MEDIUM / LOW]"

    user_prompt = f"""Generate a forensic case investigation report using ONLY the evidence below.

══════════════════════════════════════════════════════
CASE DATA
══════════════════════════════════════════════════════
Transaction ID:         {transaction_id}
Fused Fraud Confidence: {confidence_score:.4f} ({confidence_score:.1%})

Sub-Model Risk Scores:
{modality_block}{missing_note}

Typology Match Similarity: {retrieval.similarity_score:.2%}
{upstream_block}

══════════════════════════════════════════════════════
RETRIEVED FATF TYPOLOGY (your ONLY permitted crime pattern reference)
══════════════════════════════════════════════════════
{retrieval.document}
══════════════════════════════════════════════════════

Generate the report in EXACTLY this five-section format:

---
CASE INVESTIGATION REPORT
Transaction ID: {transaction_id}
Classification: {classification_line}
FATF Typology Match: [Typology Name] ({retrieval.typology_id}) — {retrieval.similarity_score:.1%} similarity

SECTION 1 — EXECUTIVE SUMMARY
[2–3 sentences. State the overall fraud confidence score, the dominant risk modality, and the matched FATF typology. Cite all figures precisely.]

SECTION 2 — MULTI-MODAL EVIDENCE ANALYSIS
[Analyze each available sub-model score individually. For each: state the score, interpret what it indicates, and link it to the retrieved FATF typology indicators. If UPSTREAM FORENSIC SIGNALS were provided, quote them here as supporting evidence. Mark unavailable modalities as timed out.]

SECTION 3 — TYPOLOGY GROUNDING
[Explain how the numerical evidence pattern matches the retrieved FATF typology. Cite specific behavioral indicators from the RETRIEVED FATF TYPOLOGY section that are supported by the sub-model scores. Reference the Typology ID.]

SECTION 4 — FORENSIC CONFIDENCE ASSESSMENT
[State the fused confidence score. Explain the ensemble fusion basis. State whether the retrieval similarity score ({retrieval.similarity_score:.1%}) is sufficient to ground the typology match. Note any data limitations from unavailable modalities.]

SECTION 5 — INVESTIGATIVE RECOMMENDATION
[State whether the case should be ESCALATED FOR IMMEDIATE REVIEW, FLAGGED FOR STANDARD REVIEW, or DISMISSED WITH MONITORING. Base this strictly on the confidence score threshold: >0.80 = escalate, 0.50–0.80 = standard review, <0.50 = dismiss with monitoring.]
---"""

    return ForensicPromptPackage(system_prompt=system_prompt, user_prompt=user_prompt)


def build_baseline_prompt(
    transaction_id: str,
    graph_score: float,
    behavioral_score: float,
    temporal_score: float,
    confidence_score: float,
) -> ForensicPromptPackage:
    """
    Builds a prompt WITHOUT any retrieved FATF typology context.
    Used as the ablation baseline to demonstrate hallucination in ungrounded LLM generation.
    The LLM receives only numerical scores and must generate a forensic narrative freely.
    """
    system_prompt = (
        "You are a Senior Financial Forensic Analyst. "
        "Generate a structured Case Investigation Report based on the numerical risk scores provided."
    )

    user_prompt = f"""Generate a forensic case investigation report for the following transaction.

CASE DATA:
Transaction ID:         {transaction_id}
Fused Fraud Confidence: {confidence_score:.4f} ({confidence_score:.1%})
Graph Network Score:    {graph_score:.4f} ({graph_score:.1%})
Behavioral Score:       {behavioral_score:.4f} ({behavioral_score:.1%})
Temporal Score:         {temporal_score:.4f} ({temporal_score:.1%})

Write a report with these five sections:

SECTION 1 — EXECUTIVE SUMMARY
SECTION 2 — MULTI-MODAL EVIDENCE ANALYSIS
SECTION 3 — PATTERN ASSESSMENT
SECTION 4 — CONFIDENCE ASSESSMENT
SECTION 5 — INVESTIGATIVE RECOMMENDATION"""

    return ForensicPromptPackage(system_prompt=system_prompt, user_prompt=user_prompt)


# ── Suspicious Activity Report drafting ──────────────────────────────────────

SAR_SYSTEM_PROMPT = """You are drafting a Suspicious Activity Report (SAR) for review by a human compliance officer.

You are NOT filing anything. You are producing a draft that a named officer will read, edit and decide upon. Write accordingly.

ABSOLUTE CONSTRAINTS
1. Use ONLY the case data supplied below. Every account identifier, amount, score, date and typology name you write must appear verbatim in that data.
2. If a field a SAR would normally contain was not supplied, write "Not available in the source record." Do NOT infer it, estimate it, or leave it out silently.
3. Never invent a customer name, address, account-holder identity, occupation, or any KYC detail. None of that is in the source data.
4. Do not state that fraud occurred. State what was observed and why it was flagged. The determination is the officer's to make, not yours.
5. Do not recommend filing or not filing. Present the evidence; the decision is a human judgement.
6. Attribute every risk score to the model that produced it, and say plainly when a model was unavailable.
7. Write in plain, factual, non-emotive language suitable for a regulator. No speculation about intent or criminality.

Produce exactly the six sections requested, and nothing else."""


def build_sar_prompt(record: dict) -> ForensicPromptPackage:
    """Draft a SAR from one persisted analysis record.

    Takes the stored record rather than a live pipeline result on purpose: a
    filing must describe what the system actually concluded at the time, and
    re-running the models could produce a different answer against a graph that
    has since moved on.

    `record` is the row from `analysis_records`, so every value here has already
    been through the pipeline and persisted. Nothing new is computed.
    """
    def val(key: str, default: str = "Not available in the source record.") -> str:
        v = record.get(key)
        return default if v is None or v == "" else str(v)

    def score_line(name: str, score_key: str, avail_key: str) -> str:
        if not record.get(avail_key):
            return f"  {name}: model unavailable for this transaction — excluded from the fused score."
        s = record.get(score_key)
        return f"  {name}: {s:.4f}" if isinstance(s, (int, float)) else f"  {name}: {val(score_key)}"

    confidence = record.get("fraud_confidence_score")
    confidence_str = f"{confidence:.4f}" if isinstance(confidence, (int, float)) else val("fraud_confidence_score")

    amount = record.get("amount")
    amount_str = f"{amount:,.2f}" if isinstance(amount, (int, float)) else val("amount")

    similarity = record.get("similarity_score")
    similarity_str = f"{similarity:.1%}" if isinstance(similarity, (int, float)) else val("similarity_score")

    user_prompt = f"""Draft a Suspicious Activity Report from the case data below.

══════════════════════════════════════════════════════
CASE DATA — the only permitted source of fact
══════════════════════════════════════════════════════
Internal reference:   {val('transaction_id')}
Detected at:          {val('created_at')}
Screened by:          {val('analysed_by', 'Automated monitoring')}

SUBJECT ACCOUNTS
  Originating account: {val('name_orig')}
  Receiving account:   {val('name_dest')}

TRANSACTION
  Type:   {val('tx_type')}
  Amount: {amount_str}
  Period: {val('step')}

MODEL ASSESSMENT
  Fused confidence: {confidence_str}
  Classification:   {val('classification')}
  Modalities contributing: {val('modalities_used')} of 3
{score_line('Relational (graph network) score', 'graph_score', 'graph_available')}
{score_line('Behavioural score', 'behavioral_score', 'behavioral_available')}
{score_line('Temporal score', 'temporal_score', 'temporal_available')}

TYPOLOGY MATCH
  Name:       {val('typology_name')}
  Reference:  {val('typology_id')}
  Similarity: {similarity_str}

SUPPORTING ANALYSIS (produced by the detection system at the time of the alert)
{val('forensic_report', 'No forensic narrative was recorded for this alert.')}
══════════════════════════════════════════════════════

Produce exactly these six sections:

SECTION 1 — SUBJECT OF REPORT
[The accounts involved and their role in the observed activity. Identifiers only — no identity details were supplied and none may be invented.]

SECTION 2 — ACTIVITY OBSERVED
[What happened, factually: the transaction, amount, type and period. No characterisation of intent.]

SECTION 3 — REASON FOR SUSPICION
[Why the activity was flagged. Cite each contributing model score and what it measured. Name any model that was unavailable and state that the fused confidence was reduced accordingly.]

SECTION 4 — TYPOLOGY ASSESSMENT
[The matched typology, its similarity, and which observed features correspond to it. State explicitly that a typology match is a similarity measure and not a determination of criminal conduct.]

SECTION 5 — SUPPORTING EVIDENCE
[Enumerate the specific evidence relied on, each traceable to the case data above.]

SECTION 6 — MATTERS FOR REVIEWER ATTENTION
[What a compliance officer must establish before deciding: information absent from this record, KYC checks not performed by this system, and any limitation in the assessment. Be candid about what the system cannot know.]"""

    return ForensicPromptPackage(system_prompt=SAR_SYSTEM_PROMPT, user_prompt=user_prompt)


# ── Plain-English restatement ────────────────────────────────────────────────

PLAIN_SYSTEM_PROMPT = """You explain fraud alerts to people who are not analysts — a branch manager, a customer-service lead, a new joiner.

RULES
1. Use ONLY facts from the report you are given. Add nothing, and do not soften or strengthen the conclusion.
2. No jargon. If a term is unavoidable, define it in the same sentence.
3. Never say fraud has occurred. Say what was noticed and why it was unusual.
4. Say plainly which checks could not run, if the report says any could not.
5. Six sentences at most, in plain paragraphs. No headings, no bullet lists.

You are restating, not re-analysing."""


def build_plain_english_prompt(report: str, classification: str) -> ForensicPromptPackage:
    """Restate a forensic report for a non-specialist reader.

    A second pass over text the system already produced, rather than a second
    look at the evidence: the technical report stays the record, and this is a
    reading of it. That ordering matters — two independent generations from the
    same evidence could disagree, and then nobody knows which one is the finding.
    """
    user_prompt = f"""Restate the following fraud alert for someone with no technical background.

The system classified it as: {classification}

──────────────── REPORT ────────────────
{report}
────────────────────────────────────────

Explain, in plain language: what happened, what looked unusual about it, how
confident the system is, and what a person should do next. If the report says a
check could not run, say so."""
    return ForensicPromptPackage(
        system_prompt=PLAIN_SYSTEM_PROMPT, user_prompt=user_prompt
    )
