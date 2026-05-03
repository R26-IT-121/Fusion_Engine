"""
Chain of Evidence Prompt Builder
Constructs a strict, grounded prompt that forces the LLM to generate forensic
narratives anchored exclusively in the provided numerical scores and retrieved
FATF typology — eliminating free-form hallucination.
"""

from dataclasses import dataclass
from backend.rag.retriever import RetrievalResult


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
) -> ForensicPromptPackage:
    """
    Builds the two-part prompt (system + user) for the LLM forensic analyst.
    The system prompt hard-codes the Chain of Evidence constraint rules.
    The user prompt injects the case-specific numerical evidence and retrieved context.
    """

    system_prompt = """You are a Senior Financial Forensic Analyst at a regulatory compliance unit.
Your task is to generate a structured, legally admissible Case Investigation Report.

MANDATORY CHAIN OF EVIDENCE RULES — you MUST follow all of these without exception:
1. You may ONLY cite the numerical scores provided in the CASE DATA section below. Do NOT invent, estimate, or approximate any other figures.
2. You may ONLY reference fraud patterns that are explicitly documented in the RETRIEVED FATF TYPOLOGY section below. Do NOT introduce fraud patterns not present in that section.
3. You MUST cite the FATF Typology ID (e.g., FATF-002) when referencing any crime pattern.
4. You MUST structure your report using the exact five-section format specified in the user message.
5. You MUST flag any modality score that was unavailable as "DATA UNAVAILABLE — modality timed out" rather than estimating it.
6. Do NOT use vague language such as "possibly", "might", or "could indicate". Use definitive analytical language grounded in the provided evidence.
7. Do NOT add recommendations, disclaimers, or commentary outside the five-section report format.
8. Your report must be suitable for submission to a regulatory body and withstand legal scrutiny."""

    modality_status_lines = []
    if graph_available:
        modality_status_lines.append(
            f"  - Graph Network Analysis Score:    {graph_score:.4f} ({graph_score:.1%})"
        )
    else:
        modality_status_lines.append(
            "  - Graph Network Analysis Score:    DATA UNAVAILABLE — modality timed out"
        )

    if behavioral_available:
        modality_status_lines.append(
            f"  - Behavioral Anomaly Score:        {behavioral_score:.4f} ({behavioral_score:.1%})"
        )
    else:
        modality_status_lines.append(
            "  - Behavioral Anomaly Score:        DATA UNAVAILABLE — modality timed out"
        )

    if temporal_available:
        modality_status_lines.append(
            f"  - Temporal Pattern Analysis Score: {temporal_score:.4f} ({temporal_score:.1%})"
        )
    else:
        modality_status_lines.append(
            "  - Temporal Pattern Analysis Score: DATA UNAVAILABLE — modality timed out"
        )

    modality_block = "\n".join(modality_status_lines)
    available_count = sum([graph_available, behavioral_available, temporal_available])
    missing_note = (
        ""
        if available_count == 3
        else f"\n  NOTE: {3 - available_count} modality/modalities were unavailable. Confidence score computed on {available_count}/3 inputs."
    )

    user_prompt = f"""Generate a forensic case investigation report using ONLY the evidence below.

══════════════════════════════════════════════════════
CASE DATA
══════════════════════════════════════════════════════
Transaction ID:         {transaction_id}
Fused Fraud Confidence: {confidence_score:.4f} ({confidence_score:.1%})

Sub-Model Risk Scores:
{modality_block}{missing_note}

Typology Match Similarity: {retrieval.similarity_score:.2%}

══════════════════════════════════════════════════════
RETRIEVED FATF TYPOLOGY (your ONLY permitted crime pattern reference)
══════════════════════════════════════════════════════
{retrieval.document}
══════════════════════════════════════════════════════

Generate the report in EXACTLY this five-section format:

---
CASE INVESTIGATION REPORT
Transaction ID: {transaction_id}
Classification: [CRITICAL / HIGH / MEDIUM / LOW] — derive from confidence score
FATF Typology Match: [Typology Name] ({retrieval.typology_id}) — {retrieval.similarity_score:.1%} similarity

SECTION 1 — EXECUTIVE SUMMARY
[2–3 sentences. State the overall fraud confidence score, the dominant risk modality, and the matched FATF typology. Cite all figures precisely.]

SECTION 2 — MULTI-MODAL EVIDENCE ANALYSIS
[Analyze each available sub-model score individually. For each: state the score, interpret what it indicates, and link it to the retrieved FATF typology indicators. Mark unavailable modalities as timed out.]

SECTION 3 — TYPOLOGY GROUNDING
[Explain how the numerical evidence pattern matches the retrieved FATF typology. Cite specific behavioral indicators from the RETRIEVED FATF TYPOLOGY section that are supported by the sub-model scores. Reference the Typology ID.]

SECTION 4 — FORENSIC CONFIDENCE ASSESSMENT
[State the fused confidence score. Explain the ensemble fusion basis. State whether the retrieval similarity score ({retrieval.similarity_score:.1%}) is sufficient to ground the typology match. Note any data limitations from unavailable modalities.]

SECTION 5 — INVESTIGATIVE RECOMMENDATION
[State whether the case should be ESCALATED FOR IMMEDIATE REVIEW, FLAGGED FOR STANDARD REVIEW, or DISMISSED WITH MONITORING. Base this strictly on the confidence score threshold: >0.80 = escalate, 0.50–0.80 = standard review, <0.50 = dismiss with monitoring.]
---"""

    return ForensicPromptPackage(system_prompt=system_prompt, user_prompt=user_prompt)
