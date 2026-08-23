"""
The detection pipeline, expressed as a stream of stage events.

Both /analyze and /analyze/stream drive this generator, so the streamed and
non-streamed paths cannot drift apart — the streaming endpoint forwards each
event, and the plain endpoint drains the generator and returns the final one.

Stage timings are measured, not simulated. The frontend renders what actually
happened, including the real latency of each upstream model.

The scoring, retrieval and LLM calls are synchronous and CPU/IO bound. They run
through asyncio.to_thread so the event loop stays free to flush each event as it
is produced — without that, every stage would complete before any bytes reached
the client and the "live" progress would be a fiction.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class Stage:
    INPUT = "input"
    MODELS = "models"
    FUSION = "fusion"
    RETRIEVAL = "retrieval"
    REPORT = "report"

    ORDER = (INPUT, MODELS, FUSION, RETRIEVAL, REPORT)


class Status:
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class StageEvent:
    stage: str
    status: str
    duration_ms: Optional[int] = None
    data: dict = field(default_factory=dict)
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineResult:
    """Terminal value — carries the assembled response payload."""

    payload: dict


class _Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self._start) * 1000)
        return False


async def run_pipeline(
    request,
    *,
    meta_classifier,
    retriever,
    forensic_reporter,
    fetch_upstream,
    classify,
) -> AsyncIterator[Any]:
    """
    Yields StageEvent objects as each stage starts and finishes, then a final
    PipelineResult.

    Dependencies are injected rather than imported so this module stays free of
    a cycle with main.py and can be driven directly in tests.
    """
    transaction_id = request.transaction_id or str(uuid.uuid4())
    mock_scenario_used = None
    behavioral_signal = graph_signal = temporal_signal = None

    # ── Stage 1: input ───────────────────────────────────────────────────────
    yield StageEvent(Stage.INPUT, Status.RUNNING)

    use_mock = request.use_mock or (
        request.transaction is None
        and request.graph_score is None
        and request.behavioral_score is None
        and request.temporal_score is None
    )

    tx = request.transaction
    input_data = {"transaction_id": transaction_id, "source": "mock" if use_mock else "live"}
    if tx is not None:
        input_data.update(
            {
                "type": tx.type,
                "amount": tx.amount,
                "nameOrig": tx.nameOrig,
                "nameDest": tx.nameDest,
                "oldbalanceOrg": tx.oldbalanceOrg,
                "newbalanceOrig": tx.newbalanceOrig,
                "step": tx.step,
            }
        )

    yield StageEvent(
        Stage.INPUT,
        Status.DONE,
        duration_ms=0,
        data=input_data,
        message=(
            "Simulated scenario — upstream models not called"
            if use_mock
            else "Transaction accepted"
        ),
    )

    # ── Stage 2: the three models ────────────────────────────────────────────
    yield StageEvent(
        Stage.MODELS,
        Status.RUNNING,
        message="Scoring with mock generator" if use_mock else "Calling three model APIs in parallel",
    )

    with _Timer() as t:
        if use_mock:
            from backend.mock_scores import FraudScenario, generate_mock_scores

            scenario = FraudScenario.RANDOM
            if request.mock_scenario:
                try:
                    scenario = FraudScenario(request.mock_scenario)
                except ValueError:
                    yield StageEvent(
                        Stage.MODELS,
                        Status.ERROR,
                        message=(
                            f"Unknown scenario '{request.mock_scenario}'. Valid: "
                            f"{[s.value for s in FraudScenario if s.value != 'random']}"
                        ),
                    )
                    return

            mock = await asyncio.to_thread(generate_mock_scores, scenario=scenario)
            graph_score = mock.graph_score
            behavioral_score = mock.behavioral_score
            temporal_score = mock.temporal_score
            mock_scenario_used = mock.scenario
            graph_available = behavioral_available = temporal_available = True

        elif tx is not None:
            b_resp, g_resp, t_resp = await fetch_upstream(tx, transaction_id)

            behavioral_score = b_resp.score if b_resp.available else None
            graph_score = g_resp.score if g_resp.available else None
            temporal_score = t_resp.score if t_resp.available else None

            behavioral_available = b_resp.available
            graph_available = g_resp.available
            temporal_available = t_resp.available

            behavioral_signal = b_resp.fraud_signal_summary
            graph_signal = g_resp.fraud_signal_summary
            temporal_signal = t_resp.fraud_signal_summary

            if not any([behavioral_available, graph_available, temporal_available]):
                logger.warning("All upstream APIs unavailable — falling back to mock.")
                from backend.mock_scores import FraudScenario, generate_mock_scores

                mock = await asyncio.to_thread(
                    generate_mock_scores, scenario=FraudScenario.RANDOM
                )
                graph_score = mock.graph_score
                behavioral_score = mock.behavioral_score
                temporal_score = mock.temporal_score
                mock_scenario_used = mock.scenario
                graph_available = behavioral_available = temporal_available = True

        else:
            graph_score = request.graph_score
            behavioral_score = request.behavioral_score
            temporal_score = request.temporal_score
            graph_available = graph_score is not None
            behavioral_available = behavioral_score is not None
            temporal_available = temporal_score is not None

    reachable = sum([graph_available, behavioral_available, temporal_available])
    yield StageEvent(
        Stage.MODELS,
        Status.DONE,
        duration_ms=t.ms,
        data={
            "graph": {
                "score": graph_score,
                "available": graph_available,
                "signal": graph_signal,
                "model": "Edge-Enhanced GraphSAGE",
                "owner": "Ewaduge",
                "modality": "Network",
            },
            "behavioral": {
                "score": behavioral_score,
                "available": behavioral_available,
                "signal": behavioral_signal,
                "model": "Stratified VAE + DSAA",
                "owner": "Wijesinghe",
                "modality": "Behaviour",
            },
            "temporal": {
                "score": temporal_score,
                "available": temporal_available,
                "signal": temporal_signal,
                "model": "System-Context TCN",
                "owner": "Pathirana",
                "modality": "Timing",
            },
            "mock_scenario": mock_scenario_used,
        },
        message=(
            f"{reachable} of 3 models contributed"
            if not mock_scenario_used
            else f"Simulated scenario: {mock_scenario_used}"
        ),
    )

    # ── Stage 3: fusion ──────────────────────────────────────────────────────
    yield StageEvent(Stage.FUSION, Status.RUNNING)

    with _Timer() as t:
        fusion = await asyncio.to_thread(
            meta_classifier.fuse,
            graph_score=graph_score if graph_available else None,
            behavioral_score=behavioral_score if behavioral_available else None,
            temporal_score=temporal_score if temporal_available else None,
        )

    classification = classify(fusion.confidence_score)
    yield StageEvent(
        Stage.FUSION,
        Status.DONE,
        duration_ms=t.ms,
        data={
            "fraud_confidence_score": fusion.confidence_score,
            "classification": classification,
            "modalities_used": fusion.modalities_used,
            "graph_score": fusion.graph_score,
            "behavioral_score": fusion.behavioral_score,
            "temporal_score": fusion.temporal_score,
        },
        message=(
            f"{fusion.modalities_used} of 3 modalities fused"
            + ("" if fusion.modalities_used == 3 else " — confidence penalised for the missing model(s)")
        ),
    )

    # ── Stage 4: retrieval ───────────────────────────────────────────────────
    yield StageEvent(Stage.RETRIEVAL, Status.RUNNING, message="Querying the FATF knowledge base")

    with _Timer() as t:
        retrievals = await asyncio.to_thread(
            retriever.retrieve,
            graph_score=fusion.graph_score,
            behavioral_score=fusion.behavioral_score,
            temporal_score=fusion.temporal_score,
            confidence_score=fusion.confidence_score,
        )

    if not retrievals:
        yield StageEvent(
            Stage.RETRIEVAL, Status.ERROR, message="Knowledge base returned no match."
        )
        return

    top = retrievals[0]
    yield StageEvent(
        Stage.RETRIEVAL,
        Status.DONE,
        duration_ms=t.ms,
        data={
            "typology_id": top.typology_id,
            "typology_name": top.typology_name,
            "stage": top.stage,
            "risk_level": top.risk_level,
            "similarity_score": top.similarity_score,
        },
        message=f"Matched {top.typology_name} at {top.similarity_score:.0%} similarity",
    )

    # ── Stage 5: forensic report ─────────────────────────────────────────────
    forensic_report = None
    baseline_report = None

    if forensic_reporter is None:
        yield StageEvent(
            Stage.REPORT,
            Status.SKIPPED,
            message="No language model configured — set a Gemini API key to enable reporting.",
        )
    else:
        yield StageEvent(
            Stage.REPORT, Status.RUNNING, message="Generating the grounded narrative"
        )

        from backend.rag.prompt_builder import UpstreamContext, build_chain_of_evidence_prompt

        prompt_package = build_chain_of_evidence_prompt(
            transaction_id=transaction_id,
            graph_score=fusion.graph_score,
            behavioral_score=fusion.behavioral_score,
            temporal_score=fusion.temporal_score,
            confidence_score=fusion.confidence_score,
            graph_available=fusion.graph_available,
            behavioral_available=fusion.behavioral_available,
            temporal_available=fusion.temporal_available,
            retrieval=top,
            upstream_context=UpstreamContext(
                behavioral_signal_summary=behavioral_signal,
                graph_signal_summary=graph_signal,
                temporal_signal_summary=temporal_signal,
            ),
        )

        with _Timer() as t:
            try:
                forensic_report = await asyncio.to_thread(
                    forensic_reporter.generate_report, prompt_package
                )
                report_status, report_message = Status.DONE, "Report grounded in the retrieved typology"
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                forensic_report = None
                report_status = Status.ERROR
                report_message = f"Report generation failed: {type(e).__name__}"

        yield StageEvent(
            Stage.REPORT,
            report_status,
            duration_ms=t.ms,
            data={"forensic_report": forensic_report},
            message=report_message,
        )

        # Ablation baseline: the same scores with no retrieved context, so the
        # two narratives can be compared side by side.
        if request.include_baseline:
            from backend.rag.prompt_builder import build_baseline_prompt

            try:
                baseline_report = await asyncio.to_thread(
                    forensic_reporter.generate_report,
                    build_baseline_prompt(
                        transaction_id=transaction_id,
                        graph_score=fusion.graph_score,
                        behavioral_score=fusion.behavioral_score,
                        temporal_score=fusion.temporal_score,
                        confidence_score=fusion.confidence_score,
                    ),
                )
            except Exception as e:
                logger.error(f"Baseline LLM generation failed: {e}")
                baseline_report = None

    yield PipelineResult(
        payload={
            "transaction_id": transaction_id,
            "fraud_confidence_score": fusion.confidence_score,
            "classification": classification,
            "graph_score": fusion.graph_score,
            "behavioral_score": fusion.behavioral_score,
            "temporal_score": fusion.temporal_score,
            "graph_available": fusion.graph_available,
            "behavioral_available": fusion.behavioral_available,
            "temporal_available": fusion.temporal_available,
            "modalities_used": fusion.modalities_used,
            "retrieval": {
                "typology_id": top.typology_id,
                "typology_name": top.typology_name,
                "stage": top.stage,
                "risk_level": top.risk_level,
                "similarity_score": top.similarity_score,
            },
            "forensic_report": forensic_report,
            "baseline_report": baseline_report,
            "mock_scenario": mock_scenario_used,
            "behavioral_signal": behavioral_signal,
            "graph_signal": graph_signal,
            "temporal_signal": temporal_signal,
        }
    )
