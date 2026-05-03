"""
Mock Score Generator
Simulates the probability outputs from the three upstream analytical models
(Graph Neural Network, Behavioral VAE, Temporal CNN).
Used when upstream APIs are unreachable during development or demo.
Scenarios are based on PaySim fraud patterns.
"""

import random
from dataclasses import dataclass
from enum import Enum


class FraudScenario(str, Enum):
    SMURFING = "smurfing"
    LAYERING = "layering"
    MULE_NETWORK = "mule_network"
    ACCOUNT_TAKEOVER = "account_takeover"
    VELOCITY_FRAUD = "velocity_fraud"
    LEGITIMATE = "legitimate"
    RANDOM = "random"


@dataclass
class UpstreamScores:
    graph_score: float
    behavioral_score: float
    temporal_score: float
    scenario: str


_SCENARIO_PROFILES: dict[FraudScenario, dict] = {
    FraudScenario.SMURFING: {
        "graph": (0.45, 0.65),
        "behavioral": (0.35, 0.55),
        "temporal": (0.75, 0.92),
    },
    FraudScenario.LAYERING: {
        "graph": (0.80, 0.95),
        "behavioral": (0.75, 0.90),
        "temporal": (0.70, 0.88),
    },
    FraudScenario.MULE_NETWORK: {
        "graph": (0.85, 0.97),
        "behavioral": (0.78, 0.93),
        "temporal": (0.50, 0.70),
    },
    FraudScenario.ACCOUNT_TAKEOVER: {
        "graph": (0.10, 0.30),
        "behavioral": (0.85, 0.97),
        "temporal": (0.75, 0.90),
    },
    FraudScenario.VELOCITY_FRAUD: {
        "graph": (0.40, 0.60),
        "behavioral": (0.45, 0.65),
        "temporal": (0.88, 0.99),
    },
    FraudScenario.LEGITIMATE: {
        "graph": (0.02, 0.20),
        "behavioral": (0.02, 0.18),
        "temporal": (0.02, 0.15),
    },
}


def generate_mock_scores(
    scenario: FraudScenario = FraudScenario.RANDOM,
    seed: int | None = None,
) -> UpstreamScores:
    if seed is not None:
        random.seed(seed)

    if scenario == FraudScenario.RANDOM:
        scenario = random.choice(list(FraudScenario)[:-1])  # exclude RANDOM itself

    profile = _SCENARIO_PROFILES[scenario]

    def sample(low: float, high: float) -> float:
        return round(random.uniform(low, high), 4)

    return UpstreamScores(
        graph_score=sample(*profile["graph"]),
        behavioral_score=sample(*profile["behavioral"]),
        temporal_score=sample(*profile["temporal"]),
        scenario=scenario.value,
    )


def get_all_scenarios() -> list[str]:
    return [s.value for s in FraudScenario if s != FraudScenario.RANDOM]
