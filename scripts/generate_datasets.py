"""
Generate the two demonstration datasets.

    python scripts/generate_datasets.py

Produces, in data/datasets/:

    fraudulent_transactions.csv   legitimate traffic with known frauds embedded
    clean_transactions.csv        legitimate traffic only, no fraud at all

Both follow the PaySim schema so they are interchangeable with the real corpus.

WHAT THIS IS AND IS NOT
-----------------------
These are synthetic fixtures. They encode the structural signatures of the FATF
typologies the models are built to detect — convergence on a sink account,
below-threshold structuring, multi-hop layering, full-balance drains, machine
regular timing.

Running our own models over data we generated demonstrates that the pipeline
works end to end. It is not a measurement of real-world accuracy, and it should
never be presented as one. Accuracy claims belong to the held-out PaySim
evaluation each model reports separately.

The fraud set carries an isFraud column as ground truth so detection can be
scored. The clean set is labelled throughout as isFraud=0, which makes any alert
raised against it a false positive by construction.
"""

import argparse
import csv
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

# PaySim column order, preserved exactly.
COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Annotation columns, written only to the fraud set so a reviewer can see which
# typology each planted fraud represents. Stripped before the file is scored.
ANNOTATION_COLUMNS = ["typology", "note"]


@dataclass
class Txn:
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int = 0
    isFlaggedFraud: int = 0
    typology: str = ""
    note: str = ""

    def row(self, annotated: bool) -> dict:
        d = asdict(self)
        for key in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
            d[key] = round(d[key], 2)
        if not annotated:
            for key in ANNOTATION_COLUMNS:
                d.pop(key)
        return d


class Accounts:
    """Issues stable account identifiers in PaySim's C/M format."""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._used: set[str] = set()

    def customer(self) -> str:
        return self._new("C")

    def merchant(self) -> str:
        return self._new("M")

    def _new(self, prefix: str) -> str:
        while True:
            name = f"{prefix}{self._rng.randint(10**8, 10**9 - 1)}"
            if name not in self._used:
                self._used.add(name)
                return name


# ── Legitimate behaviour ─────────────────────────────────────────────────────


def legitimate_batch(rng: random.Random, acc: Accounts, count: int, step_range) -> list[Txn]:
    """
    Ordinary customer activity: salary credits, card purchases, bill payments,
    modest peer transfers and cash withdrawals.

    Amounts and balances are internally consistent — a purchase leaves the
    balance it should — because an inconsistent ledger is itself an anomaly and
    would make the clean set trivially detectable.
    """
    out: list[Txn] = []

    for _ in range(count):
        step = rng.randint(*step_range)
        kind = rng.choices(
            ["PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT"],
            weights=[45, 22, 15, 13, 5],
        )[0]

        origin = acc.customer()
        opening = round(rng.uniform(2_000, 180_000), 2)

        if kind == "PAYMENT":
            amount = round(rng.uniform(5, min(2_500, opening * 0.4)), 2)
            dest = acc.merchant()
            # PaySim does not track merchant balances for PAYMENT.
            out.append(Txn(step, kind, amount, origin, opening, opening - amount, dest, 0.0, 0.0))

        elif kind == "CASH_IN":
            amount = round(rng.uniform(1_000, 40_000), 2)
            dest = acc.customer()
            out.append(
                Txn(step, kind, amount, origin, opening, opening + amount, dest, 0.0, 0.0)
            )

        elif kind == "CASH_OUT":
            amount = round(rng.uniform(100, min(12_000, opening * 0.5)), 2)
            dest = acc.merchant()
            d_open = round(rng.uniform(50_000, 900_000), 2)
            out.append(
                Txn(step, kind, amount, origin, opening, opening - amount, dest,
                    d_open, d_open + amount)
            )

        elif kind == "DEBIT":
            amount = round(rng.uniform(20, min(1_800, opening * 0.3)), 2)
            dest = acc.merchant()
            out.append(Txn(step, kind, amount, origin, opening, opening - amount, dest, 0.0, 0.0))

        else:  # TRANSFER between two customers, partial amount
            amount = round(rng.uniform(200, min(9_000, opening * 0.45)), 2)
            dest = acc.customer()
            d_open = round(rng.uniform(1_000, 120_000), 2)
            out.append(
                Txn(step, kind, amount, origin, opening, opening - amount, dest,
                    d_open, d_open + amount)
            )

    return out


# ── Fraud typologies ─────────────────────────────────────────────────────────
#
# Each generator reproduces the structural signature one model is built to see.
# The comment on each says which signal it is meant to trip.


def mule_network(rng: random.Random, acc: Accounts, step: int) -> list[Txn]:
    """
    Hub and spoke. Several freshly funded accounts push their full balance into
    a single sink within a narrow window, which is then emptied.

    Signature for the graph model: high convergence count on one destination.
    """
    out: list[Txn] = []
    sink = acc.customer()
    senders = rng.randint(4, 7)
    accumulated = 0.0

    for i in range(senders):
        sender = acc.customer()
        amount = round(rng.uniform(8_000, 45_000), 2)
        out.append(
            Txn(
                step + rng.randint(0, 2),
                "TRANSFER",
                amount,
                sender,
                amount,          # funded to exactly the transfer amount
                0.0,             # and drained completely
                sink,
                accumulated,
                accumulated + amount,
                isFraud=1,
                typology="MULE_NETWORK",
                note=f"Spoke {i + 1} of {senders} converging on sink {sink}",
            )
        )
        accumulated += amount

    # The sink is emptied once the funds have landed.
    out.append(
        Txn(
            step + 3,
            "CASH_OUT",
            accumulated,
            sink,
            accumulated,
            0.0,
            acc.merchant(),
            0.0,
            accumulated,
            isFraud=1,
            typology="MULE_NETWORK",
            note=f"Sink account emptied after receiving from {senders} spokes",
        )
    )
    return out


def smurfing(rng: random.Random, acc: Accounts, step: int) -> list[Txn]:
    """
    Structuring. One large sum split into many transfers deliberately kept under
    a reporting threshold.

    Signature: repeated near-identical amounts just below 10,000.
    """
    out: list[Txn] = []
    origin = acc.customer()
    dest = acc.customer()
    parts = rng.randint(6, 10)
    balance = round(rng.uniform(70_000, 130_000), 2)
    received = round(rng.uniform(0, 5_000), 2)

    for i in range(parts):
        # Just under the threshold, varied enough to look unscripted.
        amount = round(rng.uniform(8_700, 9_890), 2)
        if amount > balance:
            break
        out.append(
            Txn(
                step + i,
                "TRANSFER",
                amount,
                origin,
                balance,
                balance - amount,
                dest,
                received,
                received + amount,
                isFraud=1,
                typology="SMURFING",
                note=f"Tranche {i + 1} of {parts}, held below the 10,000 threshold",
            )
        )
        balance -= amount
        received += amount

    return out


def layering(rng: random.Random, acc: Accounts, step: int) -> list[Txn]:
    """
    Layering. Funds moved through a chain of intermediaries, each hop passing on
    nearly the whole amount, to break the audit trail.

    Signature: a path where each node's out-value closely tracks its in-value.
    """
    out: list[Txn] = []
    hops = rng.randint(4, 6)
    amount = round(rng.uniform(40_000, 160_000), 2)
    current = acc.customer()
    balance = amount

    for i in range(hops):
        nxt = acc.customer()
        # A small skim at each hop, as a real layering chain shows.
        passed = round(balance * rng.uniform(0.94, 0.99), 2)
        out.append(
            Txn(
                step + i,
                "TRANSFER",
                passed,
                current,
                balance,
                round(balance - passed, 2),
                nxt,
                0.0,
                passed,
                isFraud=1,
                typology="LAYERING",
                note=f"Hop {i + 1} of {hops} in the chain",
            )
        )
        current, balance = nxt, passed

    return out


def account_takeover(rng: random.Random, acc: Accounts, step: int) -> list[Txn]:
    """
    Takeover. A dormant account with a substantial balance is emptied in one or
    two moves to a destination it has never transacted with.

    Signature for the behavioural model: a large departure from the account's
    established baseline.
    """
    out: list[Txn] = []
    victim = acc.customer()
    balance = round(rng.uniform(60_000, 400_000), 2)
    dest = acc.customer()

    if rng.random() < 0.5:
        out.append(
            Txn(step, "TRANSFER", balance, victim, balance, 0.0, dest, 0.0, balance,
                isFraud=1, typology="ACCOUNT_TAKEOVER",
                note="Entire balance moved in a single transfer")
        )
    else:
        first = round(balance * 0.6, 2)
        out.append(
            Txn(step, "TRANSFER", first, victim, balance, balance - first, dest, 0.0, first,
                isFraud=1, typology="ACCOUNT_TAKEOVER",
                note="First of two withdrawals draining the account")
        )
        out.append(
            Txn(step, "CASH_OUT", balance - first, victim, balance - first, 0.0,
                acc.merchant(), 0.0, balance - first,
                isFraud=1, typology="ACCOUNT_TAKEOVER",
                note="Remaining balance withdrawn immediately after")
        )
    return out


def velocity_fraud(rng: random.Random, acc: Accounts, step: int) -> list[Txn]:
    """
    Automation. Many transfers from one account inside a single hour, spaced far
    too regularly for a person.

    Signature for the temporal model: an elevated burstiness coefficient.
    """
    out: list[Txn] = []
    origin = acc.customer()
    count = rng.randint(8, 14)
    balance = round(rng.uniform(50_000, 200_000), 2)
    # Near-identical amounts are the tell — a human does not repeat to the cent.
    base = round(rng.uniform(2_000, 6_000), 2)

    for i in range(count):
        amount = round(base + rng.uniform(-25, 25), 2)
        if amount > balance:
            break
        dest = acc.customer()
        out.append(
            Txn(
                step,                       # all inside the same hour
                "TRANSFER",
                amount,
                origin,
                balance,
                balance - amount,
                dest,
                0.0,
                amount,
                isFraud=1,
                typology="VELOCITY_FRAUD",
                note=f"Transfer {i + 1} of {count} within a single step",
            )
        )
        balance -= amount

    return out


TYPOLOGIES = [mule_network, smurfing, layering, account_takeover, velocity_fraud]


# ── Assembly ─────────────────────────────────────────────────────────────────


def build_fraud_set(rng: random.Random, legit_count: int, episodes: int) -> list[Txn]:
    acc = Accounts(rng)
    rows = legitimate_batch(rng, acc, legit_count, (1, 720))

    for _ in range(episodes):
        generator = rng.choice(TYPOLOGIES)
        rows.extend(generator(rng, acc, rng.randint(1, 700)))

    # Interleave by step so the frauds are not clustered at the end, which would
    # make them findable by position rather than by behaviour.
    rows.sort(key=lambda t: (t.step, rng.random()))
    return rows


def build_clean_set(rng: random.Random, count: int) -> list[Txn]:
    acc = Accounts(rng)
    rows = legitimate_batch(rng, acc, count, (1, 720))
    rows.sort(key=lambda t: (t.step, rng.random()))
    assert all(t.isFraud == 0 for t in rows), "clean set must contain no fraud"
    return rows


def write_csv(path: Path, rows: list[Txn], annotated: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = COLUMNS + (ANNOTATION_COLUMNS if annotated else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for t in rows:
            writer.writerow(t.row(annotated))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/datasets", type=Path)
    p.add_argument("--legit", type=int, default=260, help="legitimate rows in the fraud set")
    p.add_argument("--episodes", type=int, default=12, help="fraud episodes to plant")
    p.add_argument("--clean", type=int, default=300, help="rows in the clean set")
    p.add_argument("--seed", type=int, default=20260828, help="fixed so runs are reproducible")
    args = p.parse_args()

    rng = random.Random(args.seed)

    fraud_rows = build_fraud_set(rng, args.legit, args.episodes)
    clean_rows = build_clean_set(rng, args.clean)

    fraud_path = args.out / "fraudulent_transactions.csv"
    clean_path = args.out / "clean_transactions.csv"

    write_csv(fraud_path, fraud_rows, annotated=True)
    write_csv(clean_path, clean_rows, annotated=False)

    n_fraud = sum(t.isFraud for t in fraud_rows)
    by_typology: dict[str, int] = {}
    for t in fraud_rows:
        if t.isFraud:
            by_typology[t.typology] = by_typology.get(t.typology, 0) + 1

    print(f"{fraud_path}")
    print(f"  {len(fraud_rows)} transactions, {n_fraud} fraudulent "
          f"({n_fraud / len(fraud_rows):.1%})")
    for name, n in sorted(by_typology.items()):
        print(f"    {name:<20} {n}")

    print(f"\n{clean_path}")
    print(f"  {len(clean_rows)} transactions, 0 fraudulent")
    print("\nSeed fixed — re-running reproduces these files exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
