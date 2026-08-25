# Ablation: retrieval grounding vs ungrounded generation

Cases: 5  ·  generated 2026-08-23 13:03 UTC
Model: gemini-2.5-flash

Every measure is a deterministic check against the evidence supplied in
the prompt. No language model grades another model's output.

| Measure | Grounded | Baseline | |
|---|---:|---:|---|
| Numeric fidelity (higher better) | 100.0% | 92.1% | grounded better |
| Fabricated figures per report | 0.00 | 1.20 | grounded better |
| Reports citing a fabricated typology ID | 1 | 0 | **baseline better** |
| Reports asserting unsupported patterns | 0 | 4 | grounded better |

## Unavailable modalities

When a model does not respond, the report must say so rather than
estimate. Inventing a score here is the most serious failure in this
system: an investigator cannot tell it happened.

| | Grounded | Baseline |
|---|---:|---:|
| Instances of a missing modality | 0 | 0 |
| Correctly flagged as unavailable | 0 | 0 |
| Score invented for it | 0 | 0 |

## Examples of ungrounded figures in the baseline

- `EVAL_002` (layering): `26`, `180`
- `EVAL_001` (mule_network): `26`
- `EVAL_003` (smurfing): `26`

## Reading this

Numeric fidelity is the share of figures in a report that match a value
supplied in its prompt. A fabricated typology ID means the report cited
an FATF identifier it was never given. An unsupported pattern is a named
laundering technique asserted without appearing in the retrieved
typology text.

This measures traceability, not writing quality or real-world
correctness — only whether each claim can be sourced. That is the
property Chain of Evidence prompting enforces, and the one a regulator
would test.
