# Paper 1 frozen results bundle

This directory is the tracked, platform-independent record for the formal
benchmark. It intentionally excludes raw images, reconstruction payloads,
model caches, credentials, and machine-specific execution logs.

The bundle retains the minimum platform-independent inputs needed to inspect
the reported claims: frozen manifest provenance, protocols, aggregate tables,
the canonical per-scene table, and paired bootstrap summaries. The release
table and figure live in `paper_assets/`.

Directories:

- `canonical_n288/`: formal frozen N=288 aggregate, paired-subgroup, and
  per-scene result tables plus protocol.
- `control_a/`: frozen-artifact alternate-evaluator result tables.
- `control_b/`: stronger-text-encoder result tables and completion metadata.
- `text_forensic_gqa64/`: compact summary of a separate text-only diagnostic.
- `provenance/`: manifest provenance and a release provenance record.

Raw artifact payloads, full QA dumps, detailed staging manifests, execution
identifiers, and infrastructure records remain local and are deliberately
excluded.
