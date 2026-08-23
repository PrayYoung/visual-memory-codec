# Frozen portable results bundle

This directory is the tracked, platform-independent record for the formal
benchmark. It intentionally excludes raw images, reconstruction payloads,
model caches, credentials, and machine-specific execution logs.

The bundle retains the frozen canonical manifest, staged-control provenance,
protocols, aggregate tables, per-scene tables, and paired bootstrap tables
needed to verify every reported result. SHA-256 values for the copied
files are recorded in `provenance/frozen_run_provenance.json`.

Directories:

- `canonical_n288/`: formal frozen N=288 result tables and protocol.
- `control_a/`: frozen-artifact alternate-evaluator result tables.
- `control_b/`: stronger-text-encoder result tables and completion metadata.
- `text_forensic_gqa64/`: separate completed forensic text diagnostic.
- `provenance/`: canonical manifest and robustness-control staging manifests.

Execution identifiers and infrastructure records are deliberately excluded.
