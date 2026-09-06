# Changelog

## 1.5.0

### Memory safety

- Added `local_rag_update`: correct a retrieved semantic item in place using its ID and exact `expected_text`. Identity, source and scope remain unchanged; omitted optional semantic fields are preserved.
- Validate and embed replacements before the write transaction, then recheck the original row under the transaction. Conflicts, embedding failures and SQLite failures preserve the existing fact and its FTS entry.
- Reject hidden duplicate collisions rather than rewriting or exposing another scope's item.
- Apply visibility checks to deletion. The standalone CLI derives project authority from its actual working directory and cannot acquire session authority from caller-supplied environment variables.
- Share semantic-field validation between saves and corrections, including metadata screening and finite-number checks.

### Retrieval and image evidence

- Default text semantic admission threshold is now 0.3, applied before rank fusion. This calibration is not a general answer-accuracy guarantee.
- Curated images can carry optional pixel-derived descriptions as retrieval hints. Captions and OCR are untrusted; exact visual claims require inspecting the selected image.
- Clarify evidence guidance: retrieved candidates and paths are not proof that the model inspected their pixels.

### Verification and boundaries

- Added deterministic regression coverage for atomic updates, conflicts, rollback, validation, scoped deletion, relevance and publication privacy.
- Local native-agent acceptance covered a nine-turn memory lifecycle and seven source-identity/table/multi-source visual scenarios. These are targeted checks, not an independent general production-readiness benchmark. Private native transcripts and memory databases are not distributed.
- `expected_text` is content compare-and-swap, not a monotonic revision token: an A→B→A change completed before preflight is not detectable by this API.
- The package remains beta. No automatic raw-dialogue or image-attachment ingestion was added.

### Upgrade

Back up local memory using SQLite's backup API, install `hermes-local-rag==1.5.0` in the environment running Hermes, and restart the relevant Hermes processes when convenient to activate the code. If using the separately installed Dashboard plugin, update that plugin as well. No model swap or Hermes core patch is required. For upgrades from 1.3.2 or older, follow the migration instructions in `UPGRADE.md` before starting new writers.
