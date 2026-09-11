# Claude Code task — cloud-scheduled Naigos learning pipeline

Implement a production-minded, **cloud-scheduled candidate-learning pipeline** for this repository. The user must be able to close their laptop after one deployment; no long-running process on a local machine may be required.

## Read before modifying anything

Read `README.md`, `docs/STACK.md`, `next-steps.md`, `naigos/rl/modal_train.py`, `scripts/modal_runs.py`, `naigos/research/{cache,spec,run,allowlist}.py`, and the existing tests. Preserve existing behavior and current detached-run commands.

This is a simulation/RL project. It must remain so. Do not add real-world targeting, current weapon-system capability collection, open-ended web research, or any offensive blue-agent action. The existing fixed allowlist and `GUARDRAIL` are mandatory. Blue remains purely evasive.

## Outcome

Create a Modal-deployed control plane that runs this sequence unattended:

```text
scheduled snapshot -> immutable provenance-checked input snapshot
                   -> candidate training
                   -> held-out evaluation + verifier gates
                   -> promotion decision
                   -> atomically update champion pointer only on success
```

"Learning live" here means periodically learning from refreshed **allowlisted, non-sensitive simulation inputs and generated scenario seeds**. It does **not** mean modifying a deployed policy or its weights after every live rollout, nor automatically using unreviewed external data.

## Current constraints that your design must address

- `modal_train.py` already runs detached jobs and persists results on the `naigos-runs` Modal Volume, but it has no scheduler.
- Its image currently copies `components/` and `data_cache/` into an immutable image and intentionally keeps training offline. Do not weaken that training invariant.
- `naigos.research.cache` and `naigos.research.spec` currently derive their directories from the repo root. Refactor them to support explicit/environment-configurable cache and component roots without changing existing local defaults.
- A scheduled Modal Function cannot rely on local files or receive interactive arguments. Configuration must be an immutable checked-in default plus a versioned JSON configuration on the persistent Volume.
- The current `naigos-runs` Volume is an artifact store, not a mutable global working directory. Use namespaced, immutable directories; never mutate a completed snapshot or candidate run.

## Required implementation

### 1. Persistent layout and immutable records

Add a clear volume layout, documented in code and `docs/STACK.md`:

```text
/pipeline/config/current.json                 # versioned scheduler config
/pipeline/snapshots/<snapshot-id>/             # raw cache, components, manifest, snapshot.json
/pipeline/candidates/<candidate-id>/           # training/evaluation artifacts and decision.json
/pipeline/champions/current.json               # small atomic policy pointer
/pipeline/locks/                               # narrow lease/lock records
```

Each `snapshot.json`, candidate `run.json`, evaluation result, and promotion decision must include: UTC timestamps, code commit, config digest, parent IDs, AOI, scenario/evaluation seeds, and content hashes as appropriate. IDs must be validated and path traversal must be impossible.

Do not overwrite immutable snapshot or candidate data. Publishing the champion must write a temporary file then atomically replace the small pointer file, with a generation number. A failed or uncertain job must leave the existing champion unchanged.

### 2. Separate snapshot worker from offline training worker

Create a new Modal module (choose a clear name such as `naigos/rl/modal_pipeline.py`) with separate Functions/images:

- **Snapshot worker:** has only the research dependencies it needs, network access solely through the existing allowlist, and mounts a Volume location for its output. It runs the existing research pipeline against snapshot-specific cache/components roots. It must validate cache hashes, component citations, and AOI consistency before marking a snapshot complete.
- **Training worker:** uses the existing offline training design. It receives/uses one completed snapshot copied or mounted read-only for that candidate. It must never fetch from the network and must record the exact snapshot ID.
- **Evaluation/promoter:** evaluates a candidate against fixed held-out seeds and existing baselines, runs the independent verifier, and writes a structured decision. It is the only component allowed to update the champion pointer.

Use Modal Volumes or another Modal-native persistent store; do not depend on GitHub Actions, a local cron daemon, or a computer being awake. Keep credentials in Modal Secrets or the provider’s managed credentials, never in code, artifacts, logs, or CLI arguments.

### 3. Scheduling and idempotency

Add a deployed `modal.Cron` coordinator, with conservative checked-in defaults:

- daily snapshot refresh at a documented UTC time;
- nightly candidate attempt only when a new completed snapshot exists;
- weekly longer evaluation/training only if the prior jobs passed.

Use a scheduling configuration file and document how an operator changes the cadence by updating the config then redeploying. The coordinator must be safe if Modal invokes it twice, overlaps an old invocation, or resumes after a partial failure:

- acquire a narrow lease keyed by snapshot/candidate stage;
- use deterministic idempotency keys based on config + input snapshot + code commit;
- skip an already-completed equivalent stage;
- fail closed on a stale/ambiguous lease rather than running concurrent writers;
- record a useful status/heartbeat and explicit failure reason.

Never use blind Modal retries to duplicate expensive training. Preserve the repository’s checkpoint/resume behavior.

### 4. Promotion gates

Start with a deliberately strict default policy. A candidate may be promoted only when all are true:

- its run and data provenance verify cleanly;
- the independent verifier reports no disqualifying constraint issue;
- held-out evaluation uses fixed, recorded seeds not used for training;
- it meets configurable minimum survival and objective-reached thresholds;
- it does not regress past configurable tolerances on detection/exposure, shootdowns, terrain/out-of-bounds losses, or the avoid-plus-nap baseline;
- the current champion remains available until the replacement pointer is atomically published.

Do not manufacture metrics. Reuse or factor the project’s existing evaluator so the result is reproducible. If a required metric is not currently available, implement it with tests or mark promotion `rejected`/`inconclusive`; never silently waive the gate.

The automated default must be **shadow promotion**: write an eligible decision and candidate artifact but require `auto_promote: false` unless the operator explicitly enables it in the versioned config. Provide a documented CLI/API action to promote one named, already-approved candidate and re-run all validation before the pointer changes.

### 5. Operability

Extend or add a small CLI with commands such as:

```bash
modal deploy naigos/rl/modal_pipeline.py
uv run python scripts/pipeline.py status
uv run python scripts/pipeline.py list-snapshots
uv run python scripts/pipeline.py list-candidates
uv run python scripts/pipeline.py inspect <id>
uv run python scripts/pipeline.py promote <candidate-id>
uv run python scripts/pipeline.py pause
uv run python scripts/pipeline.py resume
```

Adapt command names if needed, but preserve these capabilities. `status` must distinguish scheduled, queued, running, completed, failed, skipped, rejected, inconclusive, and promoted. It should work from a fresh clone that has no local job index, using the remote authoritative record.

Do not claim a scheduler can be paused merely by a local flag. Since Modal schedules are deployment-defined, implement pause as a remote config gate that the coordinator checks before doing costly work, and clearly surface the paused state.

Document setup, required Modal authentication, approximate cost controls, deployment, status, how to safely pause, how to recover a failed stage, retention, and how to roll back by repointing the champion pointer. Explain that deployment is the only action that needs the laptop; deployed scheduled functions continue after it is closed.

### 6. Testing

Add focused offline tests; do not require a Modal account, GPU, network, or populated data cache. At minimum cover:

- local-default and explicit snapshot cache/component roots;
- immutable IDs/paths and parent/hash metadata;
- idempotency decisions, lease behavior, and stale/ambiguous failure behavior;
- no concurrent writers for a candidate;
- snapshot validation rejects missing bytes, hash mismatches, uncited components, or an AOI mismatch;
- evaluation uses held-out seeds and refuses an overlap;
- every promotion rejection path leaves the current champion byte-for-byte unchanged;
- atomic champion-promotion semantics and rollback;
- `auto_promote: false` never changes the champion;
- pipeline configuration/pause status and remote-state classification;
- existing test suite remains green.

Run relevant tests plus `pytest` before handoff. Do not run paid Modal jobs unless explicitly asked; implementation and offline tests are the deliverable.

## Engineering standards

- Prefer small, composable pure-Python modules for the decision logic; Modal wrappers should be thin and testable.
- Validate external/configuration JSON with explicit schemas or strict parsers.
- Redact logs using the existing `runmeta.redact` approach.
- Do not commit secrets, generated caches, or remote artifacts.
- Keep compatible local research, demo, and `scripts/modal_runs.py` workflows.
- Make small, coherent commits if the repository’s working tree is clean. Never discard unrelated user changes.

## Handoff

At completion, report: changed files, exact deployment command, default cadence, the promotion gates, all tests run, and anything intentionally deferred. Be explicit that no real remote run has been performed unless one actually was.
