# Hermes Penelopa

A small lifecycle and transport adapter around **the genuine NousResearch Hermes
agent**, not a replacement LLM/tool loop. The image derives from official
`nousresearch/hermes-agent:v2026.8.31` at immutable digest
`sha256:64923faeae267792bf9bf87fe3b4c4869e35004e360c7df01730ad801b74d524`
(upstream revision `29112bef099274229cadff79cdff7bf7b99c4b77`).

## Architecture

- Native `AIAgent.run_conversation`, `GoalManager`, SessionDB, memory, skills and
  session search handle reasoning and persistence.
- A loopback broker holds external credentials, routes every LLM request to the
  configured OpenAI-compatible provider/model chain, accounts for usage, and
  exposes only the task's allowed MCP tools through a native stdio MCP connection.
- The backend's accepted terminal MCP receipt is the only business-success signal.
  A plain assistant answer or goal-judge verdict is not sufficient. An empty
  `no_recommendation` outcome with a reason is a valid success.
- One genuine upstream background review runs after terminal acceptance. The
  runtime waits for its worker to finish and closes SessionDB before exiting.
  Recommendation success and memory-review success are separate diagnostic facts.

## Per-user storage and permissions

Mount **one named volume for one user** at `/opt/data`. Native Hermes state is in
`/opt/data/hermes`; user working files belong in `/opt/data/workspace`. The adapter
itself runs from the immutable `/opt/penelopa` directory. Never share a
volume or use multiple user profiles inside one instance. Reuse the same volume
for later tasks and image updates. Removing a container must not remove its volume.

Native history, memory and learned skills persist until explicit memory reset or
account erasure. Automatic native session pruning is disabled. Source transcript
expiry is enforced by the backend when reading or citing evidence; it does not
retroactively erase summaries or copies already retained in native history.

Only native memory/skills/session search and explicitly allowed Penelopa MCP tools
are exposed. Shell, browser, code execution, plugin loading and inline shell inside
skills are disabled. No Docker socket, host directories or other users' volumes
should be mounted. The process drops root before importing/running Hermes;
the adapter and upstream installation remain outside the writable user volume.

## Launch contract

Required environment variables, provided by the trusted task launcher:

| Variable | Meaning |
| --- | --- |
| `HERMES_USER_ID`, `HERMES_TASK_ID` | Owner and task UUIDs |
| `HERMES_CLAIM_VERSION` | Current server claim version |
| `HERMES_TASK_TOKEN` | Temporary, revocable user-data MCP capability |
| `HERMES_LIFECYCLE_TOKEN` | Separate claim-scoped receipt/lifecycle capability; no data access |
| `HERMES_INTERNAL_API_BASE_URL` | Backend base URL |
| `HERMES_BASE_URL` | Configured OpenAI-compatible provider base URL, including `/v1` |
| `HERMES_MODEL`, `HERMES_API_TOKEN` | Provider model and credential |

`HERMES_FALLBACK_MODELS` is an optional JSON array of up to two additional models
at the same provider. `HERMES_HOME` defaults to `/opt/data/hermes`. Transport settings
are `HERMES_INTERNAL_HTTP_TIMEOUT_SECONDS`, `HERMES_PROVIDER_HTTP_TIMEOUT_SECONDS`
and `HERMES_HEARTBEAT_INTERVAL_SECONDS`; they are per-request/liveness guards, not
cumulative generation budgets. Never bake credentials into an image or put them in
the persistent Hermes config, skills or memory. Launch environment values are
removed before native Hermes and MCP subprocesses start.

The backend must implement the allowlisted MCP endpoint `/_internal/hermes/mcp`
and scoped lifecycle endpoints beneath `/api/internal/hermes/tasks/{task_id}`:
`GET /receipt` and `POST /runtime-heartbeat`. It must revoke data access at terminal
acceptance while retaining only the narrow lifecycle capability during finalization.
The queue must reserve the execution slot and user volume until physical exit,
even when the business task is already marked successful.

Non-secret runtime diagnostics are atomically written to `/opt/data/runtime.json`.
An interrupted terminal exchange is reconciled with the backend before any replay.
Stopping uses SIGTERM and graceful shutdown; a forced kill is an exceptional
recovery operation, never the normal post-success path. Do not run a gateway,
cron daemon or unrelated background jobs in this container.

## Build, verify, publish

```sh
docker build -t hermes-penelopa:test .
docker run --rm --network none --entrypoint /opt/hermes/.venv/bin/python \
  hermes-penelopa:test -m unittest discover -s /opt/penelopa/tests -p test_native_contract.py -v
HERMES_TEST_IMAGE=hermes-penelopa:test python -m unittest discover \
  -s tests -p test_container_integration.py -v
pre-commit run --all-files
```

The integration tests use synthetic HTTP/MCP/LLM services and uniquely named test
volumes only. They exercise the real image, native tool dispatch, delayed review,
restart persistence, user isolation, provider fallback and terminal-ACK recovery.
They require a local Docker daemon but no paid LLM credentials.

CI builds once, runs both test layers, then pushes that exact image to
`ghcr.io/chigwell/hermes-penelopa:<git-sha>`. Consumers must pin its published
`@sha256` digest. Package visibility must explicitly be public and verified with
an unauthenticated pull; a public source repository alone does not guarantee it.
No `latest` tag or unverified image is used for production.

The upstream pinned private helper signatures are covered by image contract tests.
Updating Hermes requires updating the base digest, running the complete suite and
publishing a new immutable artifact. Existing per-user volumes are not recreated.
