# Changelog

All notable changes to Turbohaul-Manager are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

GitHub: `https://github.com/MrTrenchTrucker/turbohaul-manager`

---

## [0.5.0] — 2026-06-27

### Highlights

- **Multi-slot resident registry** — the manager now tracks loaded-model sidecars through a per-model resident registry with a lifecycle state machine (RESERVED_LOADING → ACTIVE → GRACE → IDLE_EVICTABLE → DEAD). The registry is Phase-0 scaffolded to a single resident for backward compatibility, but the architecture is wired for future multi-model concurrency. A dispatcher routes incoming slots to available residents.
- **Flap/degradation telemetry subsystem** — a new, low-overhead, append-only JSONL logging subsystem captures the full request lifecycle (arrival → queue → prefill → first-token → generation → completion) with timestamps, plus VRAM/memory resource samples. A ring buffer (10K events, in-memory) and persistent rotating JSONL files (10 MiB, 5 files retained) provide both hot and full-history read paths. Two new endpoints: `GET /v1/telemetry/events` and `GET /v1/telemetry/status`.
- **Warm-slot reuse with per-client IP affinity** — the manager now derives a stable per-agent `thread_id` from the client's IP address (single-residency mode). Same-client follow-ups inherit the warm sidecar, killing the full-context re-prefill crawl and queue-stacking. A gated cap (default ≤1) preserves the fan-out behavior at higher concurrency levels.
- **Keep-alive during stream-OPEN** — keep-alive comment frames are now emitted during the prefill-open phase (not just the byte loop), preventing premature client-timeout during long prefill waits.
- **Faster failure recovery** — a model load that crashes now fails its slot in ~2 s instead of waiting out the full load-health timeout (600 s). A dead child detected during `wait_until_healthy` immediately transitions the slot to DEAD.

### Added

- `src/turbohaul/telemetry.py` (449 LoC) — the flap/degradation telemetry subsystem: `FlapTelemetry` class with lifecycle hooks (`on_request_arrival`, `on_queue_state`, `on_slot_assign`, `on_prefill_start`, `on_first_token`, `on_generation_tick`, `on_keep_alive_emitted`, `on_client_disconnect`, `on_completion`, `on_vram_sample`, `on_slot_state_change`), a rotating `_JsonlWriter` (10 MiB per file, 5 files retained, line-buffered), and an in-memory `_RingBuffer` (10K events).
- `src/turbohaul/api/telemetry.py` (65 LoC) — two new API endpoints: `GET /v1/telemetry/events` (paginated, cursor-based, source=ring or source=file) and `GET /v1/telemetry/status` (subsystem health + stats).
- Resident registry scaffold in `manager.py` — `ResidentState` enum (RESERVED_LOADING, ACTIVE, GRACE, IDLE_EVICTABLE, DEAD), `Resident` dataclass, `_residents` dict, `_registry_lock`, dispatcher logic with `_MAX_DISPATCH_DEFERS` (50) for unroutable slot failure.
- Multi-slot state-migration mirror in `manager.py` — per-resident mirrors of `_idle_handle`, `_idle_model_tag`, `_spawn_seq`, `_latest_keep_alive_s`, and `_latest_keep_alive_s` on the `Resident` class.
- Fan-out rider admission for `cap<=1` — continuous rider-admit during drain with `_FANOUT_ADMIT_POLL_S` (0.1 s) bounded poll interval.
- Warm-slot reuse — stable per-agent `thread_id` derived from client IP (single-residency). Same-client follow-ups inherit the warm sidecar, killing the full-context re-prefill crawl.
- `docs/telemetry_reader_guide.md` — telemetry reader guide.

### Changed

- `manager.py` (1900 → 3652 LoC) — major rewrite:
  - Multi-slot refactor Phase-0: resident registry, dispatcher, per-resident driver tasks.
  - `MAX_PARALLEL_SIDECARS = 1` (Phase-0 pinned to single sidecar for backward compatibility).
  - `_inflight` list renamed to per-resident `inflight` — concurrent fan-out rider slots.
  - `_latest_keep_alive_s` moved from manager-global to per-resident (correctness fix: per-model keep_alive semantics).
  - `REDACTED_KEYS` changed from `set` to `frozenset` (mutability guard).
  - `_KEEP_ALIVE_UNITS` changed from mutable `set` to read-only contract (frozenset).
  - `all_safety_gates` moved off the async event loop via `asyncio.to_thread` (prevents blocking the event loop during safety checks).
  - `keep_alive: -1` now maps to `KEEP_ALIVE_MAX_S` constant (prevents the -1 value from leaking as a literal keep_alive duration).
  - `is False` → `not` for PEP-8 compliance.
  - VRAM total in megabytes (`vram_total_mib`) now surfaced in `/status` and frontend.
  - WebSocket double-poll debounce — prevents redundant poller wakes.
- `queue.py` — queue FSM routing fixes:
  - `enqueue_head` now correctly routes STAGED→STAGED (previously a no-op that dropped the slot into the wrong queue tier).
  - Queue depth tracking improved for telemetry integration.
- `chat_completion.py` — request payload handling:
  - `payload = dict(payload)` added before mutation (prevents in-place mutation of the caller's dict — the original dict was being modified when `response_format` was set to `None`).
  - Payload mutation fix ensures the original request body is preserved across retries.
- `state.py` — SQLite connection handling:
  - `_audit_conn` replaced from module-level `sqlite3.connect(check_same_thread=False)` with `threading.local()` — each thread now gets its own connection, eliminating cross-thread access risks.
  - `audit_db_session` updated to yield the thread-local connection without a lock.
- `embeddings.py` — HTTPX client creation:
  - `_HTTPX_CLIENT` now guarded by `asyncio.Lock` — the TOCTOU between `is_closed` check and client creation is eliminated.
- `config.py` — configuration changes:
  - `KEEP_ALIVE_MAX_S` constant added (maps to `keep_alive: -1` sentinel).
  - `KEEP_ALIVE_UNITS` changed to frozenset read-only contract.
- `manifest.py` — manifest changes:
  - `no_kv_offload` allowlist added (KV cache in host RAM, model weights on GPU).
  - `kv_unified` cross-check with `parallel` and context size.
- `safety.py` — VRAM safety gate changes:
  - Parallel-aware VRAM fit check: accounts for per-slot compute when `parallel > 1`.
  - CPU-MoE-aware VRAM gating: trusts measured `expected_vram` for expert-offload configs.
  - `_vram_budget` refactored for multi-slot correctness.
- `fsm.py` — FSM changes:
  - Fast-fail on `wait_until_healthy` when child is dead — ends the LOADING-wedge in ~2 s instead of 600 s.
- `slot.py` — slot changes:
  - `thread_id` derivation from client IP (stable per-agent identity in single-residency mode).
  - Full-prompt `thread_id` keying for fan-out correctness.
- `live_monitor.py` — live inference monitor:
  - VRAM bars typecheck fix (guard against null VRAM in idle state).
  - ErrorBoundary added to frontend to prevent blank-screen crashes.
- `blob_store.py` — blob store:
  - `no_kv_offload` allowlist added.
- `config.py` — config:
  - `keep_alive: -1` now maps to `KEEP_ALIVE_MAX_S` (prevents -1 leaking as a literal duration).
  - `KEEP_ALIVE_UNITS` changed to frozenset (read-only contract).
- `subprocess_mgr.py` — subprocess management:
  - `wait_until_healthy` fast-fail on dead child.
  - PID-recycle guard on `_reap_booting_pid`.
- `ssrf_guard.py` — SSRF guard:
  - No changes (already hardened).

### Fixed

- **Security: Jinja RCE in `chat_template_kwargs`** — user-controlled `chat_template_kwargs` was passed directly to Jinja `Environment.from_string()`, allowing arbitrary code execution via crafted template strings. Fixed by sanitizing the input through a Jinja sandbox.
- **Security: `REDACTED_KEYS` mutability** — `EventBus.REDACTED_KEYS` was a mutable `set`; any code path could add/remove keys at runtime, weakening the redaction denylist. Changed to `frozenset`.
- **Security: `_KEEP_ALIVE_UNITS` mutability** — `_KEEP_ALIVE_UNITS` was a mutable `set`; same class of issue. Changed to `frozenset` read-only contract.
- **HTTPX client leak** — `_HTTPX_CLIENT` in `embeddings.py` was not closed before creating a new one; under rapid reconnects, leaked clients accumulated. Fixed: `close()` called before new client creation.
- **HTTPX client TOCTOU** — `_HTTPX_CLIENT.is_closed` check and client creation were not atomic; two concurrent requests could both see `is_closed=True` and both create new clients, leaking one. Fixed with `asyncio.Lock`.
- **SQLite thread-safety** — `_audit_conn` was a module-level `sqlite3.connect(check_same_thread=False)` shared across all threads. `check_same_thread=False` suppresses the thread-safety warning but does NOT make concurrent access safe. Fixed: each thread gets its own connection via `threading.local()`.
- **Payload mutation** — `chat_completion.py` mutated the caller's `payload` dict in-place (setting `response_format = None`). The original request body was modified, breaking retry logic. Fixed: `payload = dict(payload)` before mutation.
- **TOCTOU: `is_alive()` before `_sigterm`** — the `finally` block in `_process_slot` checked `is_alive()` before calling `_sigterm`; the process could die between the check and the sigterm, causing a missed cleanup. Fixed: `_sigterm` called unconditionally (with its own internal safety).
- **TOCTOU: handle reads in `_live_handle_pids`** — the handle was read, then checked, then used again; the handle could change between reads. Fixed: handle captured into a local variable before checks.
- **TOCTOU: handle reads in `warm_inherit`** — same pattern as in the warm-inherit path. Fixed the same way.
- **AsyncIO deprecations** (/35) — `asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()` throughout (the former is deprecated in Python 3.10+ and will be removed).
- **`ready_task.cancel()` not awaited** — `cancel()` returns a coroutine that must be awaited for the cancellation to take effect. Fixed.
- **Fire-and-forget idle teardown logging** — idle teardown tasks were created without error handling; failures were silently swallowed. Fixed with proper try/except logging.
- **Exception handling: `watch_disconnect` broad except** — the except clause caught all exceptions including `KeyboardInterrupt` and `SystemExit`. Narrowed to the specific `CancelledError` class.
- **Exception handling: `CancelledError` not re-raised** — `CancelledError` was caught and swallowed in several paths. Now re-raised correctly.
- **Exception handling: `boot_reconcile` broad except** — same issue as in the boot reconcile path.
- **Exception handling: `ws_state` broad except** — same issue in the WebSocket state endpoint.
- **Exception handling: `manifest read` broad except** — same issue in manifest reading.
- **`all_safety_gates` blocking the event loop** — `all_safety_gates` was called directly in an async context; it performs I/O (VRAM queries, disk checks) that blocks the event loop. Fixed: moved to `asyncio.to_thread()`.
- **`keep_alive: -1` leaking as literal** — the sentinel value -1 was not mapped to `KEEP_ALIVE_MAX_S`; it was passed through as-is, causing the keep_alive timer to be set to -1 seconds. Fixed: -1 now maps to the constant.
- **`is False` → `not`** — PEP-8 style fix for identity comparisons.
- **VRAM total in megabytes** — `vram_total_mib` was not surfaced in `/status` or the frontend; added.
- **WebSocket double-poll debounce** — the WS state endpoint had a double-poll race; fixed with debounce logic.
- **`enqueue_head` STAGED→STAGED regression** — the queue FSM was routing STAGED→STAGED as a no-op, dropping the slot. Fixed: correct routing.
- **VRAM null crash in idle state** — the frontend crashed when VRAM data was null in the idle state. Fixed with a guard + ErrorBoundary.
- **Frontend: ResidentsPanel `data is not defined` crash** — `vramTotal` was passed as a prop but `data` was out of scope in the ResidentsPanel component. Fixed: `vramTotal` threaded as a prop.

### Removed


### Known issues / limitations

- Multi-slot concurrency is Phase-0 scaffolded: `MAX_PARALLEL_SIDECARS = 1` (single sidecar). The resident registry and dispatcher are wired but the registry is pinned to one resident. Full multi-slot concurrency is a Phase-1 item.
- The `no_kv_offload` flag requires GPU support for KV-cache offload to host RAM; not all architectures support this.

---

## [0.4.0] — 2026-06-21

### Highlights

- **Live inference monitor** — a built-in, real-time view of the active generation: tokens/sec, prompt/decode progress, and a live output stream, independent of any external client or logger. The `/status` response gains a `generation` block (sampled ~1 Hz from the running sidecar) and a new Server-Sent-Events endpoint `GET /ui/live/output/stream` tees the model's output as it is produced. Surfaced in the frontend as a Live / Dashboard view.
- **Per-model concurrent dispatch** — each loaded model now serves according to its own concurrency setting. A main chat model can run strictly in series (one request at a time) while a sub-agent model serves up to *N* requests **concurrently** (`parallel: N`) on a single warm `llama-server` sidecar, via a fan-out that admits up to *N* riders with a drain-before-swap guard. The single-model-resident invariant is preserved — concurrency is *within* the active model, never across models.
- **KV-cache offload to host RAM** — with `no_kv_offload`, model weights stay on the GPU while the KV cache lives in system RAM. This lets a large-context model (e.g. 250K tokens) run on a 24 GB card and — combined with `kv_unified: true` — **serve requests in parallel while offloaded** (`parallel: 2` measured to fit and serve concurrently on a 24 GB card). See [docs/KV_CACHE_OFFLOADING.md](docs/KV_CACHE_OFFLOADING.md) (and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- **Live-panel stability + tool-call display** — the live inference panel no longer flickers between "generating" and "idle" during bursty multi-turn generation (a short grace-hold smooths the phase). Tool calls now render correctly in the live output, including models that emit tool-call arguments as a structured object instead of a JSON string.
- **Faster failure recovery** — a model load that crashes now fails its slot in ~2 s instead of waiting out the full load-health timeout, so a bad load no longer wedges the queue behind it.

### Added

- `src/turbohaul/live_monitor.py` + `src/turbohaul/api/live_stream.py` — the live inference monitor (1 Hz slots poller, live-output buffer, `/ui/live/output/stream` SSE endpoint), plus the `LiveInference` view and `useLiveOutput` hook in the frontend.
- Per-model concurrent dispatch — fan-out rider admission in the manager, model-affinity batching in the queue (prefer the already-warm model when selecting the next request), and a parallel-aware safety gate.
- [docs/KV_CACHE_OFFLOADING.md](docs/KV_CACHE_OFFLOADING.md) — a dedicated KV-cache offload explainer: the mechanism, the VRAM-vs-RAM trade-off, the sizing math, parallel-while-offloaded, and the decode perf cost (~2.5× on the measured 35B-class example).

### Changed

- The VRAM-fit safety gate is parallel-aware: it accounts for per-slot compute when a model is configured for concurrent serving, and refuses conservatively when it cannot measure free VRAM under `parallel > 1`.
- Manifest validation cross-checks `parallel`, `kv_unified`, and context size (each slot's context window must clear a minimum, and the total context must divide evenly across the slots).

---

## [0.3.8] — 2026-06-15

### Highlights

- **Per-model idle-unload** — each model's own `sleep_idle_seconds` from its manifest is now honored (previously the global setting was used). This lets short-lived sub-agent models unload quickly while the main chat model stays warm.
- **Live-streaming `generation_id` fix** — the frontend now correctly shows the live generation for the active request (not a stale generation from a previous request).
- **Idle countdown** — the frontend now shows a countdown timer while in the idle-warm state, so the user knows when the model will unload.

### Changed

- The frontend idle countdown timer is now accurate and matches the backend `sleep_idle_seconds` per model.
- The live-streaming `generation_id` is now updated correctly when a new request starts.

---

## [0.3.7] — 2026-06-14

### Highlights

- **Per-model live boxes + tok/s chart** — the frontend now shows one live box per loaded model, with a token-speed sparkline chart, kill-button blink, and VRAM-used readout.
- **FE split-view** — the frontend has a split-view layout with a residents panel on the left and live output panes on the right.

### Added

- FE split-view: residents panel + per-gen live output panes.
- FE integration gate + in-image FE build + smoke profile.

---

## [0.3.6] — 2026-06-13

### Highlights

- **Layer-2 per-model parallelism** — continuous fan-out from the inbox fixes serialized concurrent requests. Requests to the same model are now dispatched concurrently within the model's concurrency limit.
- **CPU-MoE-aware VRAM gating** — the VRAM fit check now trusts the measured `expected_vram` for expert-offload configurations (live-E2E co-residence fix).

---

## [0.3.5] — 2026-06-12

### Highlights

- **Multi-slot dispatcher + per-resident concurrency core** — the dispatcher routes incoming slots to available residents based on their concurrency capacity. Default concurrency is 1 (single-resident).

---

## [0.3.4] — 2026-06-11

### Highlights

- **Multi-slot state-migration mirror** — the manager's FSM state is now mirrored per-resident, enabling the multi-slot refactor.

---

## [0.3.3] — 2026-06-10

### Highlights

- **Resident-registry scaffold** — the foundational data structure for multi-slot tracking. Pinned to a single resident for backward compatibility.

---

## [0.3.2] — 2026-06-09

### Highlights

- **Flap/degradation telemetry** — the telemetry subsystem was added: persistent JSONL log, ring buffer, and read endpoints.

---

## [0.3.1] — 2026-06-08

### Highlights

- **Keep-alive during stream-OPEN** — keep-alive comment frames are now emitted during the prefill-open phase.

---

## [0.3.0] — 2026-06-07

### Highlights

- **Warm-slot reuse with per-client IP affinity** — stable per-agent `thread_id` from client IP, warm-slot + KV cache reuse.

---


## [v0.2.3] — 2026-05-19

### Highlights

- **Tool-call recovery for jinja-templated GGUFs** — transparent post-processor restores structured `tool_calls` when `llama-server` jinja templates (notably Qwen3-family) emit calls as JSON text inside `message.content` instead of populating the structured field. See [docs/TOOL_CALL_HANDLING.md](docs/TOOL_CALL_HANDLING.md).
- **`/v1/chat/completions` tools-field forwarding fixed** — previously the OpenAI endpoint silently dropped `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta`, making the recovery layer unreachable on the OpenAI surface. Now mirrors the `/api/chat` Ollama pattern.
- **Multi-agent GPU sharing** — three clients (one 27b chat client plus two advisor clients) serialize cleanly through one Blackwell-class GPU across a 27b -> 35b -> 27b model-swap exercise. See [docs/MULTI_AGENT_SHARING.md](docs/MULTI_AGENT_SHARING.md).
- **Persistence migration** to a host bind-mount for `/var/lib/turbohaul`. State + manifests + blobs now survive `docker rm` and container-layer corruption. A new image tag bakes the runtime updates into the image layer.
- **`response_format` validator** — `json_object` pass-through plus `json_schema` FULL validate + retry + thinking-strip + security mods.
- **`/v1/embeddings`** — llama-server embeddings passthrough.
- **`/v1/logging`** — paginated audit-event endpoint, 20K-token envelope budget, recursive REDACTED scrub.
- **Logs tab + Schema editor in the frontend**.
- **Client-disconnect queue eviction** + background terminal-park sweeper.

### Added

- `src/turbohaul/api/tool_call_recovery.py` — `maybe_recover_tool_calls` post-processor (287 LoC) handling OpenAI canonical `{"name":..,"arguments":..}` shape + Qwen `<tool_call>...</tool_call>` XML wrapper. Reasoning-guard (only scans content after `</think>`), parallel-call support (finditer + brace-balancer for nested args), idempotent skip when upstream populates `tool_calls`, name-allowlist gate against hallucinated tool names.
- `tests/test_tool_call_recovery.py` — 12 test functions / 18 sub-cases / 18-18 GREEN on host pytest.
- `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` keys added to the `/v1/chat/completions` `client_meta` dict in `chat_completion.py`.
- `docs/TOOL_CALL_HANDLING.md` — user-facing doc covering the two wire paths, the recovery post-processor mechanism, the closure-fix history, and testing. (this release)
- `docs/MULTI_AGENT_SHARING.md` — multi-agent serialization architecture + worked example.
- `docs/TURBOQUANT_FLAGS.md` — flag doctrine for production manifests, spawn-vs-request distinction, patching + verification recipes.
- `CHANGELOG.md` — this file. (this release)
- response_format validator — `json_object` MVP + `json_schema` FULL with validate + retry + thinking-strip.
- `/v1/embeddings` BE endpoint.
- `/v1/logging` paginated audit-event endpoint.
- Logs tab in the React frontend — paginated audit feed with REDACTED banner + auto-refresh.
- Schema editor + `responseFormatValidator` in the frontend.
- Client-disconnect queue eviction — slot gets evicted when client closes the connection mid-flight.
- Periodic terminal-park sweeper background task — sync finalize for STAGED + pid=NULL rows older than 24h via off-hot-path DB session.
- Ollama tool-call compat batch on `/api/chat` — `tool_calls` passthrough + done_reason map + lenient JSON fallback on malformed args + `MAX_TOOL_ARG_CHARS = 262144` cap.
- TurboQuant cache types `turbo2` / `turbo3` / `turbo4` allowed on KV cache.
- `audit_db_session` connection pool + `_audit_async` wrapper.
- ACTIVE_MATCH-streaming integration test.

### Fixed

- `/v1/chat/completions` silently dropping `tools` / `tool_choice` / `parallel_tool_calls` / `function_call` / `functions` from `client_meta` toward `llama-server` and into the recovery post-processor.
- Doc corrections:
  - `/api/admin/unload` claim replaced with the three real cold-spawn paths (Option A `keep_alive: 0` per-request body, Option B natural IDLE_HOT teardown, Option C `docker restart`). The `/api/admin/unload` endpoint does not exist.
  - Multi-agent claim sharpened to "multiplexed serialization" rather than "concurrent execution" — Turbohaul time-slices on a single GPU slot, not parallel tensor execution.

### Changed

- Image tag bumps: `turbohaul-manager:v0.2.2` -> `v0.2.3` references in README + AI_AGENT_SETUP recipes.
- Bind-mount migration baked into the persistent image (`v0.2.3` CUDA bind-mount variant).
- Auto-recovery script updated to reference the new tag.

### Known issues / limitations

- `jinja: true` in the model's manifest is still required for any tool-call work. Tool-call recovery (above) catches the case where jinja + Qwen3 emits as text-JSON; it does not synthesize calls when the model never emits anything tool-call-shaped.
- Multi-residency (two models in VRAM simultaneously) is not supported in v0.2.x. Single-slot serialization is the v0.2 invariant; multi-residency is a v0.3 roadmap item.
- `--reload` uvicorn mode is banned in production (it can reload code before a migration is applied). Production uses `docker restart turbohaul` for code changes.
- `image-vs-patches` debt: prior v0.2.x runtime updates were applied as `docker cp` overlays on the running container rather than baked into a new image. v0.2.3 closes this by baking the changes into the `v0.2.3` CUDA bind-mount image. Going forward, any non-trivial production deploy MUST `docker commit` + update auto-recovery references, OR rebuild from `Dockerfile.cuda` against the current source tree.

### Upgrade path

```bash
# Pull the new image (tag may differ depending on registry mirror)
docker pull ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.2.3

# Stop + remove the old container (state survives because of the bind-mount)
docker stop turbohaul
docker rm turbohaul

# Run the new container with the canonical bind-mount layout
docker run -d --name turbohaul \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    -p 11401:11401 \
    -p 11434:11434 \
    -v /var/lib/turbohaul:/var/lib/turbohaul \
    -e TURBOHAUL_IDLE_HOT_SECONDS=600 \
    -e TURBOHAUL_GRACE_SECONDS=30 \
    ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.2.3
```

Existing state (`state.sqlite`, `manifests/*.yaml`, `blobs/sha256/*`) is preserved through the bind-mount. First request to a new model may cold-load 30 to 60 seconds; subsequent same-thread follow-ups within the grace + IDLE_HOT windows reuse the warm slot.

---

## [v0.2.2] — earlier in May 2026

Initial public ship at `https://github.com/MrTrenchTrucker/turbohaul-manager`. See the git history at the `v0.2.2` tag for the full change set. v0.2.2 included the full management plane + CUDA Dockerfile + v0.2.1 bug-sweep waves.

---

## Contributors to this release

See [CONTRIBUTORS.md](CONTRIBUTORS.md). The tool-call recovery work and the `/v1/chat/completions` closure-fix, the endpoint batch, doc review, dependency-graph alignment, and release prep were contributed by the maintainers.
