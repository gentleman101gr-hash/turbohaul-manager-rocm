# Turbohaul-Manager — Architecture & Design (v0.5.0)

**Status:** v0.5.0 architecture — current state as of the v0.5.0 public release.
**Maintainer:** MrTrench.
**Repository:** `https://github.com/MrTrenchTrucker/turbohaul-manager`

---

## 1. Mission

Turbohaul-Manager is a **standalone HTTP inference server** that:

- Mimics the Ollama API surface (`/api/generate`, `/api/chat`, `/v1/chat/completions`, `/api/pull`, `/api/tags`, `/v1/embeddings`) so any Ollama-aware client can swap it in transparently.
- Uses **Tom's Fork TurboQuant llama.cpp** (`github.com/TheTom/llama-cpp-turboquant`, branch `feature/turboquant-kv-cache`, MIT) as its inference backend — a supervised `llama-server` subprocess per active sidecar.
- Provides **BYOM** (Bring-Your-Own-Model) blob storage. Pull from Ollama registry / HuggingFace (allowlist-pinned) / vetted URL / local-staging import.
- Provides a **FIFO request queue with grace + idle hot-load** that eliminates the cross-process sidecar race of the original manager.
- Provides **per-model concurrent dispatch** — each loaded model serves according to its own concurrency setting, with fan-out rider admission up to the model's `parallel` limit.
- Provides **KV-cache offload to host RAM** — model weights stay on GPU, KV cache lives in system RAM, enabling large-context models on smaller GPUs and parallel serving while offloaded.
- Provides **flap/degradation telemetry** — a low-overhead, append-only JSONL logging subsystem that captures the full request lifecycle and resource samples, with both in-memory (ring buffer) and persistent (rotating JSONL) read paths.

**Design principle:** One-stop-shop for local inference. No external dependencies beyond the llama-server binary.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Turbohaul-Manager                          │
│                                                               │
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐         │
│  │   Queue    │───→│ Dispatch  │───→│  Resident     │         │
│  │  (2-tier)  │    │  (per-   │    │  Registry     │         │
│  │            │    │  model)  │    │  (Phase-0:    │         │
│  │ Accept Buf │    │         │    │   1 resident)  │         │
│  │  + Staging │    │         │    │                 │         │
│  └───────────┘    └───────────┘    └───────┬───────┘         │
│                                            │                  │
│  ┌───────────┐    ┌───────────┐            │                  │
│  │  Safety   │    │  Live    │            │                  │
│  │  Gates    │    │  Monitor │            │                  │
│  │ (VRAM/    │    │ (1Hz     │            │                  │
│  │  RAM/     │    │  poller) │            │                  │
│  │  CPU)     │    │         │            │                  │
│  └───────────┘    └───────────┘            │                  │
│                                            │                  │
│  ┌───────────┐    ┌───────────┐            │                  │
│  │  Telemetry│    │  FSM     │            │                  │
│  │  (JSONL + │    │  (12     │            │                  │
│  │  ring     │    │  states) │            │                  │
│  │  buffer)  │    │         │            │                  │
│  └───────────┘    └───────────┘            │                  │
│                                            │                  │
│                                      ┌─────▼──────┐           │
│                                      │ Sidecar    │           │
│                                      │ (llama-    │           │
│                                      │  server)   │           │
│                                      └────────────┘           │
│                                                               │
│  ┌───────────────────────────────────────────────┐            │
│  │         API Layer (FastAPI + Uvicorn)          │            │
│  │  /v1/chat/completions  /api/chat  /api/tags    │            │
│  │  /v1/embeddings       /api/pull  /v1/telemetry │            │
│  │  /api/config          /ws/state  /ui/live      │            │
│  └───────────────────────────────────────────────┘            │
│                                                               │
│  ┌───────────────────────────────────────────────┐            │
│  │         Persistence Layer                      │            │
│  │  state.sqlite  manifests/*.yaml  blobs/sha256  │            │
│  │  telemetry/*.jsonl                         │            │
│  └───────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 TurbohaulManager (`manager.py`)

The top-level orchestrator. Responsibilities:

- **Boot reconcile:** orphan reap + foreign-GPU detect + state.sqlite slot cleanup
- **Binary verification:** sha256 pin at boot
- **Request acceptance:** `submit()` → push to queue (head if grace match)
- **Status snapshot:** `status_snapshot()` for `/status` endpoint
- **FSM driver:** `worker_loop` drives the state machine
- **Multi-slot resident registry:** Phase-0 scaffolded to a single resident (see §4)
- **Telemetry hooks:** lifecycle events emitted to the telemetry subsystem
- **Clean shutdown:** drain all inflight requests, then tear down

### 3.2 EventBus (`manager.py`)

Pub-sub for state-level events broadcast to `/ws/state` subscribers. Enforces a redaction denylist (`REDACTED_KEYS` frozenset: `prompt`, `response`, `context`, `stderr`, `stdout`, `messages`) on publish as defense-in-depth.

### 3.3 Queue (`queue.py`)

Two-tier queue:

- **Acceptance buffer:** capped at `acceptance_max` (default 10K). Receives all fresh requests. Never blocks the API caller until cap hit.
- **Staging queue:** capped at `staging_max` (default 100). FIFO.

On enqueue: slot goes to staging if room, else to acceptance buffer.
On pop: drain from staging head; buffer feeds staging tail when staging has room.

Model-affinity pop tuning: `max_consecutive_same_model` (default 3) bounds the run-length of one model the affinity path will cluster before forcing the FIFO head. `max_other_model_wait_s` (default 20.0) is the age past which a starved other-model head request forces a swap.

### 3.4 Safety Gates (`safety.py`)

Pre-spawn checks that refuse to spawn a sidecar when the host cannot safely run it:

- **VRAM check:** `nvidia-smi` free VRAM must exceed the model's `expected_vram` (with per-slot compute floor for `parallel > 1`)
- **RAM check:** `/proc/meminfo` MemAvailable must exceed the model's minimum
- **IO-wait check:** `/proc/diskstats` IO-wait must be below threshold
- **CPU load check:** 1-min load avg per logical core must be below threshold

All gates degrade gracefully: if the underlying probe is unavailable, the gate returns "passed-no-probe" rather than blocking. The entire subsystem can be disabled via `runtime.queue.safety_enabled = False`.

### 3.5 Live Inference Monitor (`live_monitor.py`)

A pure-observer plane that gives Turbohaul its own view of what the active sidecar is decoding, independent of any client CLI:

1. **Robust Core (stream-mode independent):** `LiveSlotsPoller` polls the active llama-server's `GET /slots` ~1 Hz, derives tok/s + progress from `next_token[0].n_decoded` deltas, and writes a redacted `generation` block into `mgr.live_generation` that `status_snapshot()` exposes.
2. **Thin Live-Text Tee:** `LiveOutputBuffer` holds per-generation bounded ring buffers fed by the streaming tee in `api/chat_completion.py` and read ONLY by the dedicated SSE endpoint in `api/live_stream.py`.

Both planes key on ONE unified `generation_id` = blake2b(pid:spawn_seq:thread).

### 3.6 Telemetry (`telemetry.py` + `api/telemetry.py`)

A low-overhead, append-only JSONL logging subsystem that captures the full request lifecycle:

- **Lifecycle hooks:** `on_request_arrival`, `on_queue_state`, `on_slot_assign`, `on_prefill_start`, `on_first_token`, `on_generation_tick`, `on_keep_alive_emitted`, `on_client_disconnect`, `on_completion`, `on_vram_sample`, `on_slot_state_change`
- **Rotating JSONL writer:** 10 MiB per file, 5 files retained, line-buffered
- **In-memory ring buffer:** 10K events, thread-safe
- **Read endpoints:** `GET /v1/telemetry/events` (paginated, cursor-based, source=ring or source=file) and `GET /v1/telemetry/status` (subsystem health + stats)

---

## 4. Multi-Slot Architecture (Phase-0)

### 4.1 Resident Registry

The manager tracks loaded-model sidecars through a per-model resident registry:

```python
class ResidentState(enum.StrEnum):
    RESERVED_LOADING = "RESERVED_LOADING"
    ACTIVE = "ACTIVE"
    GRACE = "GRACE"
    IDLE_EVICTABLE = "IDLE_EVICTABLE"
    DEAD = "DEAD"
```

Each `Resident` dataclass tracks:
- `model_tag`: the model this resident serves
- `handle`: the live `SidecarHandle`
- `port`: the sidecar's listen port
- `inflight`: the concurrent fan-out rider Slots (anchor at index 0)
- `spawn_seq`: monotonic spawn counter for fixed-port swap detection
- `idle_expires_at`: monotonic deadline of the warm-idle hold
- `active_slot`: the anchor Slot currently driven on this resident
- Per-resident mirrors of `_idle_handle`, `_idle_model_tag`, `_spawn_seq`, `_latest_keep_alive_s`

### 4.2 Dispatcher

The dispatcher routes incoming slots to available residents based on their concurrency capacity. The `_MAX_DISPATCH_DEFERS` constant (50) limits how many times an unroutable slot can be re-queued before failing with a 503-equivalent.

### 4.3 Phase-0 Invariants

- `MAX_PARALLEL_SIDECARS = 1` (pinned to single sidecar for backward compatibility)
- The registry holds exactly one entry under the `_SINGLETON_RESIDENT_KEY` sentinel
- Behavior is byte-for-byte identical to the v0.3.8 single-sidecar manager
- Phase-1 (dispatcher / driver-tasks / LRU-evict / multi-spawn) is where the registry becomes authoritative

### 4.4 Fan-Out Rider Admission

For `cap<=1` models, continuous rider-admit during drain with `_FANOUT_ADMIT_POLL_S` (0.1 s) bounded poll interval. The wake only does a cheap non-blocking `pop_next` while a fan-out is already active.

### 4.5 Warm-Slot Reuse

Stable per-agent `thread_id` derived from client IP (single-residency). Same-client follow-ups inherit the warm sidecar, killing the full-context re-prefill crawl. A gated cap (default ≤1) preserves the fan-out behavior at higher concurrency levels.

---

## 5. State Machine

### 5.1 Slot States (12 states)

```
RECEIVED → ACCEPT_BUFFER → STAGED → LOADING → LOADING_FAIL → ACTIVE → GRACE → GRACE_BUSY → ACTIVE_MATCH → POPPED → IDLE_HOT → COLD
```

- **RECEIVED:** Fresh request just arrived
- **ACCEPT_BUFFER:** In the acceptance buffer (staging full)
- **STAGED:** In the staging queue (FIFO)
- **LOADING:** Sidecar spawning, waiting for health
- **LOADING_FAIL:** Sidecar failed to load (retry or fail)
- **ACTIVE:** Sidecar serving this request
- **GRACE:** Warm grace window (same-thread follow-ups accepted)
- **GRACE_BUSY:** Matched follow-up running on warm slot
- **ACTIVE_MATCH:** Mid-stream matched thread queued at FIFO head
- **POPPED:** Request completed, slot freed
- **IDLE_HOT:** Warm idle hold (any same-model request accepted)
- **COLD:** Terminal — sidecar torn down, VRAM freed

### 5.2 Legal Transitions

```python
RECEIVED → {ACCEPT_BUFFER, STAGED}
ACCEPT_BUFFER → {STAGED, COLD}
STAGED → {LOADING, COLD, ACTIVE_MATCH}
LOADING → {ACTIVE, LOADING_FAIL, COLD}
LOADING_FAIL → {STAGED, POPPED}
ACTIVE → {GRACE, ACTIVE_MATCH}
GRACE → {GRACE_BUSY, POPPED, ACTIVE}
GRACE_BUSY → {GRACE, POPPED}
ACTIVE_MATCH → {ACTIVE}
POPPED → {IDLE_HOT, STAGED, COLD}
IDLE_HOT → {ACTIVE, POPPED, COLD}
COLD → {}  # terminal
```

### 5.3 Fast-Fail on Dead Child

A model load that crashes now fails its slot in ~2 s instead of waiting out the full load-health timeout (600 s). A dead child detected during `wait_until_healthy` immediately transitions the slot to DEAD.

---

## 6. Queue and Scheduling

### 6.1 Two-Tier Queue

- **Acceptance buffer (10K cap):** Receives all fresh requests. Never blocks the API caller until cap hit.
- **Staging queue (100 cap):** FIFO. Popped by the worker_loop for dispatch.

### 6.2 Model-Affinity Pop Tuning

- **`max_consecutive_same_model` (default 3):** Bounds the run-length of one model the affinity path will cluster before forcing the FIFO head (fairness). A value of 1 disables batching entirely (strict FIFO).
- **`max_other_model_wait_s` (default 20.0):** The age past which a starved other-model head request forces a swap, overriding affinity. 0.0 means "starve immediately" (strict FIFO).

### 6.3 Grace and Idle Timers

- **Grace timer (`grace_seconds`):** Window after slot completion where the model stays loaded for same-thread follow-ups. Default 30 s.
- **Idle hot-load timer (`idle_hot_load_seconds`):** Window after the entire queue drains where the last model stays warm for any same-model-tag fresh request. Default 600 s.
- **Keep-alive (`keep_alive`):** Per-request override. `keep_alive: -1` maps to `KEEP_ALIVE_MAX_S` (1800 s). Latest request wins (Ollama semantics).

### 6.4 Client-Disconnect Eviction

A slot is evicted when the client closes the connection mid-flight. The eviction counter (`_eviction_count`) and timestamp (`_last_evicted_at_iso`) are tracked for `/status` observability.

### 6.5 Background Sweeper

A background task finalizes STAGED + pid=NULL rows older than 24h via an off-hot-path DB session. This keeps the worker_loop off the SQLite fsync stall.

---

## 7. Manifest System

### 7.1 Per-Model Manifests

Each model has a YAML manifest in `manifests/` that defines its flags, context size, and other parameters. The manifest is the single source of truth for how the model is loaded.

### 7.2 Flag Allowlist

A closed allowlist of safe llama-server flags (`SAFE_LLAMA_FLAGS`, ~80 entries) prevents injection via user-provided manifests. Adding a new flag requires a code change + review — the YAML cannot smuggle it in.

Denied flags include path-bearing/RCE risks: `model_url`, `hf_repo*`, `api_key_file`, `ssl_*`, `path`, `media_path`, `tools`, `control_vector*`, `lookup_cache_*`.

### 7.3 Numeric Bounds

`SAFE_LLAMA_FLAG_BOUNDS` prevent DoS-by-extreme (e.g., `n_gpu_layers: 999999`).

### 7.4 Atomic Writes + ETag/If-Match

Manifests are written atomically (write to temp file, rename). ETag/If-Match prevents lost-updates when multiple clients edit concurrently.

### 7.5 KV Cache Controls

- **`no_kv_offload`:** KV cache in host RAM, not VRAM. Enables large-context models on smaller GPUs.
- **`kv_unified`:** Unified KV cache (all slots share one cache).
- **`cache_type_k` / `cache_type_v`:** KV cache quantization type.

---

## 8. Safety Gates

### 8.1 Pre-Spawn Checks

Before spawning a sidecar, the manager checks:

1. **VRAM:** `nvidia-smi` free VRAM must exceed the model's `expected_vram` (with per-slot compute floor for `parallel > 1`)
2. **RAM:** `/proc/meminfo` MemAvailable must exceed the model's minimum
3. **IO-wait:** `/proc/diskstats` IO-wait must be below threshold
4. **CPU load:** 1-min load avg per logical core must be below threshold

### 8.2 Parallel-Aware VRAM Fit

The VRAM-fit safety gate is parallel-aware: it accounts for per-slot compute when a model is configured for concurrent serving, and refuses conservatively when it cannot measure free VRAM under `parallel > 1`.

### 8.3 CPU-MoE-Aware VRAM Gating

The VRAM fit check now trusts the measured `expected_vram` for expert-offload configurations (live-E2E co-residence fix).

### 8.4 Degradation

If a probe is unavailable (e.g., `nvidia-smi` missing in dev), the gate returns "passed-no-probe" rather than blocking. The entire subsystem can be disabled via `runtime.queue.safety_enabled = False`.

---

## 9. Telemetry Subsystem

### 9.1 Architecture

The telemetry subsystem is a low-overhead, append-only logging system:

1. **Ring buffer (10K events, in-memory):** Hot read path for real-time dashboards
2. **Rotating JSONL files (10 MiB per file, 5 files retained):** Persistent full-history read path
3. **Lifecycle hooks:** 11 event types covering the full request lifecycle
4. **Read endpoints:** `GET /v1/telemetry/events` and `GET /v1/telemetry/status`

### 9.2 Event Types

- `request_arrival`: When a new request arrives at the API layer
- `queue_enter`: When a slot enters the queue (staging or acceptance buffer)
- `slot_assign`: When a slot is assigned to an active sidecar
- `prefill_start`: When prefill begins for a slot
- `first_token`: When the first token is received (TTFT measurement)
- `generation_tick`: Periodically during generation (throttled)
- `keep_alive`: When a keep-alive comment is emitted during stream-OPEN
- `client_disconnect`: When a client disconnects mid-flight
- `completion`: When a slot completes successfully
- `resource_sample`: Periodic VRAM + memory state samples
- `slot_state_change`: When a slot transitions between states

### 9.3 Read Endpoints

- **`GET /v1/telemetry/events`:** Paginated, cursor-based. Parameters: `since` (monotonic sequence cursor), `event_type` (filter), `limit` (1-1000), `source` (ring or file).
- **`GET /v1/telemetry/status`:** Telemetry subsystem health + stats.

---

## 10. API Surface

### 10.1 Ollama-Compatible Endpoints

- `POST /api/chat` — Chat completion (Ollama shape)
- `POST /api/generate` — Raw generation (Ollama shape)
- `GET /api/tags` — List loaded models
- `POST /api/pull` — Pull a model from the Ollama registry

### 10.2 OpenAI-Compatible Endpoints

- `POST /v1/chat/completions` — Chat completion (OpenAI shape)
- `POST /v1/embeddings` — Embeddings (OpenAI shape)
- `GET /v1/telemetry/events` — Telemetry events
- `GET /v1/telemetry/status` — Telemetry status

### 10.3 Management Endpoints

- `GET /status` — Manager status snapshot (including live inference data)
- `PUT /api/config` — Runtime-mutable configuration
- `GET /api/config` — Current configuration
- `GET /v1/logging` — Paginated audit-event endpoint

### 10.4 WebSocket Endpoints

- `GET /ws/state` — Real-time state events (redacted)

### 10.5 Frontend Endpoints

- `GET /ui/live/output/stream` — Server-Sent Events for live output
- Static files served from the built frontend

### 10.6 Tool-Call Recovery

Transparent post-processor restores structured `tool_calls` when `llama-server` jinja templates (notably Qwen3-family) emit calls as JSON text inside `message.content` instead of populating the structured field. See [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md).

### 10.7 Response Format Validation

- `json_object` pass-through
- `json_schema` FULL validate + retry + thinking-strip

---

## 11. Frontend

### 11.1 React Frontend

A React frontend is built into the Docker image at build time. The frontend provides:

- **Dashboard view:** Live inference monitor with tok/s, progress, and VRAM data
- **Residents panel:** Per-model live boxes with tok/s sparkline, kill-button, VRAM-used readout
- **Live output panes:** Real-time token output per generation
- **Logs tab:** Paginated audit feed with REDACTED banner + auto-refresh
- **Schema editor:** Response format validation configuration

### 11.2 Live Inference Monitor

The frontend shows a real-time view of the active generation: tokens/sec, prompt/decode progress, and a live output stream. The `/status` response gains a `generation` block (sampled ~1 Hz from the running sidecar).

### 11.3 Error Boundary

The frontend has an ErrorBoundary to prevent blank-screen crashes when VRAM data is null in the idle state.

---

## 12. Persistence

### 12.1 State Database

SQLite database at `/var/lib/turbohaul/state.sqlite`. Tracks:

- Active slots and their states
- Audit events (redacted)
- Model manifest references
- Resident registry (Phase-0)

### 12.2 Blob Store

Model files stored at `/var/lib/turbohaul/blobs/sha256/`. SHA256-pinned by the manifest system.

### 12.3 Manifests

Per-model YAML manifests at `/var/lib/turbohaul/manifests/`. Atomic writes with ETag/If-Match concurrency control.

### 12.4 Telemetry Logs

Rotating JSONL files at `/var/lib/turbohaul/telemetry/`. 10 MiB per file, 5 files retained.

### 12.5 Bind-Mount Layout

The canonical bind-mount layout is:

```
/var/lib/turbohaul/
├── state.sqlite
├── manifests/
├── blobs/
└── telemetry/
```

This survives `docker rm` and container-layer corruption.

---

## 13. Configuration

### 13.1 Boot Configuration

Requires restart to change:

- `server.host` — bind address (default `127.0.0.1`; `0.0.0.0` requires `--allow-public-bind`)
- `server.port` — bind port (default 11401)
- `storage.blob_store_path` — blob storage directory
- `storage.manifests_path` — manifests directory
- `storage.state_db_path` — state database path
- `runtime.llama_server_binary` — path to the llama-server binary
- `runtime.llama_server_binary_sha256` — SHA256 pin (empty = skip verify, dev only)
- `runtime.default_port_base` — child process port base (default 11500)

### 13.2 Runtime Configuration

Mutable via `PUT /api/config`:

- `queue.staging_queue_depth` (default 100)
- `queue.acceptance_buffer_max` (default 10K)
- `queue.max_consecutive_same_model` (default 3)
- `queue.max_other_model_wait_s` (default 20.0)
- `queue.grace_seconds` (default 30)
- `queue.idle_hot_load_seconds` (default 600)
- `queue.safety_enabled` (default true)
- `queue.max_parallel_sidecars` (default 1, Phase-0 pinned)

### 13.3 Environment Variables

- `TURBOHAUL_IDLE_HOT_SECONDS` — override idle_hot_load_seconds
- `TURBOHAUL_GRACE_SECONDS` — override grace_seconds

---

## 14. Security

### 14.1 Threat Model

**In-scope:** Trusted network. All callers are known clients on the local perimeter.

**Out-of-scope:** External untrusted users, public internet exposure.

### 14.2 Bind Invariant

`server.host` defaults to `127.0.0.1`. Binding to `0.0.0.0` requires `--allow-public-bind` CLI flag AND logs a warning. Never `0.0.0.0` by default.

### 14.3 Singleton Invariant

Turbohaul-Manager MUST be the only writer to GPU 0 on a given host.

Enforcement:
- At boot, `fcntl.flock` on `state.sqlite`. If held by another process, refuse to start.
- At boot, scan `nvidia-smi --query-compute-apps=pid,used_memory` on GPU 0. If foreign `llama-server` processes are present AND not in the state.sqlite reconciliation map, refuse to start (or, with `--adopt-orphans`, kill them after warning).
- Boot-time orphan reaper scans for `llama-server` children with `parent=1` and port in `runtime.default_port_base` range.

### 14.4 Redaction

The EventBus enforces a redaction denylist on publish: `prompt`, `response`, `context`, `stderr`, `stdout`, `messages` are stripped from all events.

### 14.5 Flag Injection Prevention

The closed allowlist of safe llama-server flags prevents injection via user-provided manifests. Adding a new flag requires a code change + review.

### 14.6 Jinja RCE Prevention

User-controlled `chat_template_kwargs` is sanitized through a Jinja sandbox. The original vulnerability allowed arbitrary code execution via crafted template strings.

### 14.7 Auth Posture (v0.5.0)

No app-layer auth — matches project-wide network-perimeter posture. All endpoints trust the network perimeter. Future versions may add bearer-auth on mutating endpoints.

---

## 15. Docker Images

### 15.1 Build Targets

- **`Dockerfile.cuda`** — CUDA variant (Blackwell or older NVIDIA GPU)
- **`Dockerfile.cuda-multi`** — Multi-slot CUDA variant (Phase-0)

### 15.2 Image Tags

- `turbohaul-manager:v0.5.0` — Current release
- `turbohaul-manager:v0.3.8` — Known-good baseline for multi-slot refactor

### 15.3 Production Deployment

```bash
docker run -d --name turbohaul \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    -p 11401:11401 \
    -v /var/lib/turbohaul:/var/lib/turbohaul \
    -e TURBOHAUL_IDLE_HOT_SECONDS=600 \
    -e TURBOHAUL_GRACE_SECONDS=30 \
    ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.5.0
```

---

## 16. Roadmap

### 16.1 Phase-1 (Multi-Slot)

- Raise `MAX_PARALLEL_SIDECARS` beyond 1
- Make the resident registry authoritative for the live FSM
- Implement per-resident driver tasks
- LRU eviction of idle residents
- Multi-spawn with proper concurrency control

### 16.2 Phase-2 (Hardened Public Release)

- Bearer-auth on all mutating endpoints
- TLS termination
- External-user-facing deployment support
- Open WebUI integration

### 16.3 Ongoing

- Instrument production traffic to tune grace/idle defaults
- Add `tensor_split` to the flag allowlist (requires CSV-string parser)
- Add `fit_target` to the flag allowlist (requires CSV-string parser)

---

## 17. Cross-References

- [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md) — Tool-call recovery mechanism
- [MULTI_AGENT_SHARING.md](MULTI_AGENT_SHARING.md) — Multi-agent serialization architecture
- [KV_CACHE_OFFLOADING.md](KV_CACHE_OFFLOADING.md) — KV-cache offload explainer
- [PREFIX_CACHE_REUSE.md](PREFIX_CACHE_REUSE.md) — KV-cache prefix reuse for fast multi-turn agents
- [TURBOQUANT_FLAGS.md](TURBOQUANT_FLAGS.md) — flag doctrine for production manifests
- [telemetry_reader_guide.md](telemetry_reader_guide.md) — Telemetry subsystem guide
- [CHANGELOG.md](../CHANGELOG.md) — Version history
- [CONTRIBUTORS.md](../CONTRIBUTORS.md) — Contributors

---

*End of Architecture & Design (v0.5.0)*
