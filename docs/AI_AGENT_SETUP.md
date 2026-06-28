# Turbohaul-Manager — AI Agent Setup Guide

**Audience:** Developers connecting an AI agent (Hermes, LiteLLM-routed, langchain, llama-index, raw OpenAI SDK, Ollama-client, etc.) to a Turbohaul-Manager instance.

**Goal:** Hermes-class **multi-tool-call** agent loops work out of the box, slot-state survives many tool-call turns, your client timeouts are set so long prefills don't "flap" the connection, and you don't have to read Turbohaul's source to wire it up.

---

## TL;DR — The Two-Line Setup

Most agents need this and only this:

```yaml
base_url: http://<turbohaul-host>:11401/v1
api_key: dummy   # any string — Turbohaul doesn't require auth on the internal port
```

If your agent is OpenAI-API-shaped (Hermes, OpenAI Python SDK, langchain `ChatOpenAI`, llama-index `OpenAI`, etc.), point it at `:11401/v1`. If it's Ollama-shaped, point it at `:11434` (no `/v1` suffix). Both surfaces talk to the same container and the same slot lifecycle.

That's the entire setup. Sane defaults are baked in: idle-hot keeps slots warm 10 min between turns, ACTIVE_MATCH reuses the warm process for same-`thread_id` follow-ups, tool-call fields pass through, and streaming SSE is supported end-to-end.

The rest of this doc is for when "just works" doesn't, or when you want to tune.

---

## What Turbohaul gives your agent — for free

You don't have to set any of these; they're the default. Listed so you can recognize the behavior you'll see in logs:

| Behavior | Default | What it does for your agent |
|---|---|---|
| `idle_hot_load_seconds` | `600` (10 min) | After a request completes, the slot stays warm. The next request to the same model within 10 min skips cold-load (saves ~30-60s on a 27B GGUF). |
| `grace_seconds` | `30` | After a request completes, the slot holds for another 30s before transitioning to IDLE_HOT. Within this window, **ACTIVE_MATCH cascade** warm-reuses the slot for same-thread follow-ups (sub-second handoff). |
| `keep_alive` | client-overridable | If your agent sends Ollama-style `keep_alive: "10m"` or `1800` (seconds), Turbohaul honors it as the IDLE_HOT extension (capped at 30 min). `keep_alive: 0` = unload immediately. `keep_alive: -1` = pin (max cap). Single line in `extra_body` for OpenAI-SDK clients (see Hermes §5 below). |
| Streaming SSE pass-through | always on | When you send `stream: true`, Turbohaul opens its own `httpx.stream()` to llama-server and pipes raw SSE chunks back. The 12-second keep-alive heartbeat keeps your client socket warm during cold-load. |
| Tool-call fields | pass through | `tools`, `tool_choice`, `parallel_tool_calls`, `function_call`, `functions` forwarded verbatim to llama-server on BOTH `/v1/chat/completions` and `/api/chat`. Works on any model whose manifest sets `jinja: true`. |
| Tool-call recovery | transparent post-processor | When a jinja-templated model (notably Qwen3-family per upstream llama.cpp issues #20809 / #20837 / #20260) emits a tool call as text JSON inside `message.content` instead of populating `message.tool_calls`, Turbohaul extracts and restores it into the structured field, flips `finish_reason` to `tool_calls`, and strips the matched JSON from `content`. Idempotent — no-op when upstream already populates correctly. See [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md). |
| Thinking models | reasoning preserved | Qwen3.6, DeepSeek-R1, and similar thinking models get their `<think>...</think>` blocks PLUS structured `reasoning_content` deltas in streaming responses. No client-side merging needed. |
| Per-thread warm reuse | via `thread_id` | If you include a `thread_id` field in the request body, same-thread follow-ups within the grace window hit ACTIVE_MATCH (warm slot, no re-spawn). Many OpenAI-SDK clients can pass arbitrary fields via `extra_body`. |
| Safety guardrails | always on | Pre-spawn VRAM/RAM/CPU/IO-wait checks refuse to spawn into an OOM or IO-stuck host (returns HTTP 503 + `Retry-After` rather than crashing the box). |

If your agent gets a 503 with `Retry-After`, that's a safety refusal — back off and retry; nothing's broken.

---

## Per-Agent Setup

### 1. Hermes Agent (validated reference)

Hermes is a supported reference agent — an OpenAI-SDK-shaped multi-tool-call harness. Tested end-to-end with Turbohaul up to N≥3-tool agent loops. The config below front-loads the two things that matter most against Turbohaul: **generous timeouts** (so a long prefill is never killed — see *Anti-flap* below) and a **main + auxiliary model** split (see *Dual-model* below).

`hermes-config/config.yaml`:

```yaml
model:
  default: dense-27b-mtp               # MAIN model — your highest-reasoning model
  context_length: 250000
  max_tokens: 16384                    # >= 2000 for thinking models so chain-of-thought has room
  provider: custom
  base_url: http://<turbohaul-host>:11401/v1
  api_key: dummy

providers:
  custom:
    request_timeout_seconds: 7200      # must outlast a long large-context prefill (see Anti-flap)
    stale_timeout_seconds: 7200        # CRITICAL — a short value here causes flapping (see Anti-flap)
    base_url: http://<turbohaul-host>:11401/v1
    api_key: dummy
    api_mode: openai                   # speaks /v1/chat/completions

agent:
  max_turns: 250                       # high — the default (~90) hard-stops a long agent mid-task (see max_turns)
  gateway_timeout: 7200
  api_max_retries: 240                 # 240 * 30s = ~2h retry ceiling
  reasoning_effort: high               # harness-internal planning verbosity (NOT a Turbohaul knob)

# AUXILIARY model — context compression AND sub-agent delegation (see Dual-model)
auxiliary:
  default:
    model: moe-35b-mtp                 # AUX — an MoE at parallel:2 so sub-agents share one warm slot
    base_url: http://<turbohaul-host>:11401/v1
    timeout: 600
```

**Important nuance about `streaming: false`**: Hermes' OpenAI Python SDK chat-completions code path **always emits `stream: true` on the wire** (the `streaming: false` config flag is for Hermes' display layer, not the request body). Turbohaul handles this correctly — the streaming path is the validated multi-tool-call code path.

If you also want to send Ollama-only fields (like `keep_alive`), use `auxiliary.<task>.extra_body`:

```yaml
auxiliary:
  vision:
    extra_body:
      keep_alive: "10m"
```

The `extra_body` map is merged into the request JSON Hermes sends. Per [Ollama Issue #11458](https://github.com/ollama/ollama/issues/11458), this is the only way OpenAI-SDK clients can express `keep_alive` — the SDK doesn't natively support it.

#### Anti-flap: set the client timeouts high

The single most common cause of an agent "flapping" against Turbohaul — connecting, hanging, disconnecting, retrying in a loop — is a **client-side timeout that is shorter than a prefill**. A large-context model prefilling tens of thousands of tokens can take **minutes** before it emits its first byte. If the client's stale/idle timeout fires during that prefill, the client tears down the connection mid-work; Turbohaul finishes (or evicts) the slot, the client retries, and the cycle repeats — looking like instability when nothing is actually broken.

Set both of these on the provider to comfortably outlast any prefill:

- **`stale_timeout_seconds: 7200`** — the idle/no-byte timeout. This is the one that bites: keep it high. Turbohaul emits keep-alive comment frames during the prefill-open phase, but the client must be willing to wait through them.
- **`request_timeout_seconds: 7200`** — the overall socket timeout. Same reasoning.
- **`gateway_timeout: 7200`** — if your harness has a separate gateway/agent-loop timeout, raise it too.

7200 s (2 h) is deliberately generous; the point is that the ceiling should never be the thing that ends a request. Turbohaul's own slot lifecycle (grace + idle-hot) governs when a model is actually freed.

#### max_turns: raise it for long agent loops

A multi-tool-call agent doing real work (audits, multi-file edits, research) can easily run dozens of tool-call turns. If the harness caps turns too low (defaults are often ~90), it **hard-stops mid-task** — the run just ends with work unfinished. Raise `max_turns` (250 is a comfortable ceiling for a long-running agent) so the turn limit isn't the thing that stops useful work. Pair it with a high `api_max_retries` so transient upstream hiccups don't burn the run.

#### Dual-model: a main model for reasoning, an aux model for compression + sub-agents

A capable agent harness runs **two** models, both served by the same Turbohaul instance:

- **Main model** (`model.default`) — your highest-reasoning model, used for the primary think-and-act loop. A dense **27B-MTP** is a good main: strong reasoning with MTP speeding up decode (see [MODEL_SETTINGS.md](MODEL_SETTINGS.md) Example 1).
- **Auxiliary model** (`auxiliary.*.model`) — handles the harness's **context compression** (summarizing history when the window fills) **and sub-agent delegation** (spawned helper agents). An **MoE 35B-MTP at `parallel: 2`** is a good aux: the MoE is cheap to run, and `parallel: 2` lets several sub-agents share **one warm slot** concurrently instead of serializing or thrashing the model in and out (see [MODEL_SETTINGS.md](MODEL_SETTINGS.md) Example 2).

Because both models point at the same `:11401/v1` endpoint, Turbohaul's single-residency lifecycle still applies: the main model holds the warm slot while it's active; when the aux model is needed (compression or a sub-agent burst), Turbohaul swaps to it and its `parallel: 2` fan-out serves the sub-agents together. Keep the aux model's manifest at `parallel: 2` (+ `cont_batching` + `kv_unified`) so a sub-agent burst lands on one shared warm slot.

### 2. OpenAI Python SDK (raw)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<turbohaul-host>:11401/v1",
    api_key="dummy",
)

resp = client.chat.completions.create(
    model="qwen3.6-27b-dense",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,               # required for tool-calls on thinking models; recommended in general
    max_tokens=2048,
    extra_body={
        "thread_id": "session-abc-123",   # opt-in: enables ACTIVE_MATCH warm reuse
        "keep_alive": "5m",               # Ollama-compat; OpenAI SDK doesn't have this natively
    },
)
for chunk in resp:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

If you'll do multi-turn agent loops on the same conversation, **always pass `thread_id`**. Without it, every turn after the first will cold-load (each request gets a fresh slot). With it, turns N>1 hit the warm slot in ~1s.

### 3. langchain `ChatOpenAI`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://<turbohaul-host>:11401/v1",
    api_key="dummy",
    model="qwen3.6-27b-dense",
    streaming=True,
    max_tokens=2048,
    model_kwargs={
        "extra_body": {
            "thread_id": "session-abc-123",
            "keep_alive": "5m",
        }
    },
)
```

For tool-calling agents (langgraph, AgentExecutor), bind tools as usual:

```python
llm_with_tools = llm.bind_tools([my_tool_1, my_tool_2])
```

langchain will pass `tools` and `tool_choice` in the request body. Turbohaul forwards them to llama-server, which produces structured `tool_calls` chunks in the SSE stream. langchain's OpenAI tool-call parser handles those automatically.

### 4. llama-index `OpenAI`

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://<turbohaul-host>:11401/v1",
    api_key="dummy",
    model="qwen3.6-27b-dense",
    max_tokens=2048,
    additional_kwargs={
        "extra_body": {
            "thread_id": "rag-session-7",
            "stream": True,
        }
    },
)
```

### 5. LiteLLM (router / proxy)

If you have LiteLLM in front of multiple providers, register Turbohaul as a custom OpenAI-compat provider:

```yaml
# litellm_config.yaml
model_list:
  - model_name: qwen3.6-27b-turbohaul
    litellm_params:
      model: openai/qwen3.6-27b-dense
      api_base: http://<turbohaul-host>:11401/v1
      api_key: dummy
      stream: true
      extra_body:
        keep_alive: "10m"

  - model_name: gemma-4-26b-turbohaul
    litellm_params:
      model: openai/gemma-4-26b-a4b-moe
      api_base: http://<turbohaul-host>:11401/v1
      api_key: dummy
      stream: true
```

LiteLLM handles fallback / load-balance / retry logic. Turbohaul handles slot lifecycle.

### 6. Ollama-shape clients (Ollama Python, Ollama JS, OpenWebUI, etc.)

Use the **Ollama port** `11434`:

```python
import ollama

client = ollama.Client(host="http://<turbohaul-host>:11434")

resp = client.chat(
    model="qwen3.6-27b-dense",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    keep_alive="5m",
    options={"num_ctx": 8192},
)
```

The Ollama port speaks `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, etc. — drop-in compatible with the Ollama wire format. Internally Turbohaul converts to its single slot lifecycle.

### 7. Generic HTTP clients (curl, requests, fetch)

```bash
curl -sN http://<turbohaul-host>:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b-dense",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 256,
    "thread_id": "curl-test-1",
    "keep_alive": "1m"
  }'
```

`-N` is important — disables curl's default line-buffering so you see SSE chunks as they arrive.

---

## Multi-Tool-Call Agent Loops — What to know

This is the workflow Turbohaul was built for and stress-tested against.

1. **Each turn is a fresh HTTP request.** OpenAI Chat Completions is stateless — your agent re-sends the full message history each turn (including assistant tool-call messages and tool-result messages).
2. **Same-conversation continuity comes from `thread_id`.** If you include the same `thread_id` on every turn, Turbohaul's ACTIVE_MATCH cascade keeps the slot warm and sub-second.
3. **No special handling for thinking models.** Qwen3.6 and friends emit `<think>...</think>` in `delta.content` plus structured `delta.reasoning_content`. Your agent can either:
   - Display reasoning separately (read `reasoning_content`)
   - Strip thinking blocks (regex `<think>.*?</think>` on accumulated content)
   - Use raw content as-is
4. **Tool calls arrive as structured chunks.** When the model decides to call a tool, you'll get SSE chunks with `delta.tool_calls = [{index, id, function: {name, arguments}}]`. Arguments may stream incrementally — accumulate before parsing.
5. **`finish_reason: "tool_calls"` ends the turn.** Execute the tool, append `{"role": "tool", "tool_call_id": "...", "content": "..."}` to your messages, then POST another request. Don't forget the `thread_id`.

### Common pitfall — `max_tokens` too small

Thinking models exhaust `max_tokens` inside `<think>` if `max_tokens` is too small (default 512 in some clients is way too small). **Set `max_tokens` to at least 2000** for any thinking-model agent loop. Hermes uses 8192 as default for this reason.

### Common pitfall — Missing `thread_id`

If multi-turn loops feel slow (every turn cold-loads 30-60s), check that you're passing the same `thread_id` on each turn. Without it, ACTIVE_MATCH can't fire and each turn pays full cold-load cost.

### Common pitfall — Qwen3 text-JSON tool calls

Qwen3-family GGUFs on llama.cpp jinja templates sometimes emit a tool call as text JSON inside `message.content` (or wrapped in `<tool_call>...</tool_call>`) instead of populating `message.tool_calls`. Without intervention, OpenAI-shape clients that read only the structured field see "no tool call" and the loop stalls. Turbohaul recovers these automatically (post-processor runs AFTER `_merge_reasoning_into_content`, BEFORE returning to the client); your agent should see normal structured `tool_calls` and `finish_reason: "tool_calls"`. If you want the diagnostic detail (which candidates matched, which were rejected by the allowlist, etc.), set the `turbohaul.api.tool_call_recovery` logger to DEBUG. Full mechanism: [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md).

---

## Production Setup Notes

### Docker run with sane defaults (CUDA / NVIDIA host)

```bash
docker run -d --name turbohaul \
  --gpus all \
  -p 11401:11401 \
  -p 11434:11434 \
  -v $(pwd)/state:/var/lib/turbohaul \
  -v $(pwd)/models:/var/lib/turbohaul/import-staging \
  -e TURBOHAUL_IDLE_HOT_SECONDS=600 \
  -e TURBOHAUL_GRACE_SECONDS=30 \
  ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.5.0
```

Defaults are reasonable; you only need env-overrides if you want different timing.

### Manifest setup (per model)

Each model needs a manifest at `/var/lib/turbohaul/manifests/<model-tag>.yaml`. Example for Qwen3.6-27B:

```yaml
model_tag: qwen3.6-27b-dense
gguf_path: /var/lib/turbohaul/blobs/qwen3.6-27b-dense-q4_k_xl.gguf
quant: Q4_K_XL
context_length: 92160
gpu_layers: 999

llama_server_flags:
  ctx_size: 92160
  n_gpu_layers: 999
  cache_type_k: q4_0
  cache_type_v: q4_0
  n_predict: -1
  reasoning: auto
  reasoning_budget: 500          # caps thinking depth — tune 200-2000 for tool-loop speed
  jinja: true                    # REQUIRED for tool_calls + Qwen3 thinking-block preservation
```

The **`jinja: true`** flag is load-bearing for two things:
- `tool_calls` only work when llama-server uses the Jinja chat-template branch
- Qwen3-class thinking models only preserve `<think>` blocks in the response under `--jinja`

If you copy a manifest, keep this flag.

### Multi-model deployment

You can load multiple manifests; Turbohaul will swap models on demand (tears down the warm holder for one model when a request for a different model arrives). The invariant is **single-model-resident** — one model active at a time. Within the active model, **per-model concurrent serving** (fan-out up to the manifest's `parallel` limit) lets several callers or sub-agents share the one warm sidecar — this is what the main/aux dual-model pattern relies on. Cross-model multi-residency (two *different* models in VRAM simultaneously) is a Phase-1 roadmap item.

### Health checks

- `GET http://<host>:11401/health` → `{"status":"ok"}` (lightweight, no slot interaction)
- `GET http://<host>:11401/api/status` → full slot state, current model, queue depth, idle-hot timer
- `GET http://<host>:11401/api/config` → effective runtime config (idle_hot_seconds, grace, etc.)
- `GET http://<host>:11401/api/tags` → list of available models (Ollama-shape)
- `GET http://<host>:11401/v1/models` → list of available models (OpenAI-shape)

---

## Validation Smoke Tests (run after setup)

### Quick streaming smoke

```bash
curl -sN http://<host>:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-dense","messages":[{"role":"user","content":"hi"}],"stream":true,"max_tokens":32}' \
  | head -50
```

Expect: 12s keep-alive heartbeat then a stream of `data: {"choices":[{"delta":{"content":"..."}}]}` chunks ending in `data: [DONE]`.

### Multi-turn smoke (proves ACTIVE_MATCH works)

```python
import time, requests, json

URL = "http://<host>:11401/v1/chat/completions"
TID = "smoke-thread-" + str(int(time.time()))

def turn(content, history):
    history = history + [{"role": "user", "content": content}]
    body = {
        "model": "qwen3.6-27b-dense",
        "messages": history,
        "stream": True,
        "max_tokens": 64,
        "thread_id": TID,
    }
    start = time.time()
    with requests.post(URL, json=body, stream=True, timeout=180) as r:
        chunks = sum(1 for line in r.iter_lines() if line)
        wall = time.time() - start
    print(f"  wall={wall:.1f}s chunks={chunks}")
    return history + [{"role": "assistant", "content": "<reply>"}]

hist = []
print("Turn 1 (cold load expected ~30-60s on first call):")
hist = turn("What is 2+2?", hist)
print("Turn 2 (ACTIVE_MATCH — should be <5s):")
hist = turn("Add 5.", hist)
print("Turn 3 (ACTIVE_MATCH — should be <5s):")
hist = turn("Multiply by 3.", hist)
```

Turn 1: 30-60s wall (cold load). Turns 2+3: < 5s wall (warm reuse). If turns 2/3 take 30-60s, your `thread_id` isn't being forwarded — check `extra_body`.

### Tool-call smoke

```bash
curl -sN http://<host>:11401/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b-dense",
    "messages": [{"role":"user","content":"What is the weather in Boston?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
      }
    }],
    "tool_choice": "auto",
    "stream": true,
    "max_tokens": 256
  }'
```

Expect: at least one chunk with `delta.tool_calls = [{...}]` containing `function.name: "get_weather"` and `function.arguments` accumulating JSON. `finish_reason: "tool_calls"` ends the stream. (Model must have `jinja: true` in manifest for this to work.)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Every turn takes 30-60s wall | `thread_id` not being forwarded | Use `extra_body: {"thread_id": "..."}` in OpenAI SDK; same in langchain `model_kwargs`. |
| "Slot did not reach ACTIVE within 600s" | (Should not happen in current builds) | Upgrade to the latest release. This was an early bug in the ACTIVE_MATCH streaming path. |
| HTTP 503 with `Retry-After` | Safety guardrail refused spawn (host OOM / IO-stuck) | Wait and retry; the `Retry-After` header tells you how long. |
| HTTP 502 with `upstream_status` | llama-server returned an error (context overflow, malformed payload) | Check `upstream_body` in the response; usually means your prompt exceeded the model's `ctx_size`. |
| HTTP 504 | llama-server timed out generating | Reduce `max_tokens` or check that the model isn't stuck in a thinking loop (lower `reasoning_budget`). |
| Tool calls never fire (model just describes the tool instead) | Manifest missing `jinja: true` | Add `jinja: true` to the model's manifest under `llama_server_flags`. Restart container. |
| Tool calls fire on Ollama `/api/chat` but not OpenAI `/v1/chat/completions` | An older build silently dropped `tools` on the OpenAI endpoint's `client_meta` | Upgrade to ≥ v0.2.3. |
| Tool calls show as text JSON in `message.content`, `message.tool_calls` empty | Either an older build (no recovery layer) OR request did not advertise `tools` (recovery requires the allowlist) | Upgrade to ≥ v0.2.3 AND ensure your request body includes `tools: [...]`. See [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md). |
| Streaming hangs at start, no chunks | Client doesn't accept SSE Content-Type | Set `Accept: text/event-stream` header, or use a real SSE library (sseclient-py, eventsource). |
| Hermes pane stuck "pondering..." | Inter-turn slot didn't promote via ACTIVE_MATCH | Make sure your model's manifest has `reasoning_budget` capped (recommend 500 for tool-loops). |
| `tools` field rejected with HTTP 400 | Old Turbohaul version | Upgrade to the latest release. |

If the issue isn't here, check `docker logs turbohaul` for the wrapper-side view and `/api/status` for the current slot state.

---

## What Turbohaul does NOT do (yet)

To save you from chasing things that aren't supported:

- **Cross-model multi-residency.** Single-model-resident is the current invariant (one model in VRAM at a time, swapped on demand). Per-model concurrent serving (fan-out *within* the active model) is supported; two *different* models co-resident is a Phase-1 roadmap item.
- **`/v1/completions` (FIM)** and **vision content-parts (`image_url`)** — not yet supported. (`/v1/embeddings` and `response_format` — both `json_object` and `json_schema` — *are* supported.)
- **Cross-process queue sharing.** If you spin up multiple Turbohaul instances, each has its own queue. Use a single instance behind a load balancer if you need request-level isolation.
- **Authentication.** the v1 internal port has no auth (`dummy` api_key works). Production-external deployment requires bearer-auth + TLS termination in front (v2 roadmap).
- **GPU mid-flight cancel.** Once a request is mid-generation, canceling the client connection signals server-side disconnect but the in-flight generation completes before the slot frees.

---

## Where to go next

- **Architecture deep-dive:** [ARCHITECTURE.md](ARCHITECTURE.md) (components / slot FSM / IDLE_HOT / ACTIVE_MATCH / telemetry internals)
- **Model settings & MTP examples:** [MODEL_SETTINGS.md](MODEL_SETTINGS.md) (27B-MTP + 35B-MoE-MTP worked manifests)
- **KV-cache prefix reuse (fast multi-turn):** [PREFIX_CACHE_REUSE.md](PREFIX_CACHE_REUSE.md) (only prefill the new tokens each turn — ~30× on long contexts)
- **Tool-call recovery layer:** [TOOL_CALL_HANDLING.md](TOOL_CALL_HANDLING.md) (wire shape, recovery post-processor, Qwen3 text-JSON case, testing)
- **Version history:** [CHANGELOG.md](../CHANGELOG.md)
- **GitHub:** `https://github.com/MrTrenchTrucker/turbohaul-manager` (public repository)

---

*This doc is the contract between Turbohaul and the agents that use it. If you find a wire-shape detail that's not documented here, that's a doc bug — open an issue.*
