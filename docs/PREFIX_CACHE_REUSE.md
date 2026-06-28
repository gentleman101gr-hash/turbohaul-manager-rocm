# Turbohaul-Manager — KV-Cache Prefix Reuse (fast multi-turn agents)

**Why this matters for local inference:** it is the difference between a multi-turn agent that re-prefills its *entire* growing context every single turn (seconds to minutes per turn, getting worse as the conversation grows) and one that prefills only the **new** tokens each turn (sub-second follow-ups). On a long agent loop the speedup is dramatic — a measured **~28.7s → ~0.9s** per follow-up (roughly **30×**) on a ~125K-token context, and the win grows with context length.

If you run any agent that re-sends its conversation history each turn (every OpenAI-Chat-Completions agent does — the API is stateless), this is the single most impactful thing to get right.

---

## The problem: the re-prefill "crawl"

OpenAI Chat Completions is stateless. Every turn, your agent re-sends the **full** message history (system prompt + every prior user/assistant/tool message) plus the new input. Naively, the server must **prefill** that entire prompt — run it through the model to rebuild the KV cache — before it can generate the first new token.

For a 125K-token agent context that prefill can take tens of seconds **on every turn**, and it gets slower as the conversation grows. The agent appears to "crawl": each turn is dominated by re-reading context it already processed last turn.

---

## The fix: reuse the cached prefix on a warm slot

The model's KV cache for turn N's prompt is *almost* the same as turn N+1's prompt — they share a long common prefix (everything except the new tokens). If the KV cache is **kept resident on the GPU** and the next request lands on that **same warm slot**, the server can detect the shared prefix, **reuse** that part of the cache, and prefill only the delta. The result is a near-instant follow-up.

Two things have to be true for this to fire:

1. **The KV cache must stay on the warm slot** (not be torn down or shifted out) — governed by the manifest flags below.
2. **The follow-up must land on the same warm slot** — governed by a stable `thread_id` (see [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md)). If each turn looks like a brand-new identity, it gets a fresh slot with a cold cache and you pay the full prefill again.

Both are required. Prefix-reuse flags with no warm-slot identity = the cache is there but the request never finds it. A warm slot with prefix-reuse off = the request finds the slot but re-prefills anyway.

---

## How to enable it

### 1. Manifest flags (keep the cache and reuse the prefix)

In the model's manifest under `llama_server_flags`:

```yaml
llama_server_flags:
  cache_reuse: 256              # reuse a cached prefix when >= 256 tokens match (prefix-cache reuse)
  slot_prompt_similarity: 0.5   # how similar a prompt must be to reuse a slot's cache (0.0-1.0)
  no_context_shift: true        # never silently shift/drop context out of the KV window
  flash_attn: true              # efficient attention; pairs well with large resident caches
  # ... your other flags (ctx_size, n_gpu_layers, cache_type_k/v, etc.)
```

- **`cache_reuse`** is the core knob: it tells `llama-server` to reuse a cached KV prefix when at least this many leading tokens match what's already cached, prefilling only the rest. `256` is a sensible floor.
- **`slot_prompt_similarity`** controls how aggressively a new prompt is matched to an existing slot's cache (higher = stricter match required).
- **`no_context_shift`** keeps the cache stable — without it, a full window can silently shift older tokens out, which breaks prefix matching (and can corrupt long-context correctness). Keep it on for agent workloads.

These are the same flags you'll see in the worked examples in [MODEL_SETTINGS.md](MODEL_SETTINGS.md).

### 2. Stable `thread_id` (land follow-ups on the warm slot)

Send the **same `thread_id`** on every turn of a conversation so Turbohaul routes the follow-up to the warm slot that already holds the cache. For OpenAI-SDK clients this goes in `extra_body`:

```python
resp = client.chat.completions.create(
    model="dense-27b-mtp",
    messages=full_history,            # the whole conversation, re-sent each turn
    stream=True,
    extra_body={"thread_id": "session-abc-123"},   # SAME id every turn -> warm slot -> prefix reuse
)
```

Within the grace/idle-hot window, same-`thread_id` follow-ups hit the warm slot (Turbohaul's ACTIVE_MATCH path), the prefix-reuse flags fire, and only the new tokens prefill. See [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) for the per-client recipes.

---

## What it looks like when it's working

- **Turn 1:** cold (or warm-model) prefill of the full prompt — the normal cost.
- **Turns 2+ (same `thread_id`):** the follow-up reuses the cached prefix and prefills only the new tokens — **sub-second** even on a six-figure-token context, instead of re-reading the whole history.

If your turns 2+ are still slow (re-prefilling), check, in order:
1. Are you sending the **same `thread_id`** every turn? (Most common miss — without it every turn is a fresh slot.)
2. Does the manifest have **`cache_reuse`** + **`no_context_shift`**?
3. Are the follow-ups landing inside the grace / idle-hot window (i.e. not so far apart the slot went cold)?

---

## Caveat: sub-agents with *different* contexts

Prefix reuse helps when consecutive requests **share a prefix**. If several requests share an identity but carry **different** contexts (e.g. parallel sub-agents each with their own conversation), they do **not** share a prefix — each correctly gets its own cache, and Turbohaul keeps them isolated (no cross-contamination). That's expected: prefix reuse accelerates *one* conversation growing over many turns, not unrelated conversations. For concurrent same-model sub-agents, see the `parallel:` fan-out in [MODEL_SETTINGS.md](MODEL_SETTINGS.md).

---

## See also

- [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) — `thread_id` wiring per client + the anti-flap timeouts
- [MODEL_SETTINGS.md](MODEL_SETTINGS.md) — where `cache_reuse` / `slot_prompt_similarity` / `no_context_shift` live in a full manifest
- [ARCHITECTURE.md](ARCHITECTURE.md) — slot lifecycle, grace / idle-hot, ACTIVE_MATCH warm reuse
