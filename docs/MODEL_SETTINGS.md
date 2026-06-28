# Turbohaul-Manager — Model Settings & Example Manifests

**Audience:** Operators writing per-model manifests, especially for **MTP** (Multi-Token-Prediction / speculative-decode) GGUFs and large-context configs.

Each model loaded into Turbohaul has a YAML manifest at `/var/lib/turbohaul/manifests/<model-tag>.yaml`. The manifest is the single source of truth for how the model's `llama-server` sidecar is launched: its flags, context size, and KV-cache layout. Only flags on the closed `SAFE_LLAMA_FLAGS` allowlist are accepted (a manifest cannot smuggle in an arbitrary flag).

This doc gives **two worked examples** — a dense 27B model and a Mixture-of-Experts 35B model, both **MTP-enabled** — that you can copy and adapt. For the full flag doctrine see [TURBOQUANT_FLAGS.md](TURBOQUANT_FLAGS.md); for the KV-offload trade-off see [KV_CACHE_OFFLOADING.md](KV_CACHE_OFFLOADING.md).

---

## What MTP buys you

MTP (Multi-Token Prediction) is speculative decoding driven by a draft head baked into the GGUF (e.g. a `blk.N.nextn` tensor). With `spec_type: draft-mtp`, `llama-server` proposes several tokens per step and verifies them in one pass, so decode is faster on the dense path with no separate draft model to load. Two flags govern it:

- **`spec_type: draft-mtp`** — turn on MTP speculative decoding using the model's own draft head.
- **`spec_draft_n_max: N`** — how many tokens the draft head may propose per step. 2–3 is a good range; higher helps only if the draft head's acceptance rate stays high.

> Verify MTP is actually active on a loaded model via the sidecar's `/props` (`spec.types`) and `/slots` (draft-accept rate) — a flag being *present* in `--help` is not proof it's *running*.

---

## Example 1 — Dense 27B, MTP, single warm slot

A dense ~27B model serving a long context on a 24 GB-class GPU. Weights on GPU, KV cache quantized in VRAM (`turbo2`), MTP on. `split_mode: layer` spreads layers across two cards when present (drop it for a single card and lower `ctx_size` to fit).

```yaml
model_tag: dense-27b-mtp
display_name: 'Dense 27B (MTP)'
description: 'Dense 27B, MTP-enabled, large-context VRAM-KV serving config.'
gguf_blob_sha256: '<sha256 of your GGUF blob>'
context_size: 250000
expected_vram_bytes: 24000000000        # ~24 GB; the safety gate checks free VRAM against this
revision: 1
llama_server_flags:
  ctx_size: 250000
  n_gpu_layers: 999                      # offload all layers to GPU
  split_mode: layer                      # spread layers across cards (omit for single-GPU)
  main_gpu: 0
  cache_type_k: turbo2                   # quantized KV in VRAM (K)
  cache_type_v: turbo2                   # quantized KV in VRAM (V)
  flash_attn: true
  no_context_shift: true                 # never silently drop context; error instead
  cache_reuse: 256                       # reuse prefix KV across same-thread follow-ups
  slot_prompt_similarity: 0.5
  reasoning: auto
  reasoning_format: auto
  reasoning_budget: 9192                 # room for chain-of-thought on a thinking model
  jinja: true                            # REQUIRED for tool_calls + thinking-block preservation
  n_predict: -1
  no_perf: true
  spec_type: draft-mtp                   # MTP speculative decode
  spec_draft_n_max: 2
prompt_template:
  system_default: ''
  stop_tokens: []
```

**Why these values**
- `cache_type_k/v: turbo2` keeps the KV cache small enough that 250K context fits in VRAM alongside the weights on a 24 GB card. If it doesn't fit, either lower `ctx_size` or move the KV to host RAM (see Example 2's offload note).
- `reasoning_budget: 9192` gives a thinking model enough room that it doesn't exhaust its budget mid-`<think>`. Lower it (e.g. 500–2000) for faster, shallower tool-loops.
- `jinja: true` is load-bearing — `tool_calls` and `<think>`-block preservation only work on the Jinja template branch.

---

## Example 2 — MoE 35B, MTP, parallel fan-out

A Mixture-of-Experts ~35B (small active-param) model configured to serve **multiple requests concurrently** on one warm sidecar. `parallel: 2` + `cont_batching` + `kv_unified` lets two riders share the slot; the total context is split evenly across the slots (`500000` total = `250000` per slot). `turbo3` keeps the larger MoE KV footprint in check.

```yaml
model_tag: moe-35b-mtp
display_name: 'MoE 35B-A3B (MTP, parallel:2)'
description: 'MoE 35B (small active params), MTP-enabled, 2-way parallel serving config.'
gguf_blob_sha256: '<sha256 of your GGUF blob>'
context_size: 500000                     # total across slots; 250000 per slot at parallel:2
expected_vram_bytes: 24000000000
revision: 1
llama_server_flags:
  parallel: 2                            # two concurrent rider slots on one sidecar
  cont_batching: true                    # continuous batching across the slots
  kv_unified: true                       # slots share one unified KV cache
  ctx_size: 500000                       # divided evenly: 250000 per slot
  n_gpu_layers: 999
  split_mode: layer
  main_gpu: 0
  cache_type_k: turbo3                   # heavier KV quant for the larger MoE cache
  cache_type_v: turbo3
  flash_attn: true
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  reasoning: auto
  reasoning_format: deepseek-legacy
  reasoning_budget: 8192
  jinja: true
  n_predict: -1
  no_perf: true
  spec_type: draft-mtp
  spec_draft_n_max: 3
prompt_template:
  system_default: ''
  stop_tokens: []
```

**Why these values**
- `parallel: 2` is what lets one warm model serve two callers (or one caller's two sub-agents) at once. With `kv_unified: true`, the manifest validator cross-checks that `ctx_size` divides evenly across the slots — each slot's window (`ctx_size / parallel`) must clear the minimum.
- `cache_type_k/v: turbo3` trades a little quality for a smaller KV footprint, which matters more on a MoE model whose cache is larger per token.
- If even quantized KV won't fit at this context, add **`no_kv_offload: true`** (+ a `cache_ram` budget) to move the KV cache to host RAM while the weights stay on GPU — see [KV_CACHE_OFFLOADING.md](KV_CACHE_OFFLOADING.md). It enables very large contexts on a small card at a decode-speed cost.

---

## Picking settings

| Goal | Start from | Key knobs |
|---|---|---|
| Fastest decode, fits in VRAM | Example 1 | `spec_type: draft-mtp`, `cache_type_*: turbo2`, single warm slot |
| Serve 2+ callers / sub-agents at once | Example 2 | `parallel: 2`, `cont_batching`, `kv_unified` |
| Huge context on a small GPU | Example 2 + offload | `no_kv_offload: true`, `cache_ram`, `cache_type_*: turbo3` |
| Single GPU | either, minus `split_mode` | drop `split_mode`/`main_gpu`, lower `ctx_size` to fit one card |

All flags here are on the `SAFE_LLAMA_FLAGS` allowlist. Adding a flag the allowlist doesn't know requires a code change + review — the YAML alone cannot enable it.

---

## See also

- [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md) — wiring an agent to these models (incl. the main + aux dual-model pattern)
- [PREFIX_CACHE_REUSE.md](PREFIX_CACHE_REUSE.md) — why `cache_reuse` / `slot_prompt_similarity` / `no_context_shift` give multi-turn agents a huge prefill speedup
- [TURBOQUANT_FLAGS.md](TURBOQUANT_FLAGS.md) — full flag doctrine for production manifests
- [KV_CACHE_OFFLOADING.md](KV_CACHE_OFFLOADING.md) — KV-cache offload to host RAM
- [ARCHITECTURE.md](ARCHITECTURE.md) — manifest system, safety gates, slot lifecycle
