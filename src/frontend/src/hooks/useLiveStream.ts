import { useEffect, useRef, useState, useCallback } from 'react';
import type { LiveOutputFrame } from '../api';

// P2: SSE hook for /ui/live/output/stream
// Demultiplexes concurrent generations into ONE pane PER LOADED MODEL
// (keyed by model_tag, falling back to generation_id when the backend
// omits the tag). A new turn (new generation_id) RESETS that model's box
// text in place rather than spawning a fresh box per tool-call.

const TOK_HISTORY_CAP = 40;

// BUG 1 grace-hold: an `idle` frame arrives in the brief gaps between a model's
// rapid follow-up / sub-agent calls (observed ~5-6s). If we flip every pane to
// done:true on the very first idle frame, the LIVE/DONE badge oscillates
// grey<->green every gap. Instead, an idle frame ARMS a single timer; only if
// the idle persists past RECENT_HOLD_MS (no intervening reset/delta) do we flip
// all panes to done:true. Any non-idle frame cancels the pending flip, so brief
// gaps stay LIVE and only a sustained (>10s) stop reads DONE.
const RECENT_HOLD_MS = 10000;

export interface GenPane {
  paneKey: string;
  generation_id: string;
  model_tag: string | null;
  text: string;
  done: boolean;
  lastTokS: number | null;
  lastFrameAt: number;
  tokHistory: number[];
}

export interface UseLiveStreamResult {
  panes: Record<string, GenPane>;
  connected: boolean;
  error: string | null;
}

const SSE_URL = '/ui/live/output/stream';
const RECONNECT_MS = 2000;

export function useLiveStream(): UseLiveStreamResult {
  const [panes, setPanes] = useState<Record<string, GenPane>>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // BUG 1: pending idle->done flip timer. A single timer shared by all panes;
  // re-armed on each idle frame, cleared on any non-idle frame, on reconnect,
  // and on unmount.
  const idleTimer = useRef<number | undefined>(undefined);

  const clearIdleTimer = useCallback(() => {
    if (idleTimer.current !== undefined) {
      window.clearTimeout(idleTimer.current);
      idleTimer.current = undefined;
    }
  }, []);

  const connect = useCallback(async () => {
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    // Fresh connection: drop any stale pending idle flip from a prior socket.
    clearIdleTimer();

    // Reconnect loop: a closed/dropped stream (e.g. the turbohaul container
    // restarting) is retried with backoff until this hook unmounts (abort).
    while (!ctrl.signal.aborted) {
      try {
        const resp = await fetch(SSE_URL, {
          headers: { Accept: 'text/event-stream' },
          signal: ctrl.signal,
        });

        if (!resp.ok) {
          setError(`SSE ${resp.status}`);
          setConnected(false);
        } else {
          const reader = resp.body?.getReader();
          if (!reader) {
            setError('No reader');
            setConnected(false);
          } else {
            setConnected(true);
            setError(null);

            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              // Keep the last incomplete line in buffer
              buffer = lines.pop() || '';

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith(':')) continue; // skip blank/comments

                // SSE data lines start with "data: "
                if (!trimmed.startsWith('data: ')) continue;

                const raw = trimmed.slice(6);
                try {
                  const frame = JSON.parse(raw) as LiveOutputFrame;
                  // Idle: backend reports no active generation. Do NOT clear the
                  // boxes (that made them BLINK in/out during the brief gaps
                  // between a model's rapid follow-up calls).
                  //
                  // BUG 1 FIX — grace-hold on the idle->done transition: instead
                  // of flipping every pane done:true immediately (which made the
                  // badge oscillate DONE<->LIVE across inter-burst gaps), ARM a
                  // single timer. Only a SUSTAINED idle (no reset/delta for
                  // RECENT_HOLD_MS) actually flips the panes to done. The timer
                  // is replaced on each idle frame and cancelled by any non-idle
                  // frame below, so brief gaps keep the boxes LIVE.
                  if (frame.idle) {
                    clearIdleTimer();
                    idleTimer.current = window.setTimeout(() => {
                      idleTimer.current = undefined;
                      setPanes(prev =>
                        Object.fromEntries(
                          Object.entries(prev).map(([k, v]) => [k, { ...v, done: true }]),
                        ),
                      );
                    }, RECENT_HOLD_MS);
                    continue;
                  }
                  const genId = frame.generation_id;
                  if (!genId) continue;

                  // A real (non-idle) frame arrived: cancel any pending idle
                  // flip so a brief gap never reads as DONE.
                  clearIdleTimer();

                  // KEY by model_tag (one persistent box per loaded model).
                  // When the backend omits the tag (cap<=1 single-slot: the
                  // poller writes only the live_generation alias so the
                  // reverse-lookup returns null), fall back to keying by
                  // generation_id AND preserve the original single-box
                  // behavior: a new generation REPLACES the prior box instead
                  // of stacking a dead box per follow-up tool-call.
                  const modelTag = frame.model_tag ?? null;
                  const paneKey = modelTag ?? genId;

                  setPanes(prev => {
                    const existing = prev[paneKey];
                    // A reset frame = a NEW generation.
                    if (frame.reset) {
                      const next: Record<string, GenPane> =
                        modelTag == null
                          ? // null-tag (cap<=1): drop other untagged boxes so a
                            // new generation does not accumulate dead panes;
                            // keep any tagged (cap>=2) boxes untouched.
                            Object.fromEntries(
                              Object.entries(prev).filter(
                                ([, v]) => v.model_tag != null,
                              ),
                            )
                          : { ...prev };
                      next[paneKey] = {
                        paneKey,
                        generation_id: genId,
                        model_tag: modelTag,
                        text: frame.text,
                        done: frame.done ?? false,
                        lastTokS: frame.tok_s ?? null,
                        lastFrameAt: Date.now(),
                        tokHistory: frame.tok_s != null ? [frame.tok_s] : [],
                      };
                      return next;
                    }
                    // A delta: append text + push tok_s to this model's chart.
                    const prevHistory = existing?.tokHistory ?? [];
                    const tokHistory =
                      frame.tok_s != null
                        ? [...prevHistory, frame.tok_s].slice(-TOK_HISTORY_CAP)
                        : prevHistory;
                    return {
                      ...prev,
                      [paneKey]: {
                        paneKey,
                        generation_id: genId,
                        model_tag: modelTag ?? existing?.model_tag ?? null,
                        text: (existing?.text ?? '') + frame.text,
                        done: frame.done ?? existing?.done ?? false,
                        lastTokS: frame.tok_s ?? existing?.lastTokS ?? null,
                        lastFrameAt: Date.now(),
                        tokHistory,
                      },
                    };
                  });
                } catch {
                  // skip malformed frames
                }
              }
            }
            // Stream ended (server closed it) — fall through to reconnect.
            setConnected(false);
          }
        }
      } catch (e: unknown) {
        if (e instanceof DOMException && e.name === 'AbortError') return;
        setError(e instanceof Error ? e.message : String(e));
        setConnected(false);
      }

      // Backoff before reconnecting, unless this hook has been unmounted.
      if (ctrl.signal.aborted) return;
      await new Promise(r => setTimeout(r, RECONNECT_MS));
    }
  }, [clearIdleTimer]);

  useEffect(() => {
    void connect();
    return () => {
      abortRef.current?.abort();
      clearIdleTimer();
    };
  }, [connect, clearIdleTimer]);

  return { panes, connected, error };
}
