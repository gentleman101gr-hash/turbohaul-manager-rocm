"""TurbohaulManager: top-level orchestrator wiring queue + subprocess + state + timers.

Per v0.2 ARCHITECTURE.md - orchestrates the whole lifecycle described in §6 state
machine. The foundational interface ships first; the full worker_loop
streaming implementation lands alongside the API layer that forwards
to llama-server.
"""
import asyncio
import enum
import contextlib
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from turbohaul.config import KEEP_ALIVE_MAX_S, BootConfig, RuntimeConfig
from turbohaul.fsm import LEGAL_TRANSITIONS, InvalidTransition, is_terminal, transition
from turbohaul.live_monitor import LiveOutputBuffer, idle_generation
from turbohaul.manifest import flags_to_argv, read_manifest
from turbohaul.queue import GraceTimer, IdleHotTimer, TurbohaulQueue
from turbohaul.safety import (
    all_safety_gates,
    estimate_kv_cache_mib,
    PER_SLOT_COMPUTE_FLOOR_MIB,
    _vram_budget,
)
from turbohaul.singleton import (
    boot_orphan_reaper,
    detect_foreign_gpu_apps,
    intra_lifetime_orphan_scan,
)
from turbohaul.slot import Slot, SlotEvictedError, SlotState, derive_thread_id_prefix_hash
from turbohaul.state import (
    audit_db_session,
    known_active_pids,
    mark_slot_ended,
    open_state_db,
    reconcile_orphaned_slots,
    record_audit_event,
    state_db_session,
    upsert_slot,
)
from turbohaul.subprocess_mgr import (
    SidecarHandle,
    drained_sigterm,
    open_and_verify_binary,
    spawn_sidecar,
    verify_binary_sha256,
    verify_vram_cleared,
    wait_until_healthy,
)
from turbohaul.telemetry import FlapTelemetry, init_telemetry


log = logging.getLogger(__name__)


class EventBus:
    """Pub-sub for state-level events broadcast to /ws/state subscribers.

    Per v0.2 §11.1 redaction policy: callers are responsible for emitting only
    safe events. This bus enforces a denylist (prompt/response/stderr/context)
    on publish as defense-in-depth — even if a caller accidentally includes one
    of those keys, it gets stripped before fan-out.
    """

    REDACTED_KEYS: frozenset[str] = frozenset({
        "prompt",
        "response",
        "context",
        "stderr",
        "stdout",
        "messages",
    })

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.add(q)

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish_nowait(self, event: dict) -> None:
        """Publish an event. Sensitive keys are stripped (denylist).

        Each subscriber gets a copy. Full subscriber queues drop on back-pressure
        rather than block the publisher (worker_loop must stay responsive).
        """
        safe_event = {k: v for k, v in event.items() if k not in self.REDACTED_KEYS}
        for q in list(self._subscribers):
            try:
                q.put_nowait(safe_event)
            except asyncio.QueueFull:
                log.warning("event_bus subscriber queue full — dropping event")


def _pid_is_alive(pid: int, kill_fn: Callable[[int, int], None] | None = None) -> bool:
    """Defensive check: is pid currently alive on this host?"""
    fn = kill_fn or os.kill
    try:
        fn(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours - conservatively treat as alive


# Multi-slot refactor PHASE-0 (resident-registry scaffold). The manager today
# runs a single-sidecar invariant (one model loaded at a time); the eventual
# multi-slot work needs a per-model resident registry so >1 sidecar can be
# tracked concurrently. This constant PINS the registry to exactly ONE resident
# for Phase-0 so runtime behaviour is byte-for-byte identical to the deployed
# v0.3.8 single-sidecar manager. Phase-1 (dispatcher / driver-tasks / LRU-evict
# / multi-spawn — explicitly OUT of Phase-0 scope) is the place that raises this
# and migrates the live FSM state onto the registry behind its own concurrency
# tests. Do NOT bump this in Phase-0.
MAX_PARALLEL_SIDECARS = 1

# max times the dispatcher re-queues an unroutable slot (all
# residents busy, no idle victim to evict) before failing its future (a 503-
# equivalent) so an unroutable slot can't busy-loop the dispatcher / starve.
_MAX_DISPATCH_DEFERS = 50

# Bounded poll interval for the cap<=1 fan-out drain's
# continuous rider-admit. The drain wakes on EITHER a slot completion OR this
# timeout, so a same-model rider that arrives mid-burst (after the one-shot
# admit already ran, while a --parallel slot is still free) joins within this
# bound instead of waiting out the anchor. Small relative to any real request
# (sub-agents run for seconds); the wake only does a cheap non-blocking
# pop_next while a fan-out is already active.
_FANOUT_ADMIT_POLL_S = 0.1

# The single registry key under which the lone Phase-0 resident lives. Phase-1
# keys residents by ``model_tag``; in Phase-0 the registry holds exactly one
# entry under this sentinel so ``_residents`` is non-empty and shape-correct
# without claiming a model binding the FSM hasn't actually made yet.
_SINGLETON_RESIDENT_KEY = "__phase0_singleton__"


class ResidentState(enum.StrEnum):
    """Resident lifecycle. RESERVED_LOADING: budget claimed under
    _registry_lock, sidecar spawning. ACTIVE: serving. GRACE: warm grace window.
    IDLE_EVICTABLE: warm-idle + LRU-evictable. DEAD: driver died/evicted, pending
    deregister. The dispatcher is the SOLE writer of the _residents dict + each
    resident's ``state`` (under _registry_lock); the per-resident driver owns r.*."""

    RESERVED_LOADING = "RESERVED_LOADING"
    ACTIVE = "ACTIVE"
    GRACE = "GRACE"
    IDLE_EVICTABLE = "IDLE_EVICTABLE"
    DEAD = "DEAD"


@dataclass
class Resident:
    """A single loaded-model sidecar tracked by the manager (PHASE-0 scaffold).

    This is the data structure the multi-slot refactor will key by ``model_tag``
    in ``TurbohaulManager._residents``. In PHASE-0 the registry is pinned to one
    resident (``MAX_PARALLEL_SIDECARS == 1``) and is NOT yet the authoritative
    source for the live FSM scalars — the manager continues to drive the
    deployed single-sidecar attributes (``_active_handle`` / ``_active_slot`` /
    ``_spawn_seq`` / the ``_idle_*`` holder) verbatim. The fields below mirror
    the exact shape Phase-1 will adopt; only ``inflight`` is wired to the live
    list (shared by reference with ``_inflight`` so in-place ``append``/``remove``
    on the manager's fan-out rider set stays visible through the resident view).
    The scalar fields stay at their construction placeholders in Phase-0 and
    become authoritative only when Phase-1 routes the FSM through the registry.

    Fields:
      model_tag:       the model this resident serves (None until Phase-1 binds).
      handle:          the live ``SidecarHandle`` (Phase-1 authoritative).
      port:            the sidecar's listen port (Phase-1 authoritative).
      inflight:        the concurrent fan-out rider Slots (anchor at index 0).
      spawn_seq:       monotonic spawn counter for fixed-port swap detection.
      idle_expires_at: monotonic deadline of the warm-idle hold, if any.
      active_slot:     the anchor Slot currently driven on this resident.
    """

    model_tag: str | None = None
    handle: "SidecarHandle | None" = None
    port: int | None = None
    inflight: list[Slot] = field(default_factory=list)
    spawn_seq: int = 0
    idle_expires_at: float | None = None
    active_slot: "Slot | None" = None
    # state-migration: per-resident mirrors of the manager-global
    # idle holder (``_idle_handle`` / ``_idle_model_tag``) + the latest keep_alive
    # intent + this resident's own grace/idle timers. At ``MAX_PARALLEL_SIDECARS
    # == 1`` they mirror the singleton manager scalars 1:1 (byte-identical); a later phase's
    # dispatcher makes them authoritative per concurrent resident. ``grace``/
    # ``idle`` are constructed + wired in ``TurbohaulManager.__init__`` (the
    # GraceTimer/IdleHotTimer classes live in queue.py, imported — queue.py
    # untouched per the RC guardrail).
    idle_handle: "SidecarHandle | None" = None
    idle_model_tag: str | None = None
    latest_keep_alive_s: int | None = None
    grace: "GraceTimer | None" = None
    idle: "IdleHotTimer | None" = None
    # dispatcher/concurrency fields. Populated under _registry_lock
    # at reservation (state / reserved_need_mib / parallel / main_gpu / port from the
    # in-lock manifest read) and on spawn (booting_pid then handle). ``inbox`` is the
    # dispatcher->driver slot hand-off queue; ``driver_task`` is this resident's
    # _drive_resident task; ``torn_down`` is the lock-guarded exactly-once teardown
    # claim shared by the driver finally + the death-supervisor reaper.
    state: ResidentState = ResidentState.ACTIVE
    reserved_need_mib: int = 0
    booting_pid: int | None = None
    parallel: int = 1
    main_gpu: int = 0
    # the manifest split_mode this resident loaded under. Co-residence
    # is supported ONLY for single-GPU-pinned ('none') models on DISTINCT cards; a
    # layer/row/tensor-split sibling spans all cards (no free distinct card to
    # guarantee) so the cross-resident gate refuse-blinds against it (interim until
    # per-card layer-split accounting lands). Default 'layer' matches the footprint
    # degrade-open default.
    split_mode: str = "layer"
    # Per-model sleep_idle_seconds from the manifest. -1 = pin/keep-warm (never
    # idle-unload this model), 0 = unload immediately after request, N>0 = idle
    # timeout in seconds. Default 0 means "use global default" — the driver will
    # fall back to runtime.queue.idle_hot_load_seconds.
    sleep_idle_seconds: int = 0
    last_active_monotonic: float = 0.0
    torn_down: bool = False
    driver_task: "asyncio.Task | None" = None
    inbox: "asyncio.Queue | None" = None


class TurbohaulManager:
    """Top-level orchestrator.

    Responsibilities:
    - Boot reconcile: orphan reap + foreign-GPU detect + state.sqlite slot cleanup
    - Verify binary sha256 pin at boot (v0.2 §7.1)
    - Accept fresh requests via submit() → push to queue (head if grace match)
    - Expose status_snapshot() for /status endpoint
    - Drive the FSM via worker_loop (skeleton first; full streaming follows)
    - Clean shutdown
    """

    def __init__(
        self,
        boot: BootConfig,
        runtime: RuntimeConfig,
        *,
        spawn_fn: Callable | None = None,
        health_fn: Callable | None = None,
        sigterm_fn: Callable | None = None,
        vram_fn: Callable | None = None,
        complete_fn: Callable | None = None,
    ) -> None:
        self.boot = boot
        self.runtime = runtime
        self.queue = TurbohaulQueue(
            staging_max=runtime.queue.staging_queue_depth,
            acceptance_max=runtime.queue.acceptance_buffer_max,
            max_consecutive_same_model=runtime.queue.max_consecutive_same_model,
            max_other_model_wait_s=runtime.queue.max_other_model_wait_s,
        )
        self.grace = GraceTimer(
            grace_seconds=runtime.queue.grace_seconds,
            max_extensions=runtime.queue.max_grace_extensions,
        )
        self.idle = IdleHotTimer(idle_seconds=runtime.queue.idle_hot_load_seconds)
        self._active_handle: SidecarHandle | None = None
        self._active_slot: Slot | None = None
        # Per-model concurrent dispatch (Design #1): the rider set for the
        # CURRENT anchor cycle when the active model's manifest declares
        # parallel>1. Element 0 is the anchor. Mutated ONLY by worker_loop
        # (append at fan-out admit, remove at per-rider drain, cleared after the
        # drain barrier); read await-free by status_snapshot. Stays [] for
        # parallel:1 models, so the single-mutator discipline above (no lock) is
        # preserved verbatim — worker_loop remains the sole writer and routes
        # only ever set their OWN slot.stream_done_event.
        self._inflight: list[Slot] = []
        # Manager-level idle holder (model warm post-grace).
        # When grace expires without a thread match, the sidecar is NOT
        # torn down -- it migrates here and stays alive for idle_seconds.
        # Next slot of same model_tag inherits the handle; different
        # model_tag tears it down first.
        self._idle_handle: SidecarHandle | None = None
        self._idle_model_tag: str | None = None
        self._idle_expires_at: float | None = None
        # Latest request's keep_alive intent across the ACTIVE_MATCH
        # chain on a single warm slot. Reset per anchor (_process_slot entry);
        # captured on ACTIVE for the anchor and on each ACTIVE_MATCH promotion
        # of a matched follow-up; consumed (cleared) at grace→idle entry. The
        # "latest request wins" rule mirrors Ollama keep_alive semantics
        # (timer resets on request receipt, not on response completion).
        # Without this, stale keep_alive from request N leaks into IDLE_HOT
        # window computed after request N+M — an edge case in keep_alive ordering.
        self._latest_keep_alive_s: int | None = None
        # /status metrics counters for client-disconnect
        # eviction observability. Updated in worker_loop's is_evicted branch.
        self._eviction_count: int = 0
        self._last_evicted_at_iso: str | None = None
        #  background sweeper state-row finalizer counters.
        # Finalizes the disconnect-eviction path evictions that landed audit-only on the hot path
        # (deferred state-row write to keep worker_loop off SQLite
        # fsync stall). Sweeper is OFF the hot path — its sync SQL is fine.
        self._slots_finalized_lifetime: int = 0
        self._last_sweep_iso: str | None = None
        self._sweeper_task: asyncio.Task | None = None
        self._worker_task: asyncio.Task | None = None
        # Live inference monitor. Pure-observer plane:
        # worker_loop never reads/writes live_generation/live_output, so the
        # single-mutator FSM invariant is structurally untouched. _spawn_seq is
        # bumped by worker_loop at each _active_handle assignment so the poller
        # can detect a fixed-port (11500) sidecar swap across its httpx await.
        self._spawn_seq: int = 0
        self.live_generation: dict | None = None  # written ONLY by LiveSlotsPoller
        self.live_output = LiveOutputBuffer()      # fed ONLY by the streaming tee
        self._live_poller = None
        self._live_poller_task: asyncio.Task | None = None
        # per-resident live-inference monitor (cap>=2). The single
        # LiveSlotsPoller above stays the cap<=1 path (byte-identical). At cap>=2 a
        # LiveResidentsSupervisor task polls EACH resident's /slots and writes its
        # generation into ``live_generations`` (keyed by model_tag), mirroring the
        # most-recently-active resident into ``live_generation`` (the back-compat
        # alias status_snapshot + live_stream.py already read). It also caches per-GPU
        # free VRAM off the hot path so status_snapshot can emit ``vram[]`` await-free
        # (status_snapshot is LOCK-FREE and must NEVER call nvidia-smi synchronously).
        self.live_generations: dict[str, dict] = {}   # model_tag -> generation block
        self._vram_free_mib: list[int] | None = None  # per-GPU free MiB (cached ~1Hz)
        self._vram_total_mib: list[int] | None = None  # per-GPU total MiB (boot-time read)
        self._live_supervisor = None
        self._live_supervisor_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        # Removed unused self._lock = asyncio.Lock(). It had
        # zero acquire sites (single-task worker_loop discipline protects
        # _active_handle / _active_slot mutations). If a future
        # second mutating task, re-add it purposefully and wrap the call
        # sites that actually need it.
        #
        # MULTI-SLOT PHASE-0 (resident-registry scaffold): introduce the
        # registry data structure anticipated above, PINNED to one resident
        # (``MAX_PARALLEL_SIDECARS == 1``) so runtime behaviour is byte-for-byte
        # identical to the deployed single-sidecar manager. In Phase-0 the
        # registry is NOT yet authoritative — the FSM keeps driving the
        # ``_active_*`` / ``_inflight`` / ``_spawn_seq`` / ``_idle_*`` attributes
        # verbatim (zero call-site churn = the 480-green proof is a tautology).
        # The lone resident's ``inflight`` SHARES the live ``self._inflight``
        # list object so the registry view never diverges from the fan-out rider
        # set; the scalar fields stay at placeholders until Phase-1 migrates the
        # FSM onto the registry behind its own concurrency tests.
        #
        # ``_registry_lock`` is the lock to "re-add purposefully"
        # when a second mutating task lands. It is DEFINED here but INTENTIONALLY
        # UNACQUIRED in Phase-0 (single-task worker_loop discipline is unchanged);
        # the Phase-1 dispatcher/driver-tasks that introduce a second mutator MUST
        # acquire it around every ``_residents`` read-modify-write.
        self._residents: dict[str, Resident] = {}
        self._registry_lock = asyncio.Lock()
        # strong refs to load-bearing fire-and-forget tasks
        # (teardown / inbox-requeue / death-reap). asyncio holds only a WEAK ref to
        # a bare create_task result, so an unreferenced task can be GC'd mid-flight,
        # silently cancelling the cleanup. _spawn_bg() parks each here and discards
        # on completion. Only used on the cap>=2 dispatcher path.
        self._bg_tasks: set[asyncio.Task] = set()
        # Eagerly create the one permanent Phase-0 resident (never dropped in
        # Phase-0 — its lifetime is the manager's). ``inflight`` is the SAME list
        # object as ``self._inflight`` so in-place append/remove stay in sync.
        self._residents[_SINGLETON_RESIDENT_KEY] = Resident(
            inflight=self._inflight,
            # The lone resident HOLDS the manager's grace/idle timers (same
            # objects at max=1, so every existing self.grace/self.idle read stays
            # byte-identical). A later phase gives each concurrent resident its OWN timer
            # pair + routes the FSM through the active resident's pair.
            grace=self.grace,
            idle=self.idle,
        )
        self._binary_fd: int | None = None  # TOCTOU-pinned fd
        # Event bus for /ws/state subscribers (v0.2 §11.1 redacted)
        self.event_bus = EventBus()
        # Injection points (default = real subprocess_mgr; tests inject mocks)
        self._spawn = spawn_fn or spawn_sidecar
        self._wait_healthy = health_fn or wait_until_healthy
        self._sigterm = sigterm_fn or drained_sigterm
        self._vram_verify = vram_fn or verify_vram_cleared
        # _complete_fn: Phase 3 will replace with httpx → llama-server /v1/chat/completions
        self._complete_fn = complete_fn or self._default_complete
        # flap/degradation telemetry (observe-only)
        self._telemetry = init_telemetry(
            log_dir=getattr(boot.storage, "state_db_path", None) and
                     boot.storage.state_db_path.parent / "telemetry",
            enabled=True,
        )

    async def _default_complete(self, slot: Slot, handle: SidecarHandle) -> dict | None:
        """Default no-op completion (Phase 2). Phase 3 wires httpx proxy via DI."""
        await asyncio.sleep(0.001)
        return None

    async def submit_and_wait(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        timeout_s: float = 7200.0,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> tuple[Slot, Any]:
        """Submit + await the slot's completion. Returns (slot, completion_result).

         caller may pass a ``disconnect_event`` that the
        route's ``watch_disconnect`` task sets on client close. The queue
        pops with eviction-awareness; if the slot is evicted before activation,
        worker_loop fails the completion_future with SlotEvictedError which
        propagates out of this ``await`` for the caller to map to HTTP 499.
        """
        slot = await self.submit(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            wait_for_completion=True,
            disconnect_event=disconnect_event,
        )
        try:
            result = await asyncio.wait_for(slot.completion_future, timeout=timeout_s)
        except asyncio.TimeoutError:
            raise
        return slot, result

    async def submit_for_streaming(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> Slot:
        """SSE streaming pass-through.

        Submit a request that will be consumed by an SSE-streaming route. Unlike
        submit_and_wait, this returns the slot immediately (without awaiting
        the completion_future). The slot has two pre-armed asyncio.Events:

          - ``slot.stream_ready_event``: worker_loop sets this once the slot
            reaches ACTIVE and the SidecarHandle is stored on
            ``slot.stream_handle``. The route awaits this before opening its
            own httpx.stream() to the sidecar's port.
          - ``slot.stream_done_event``: the route sets this when the stream
            closes (normal exhaustion, client disconnect, or error). Only then
            does worker_loop advance the slot ACTIVE → GRACE.

        client_meta MUST include ``"stream": True`` for worker_loop to
        recognise the streaming path. Existing non-streaming callers are
        unaffected.

        By design, keeping the slot
        ACTIVE for the full stream lifetime prevents ACTIVE_MATCH from
        promoting a second submission against the same sidecar (single-slot
        invariant preserved).
        """
        slot = await self.submit(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
            wait_for_completion=True,  # still attach future so error paths surface
            disconnect_event=disconnect_event,
        )
        # Pre-arm the streaming coordination events. worker_loop will check for
        # client_meta["stream"] == True and use these instead of calling
        # self._complete_fn.
        slot.stream_ready_event = asyncio.Event()
        slot.stream_done_event = asyncio.Event()
        slot.stream_handle = None
        return slot

    # === Error propagation on worker exceptions ==============================

    def _fail_completion_future(self, slot: Slot, exc: BaseException) -> None:
        """If a slot has a pending completion_future, mark it failed (don't hang caller)."""
        fut = slot.completion_future
        if fut is not None and not fut.done():
            fut.set_exception(exc)

    # === Boot lifecycle =====================================================

    def boot_reconcile(self, pid_is_alive_fn: Callable[[int], bool] | None = None) -> dict:
        """Run at startup. Returns summary dict for audit logging."""
        port_base = self.boot.runtime.default_port_base

        # 1. orphan reaper (kills /proc/<pid> llama-server orphans w/ PPid=1)
        reap = boot_orphan_reaper(port_base=port_base)

        # 2. foreign GPU detect — informational only (we don't refuse to start here;
        #    that's a CLI-flag decision)
        foreign = detect_foreign_gpu_apps()

        # 3. state.sqlite reconcile: any slot whose pid is no longer alive → COLD
        check_alive = pid_is_alive_fn or _pid_is_alive
        # read + slot-write stay on state_db_session; audit-write uses pool.
        with state_db_session(self.boot.storage.state_db_path) as conn:
            stale_pids = known_active_pids(conn)
            live_pids = {pid for pid in stale_pids if check_alive(pid)}
            reconciled = reconcile_orphaned_slots(conn, live_pids)
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(
                conn,
                "boot_reconcile",
                {
                    "orphans_reaped": reap["reaped"],
                    "foreign_gpu_apps_count": len(foreign),
                    "slots_reconciled_to_cold": reconciled,
                },
            )

        return {
            "orphans_reaped": reap["reaped"],
            "orphans_failed": reap["failed"],
            "foreign_gpu_apps": foreign,
            "slots_reconciled_to_cold": reconciled,
        }

    def verify_binary(self) -> bool:
        """Verify + TOCTOU-pin llama_server_binary at boot (v0.2 §7.1).

        Empty expected_sha256 = dev mode -- verify_binary_sha256 returns True
        with no fd pinning; spawn_sidecar falls back to path-based exec.
        Non-empty + matching hash = an inode-pinned fd is held on
        self._binary_fd and every spawn execs via ``/proc/self/fd/<fd>``,
        closing the swap window.
        """
        binary_path = self.boot.runtime.llama_server_binary
        expected = self.boot.runtime.llama_server_binary_sha256
        if not verify_binary_sha256(binary_path, expected):
            return False
        if self._binary_fd is not None:
            # Defensive: close prior fd if verify_binary called twice
            with contextlib.suppress(OSError):
                os.close(self._binary_fd)
            self._binary_fd = None
        if expected:
            self._binary_fd = open_and_verify_binary(binary_path, expected)
            if self._binary_fd is None:
                log.error(
                    "binary hash drift between verify and fd-pin -- refusing"
                )
                return False
        return True

    # === Request acceptance =================================================

    async def submit(
        self,
        model_tag: str,
        prompt: str = "",
        thread_id: str = "",
        context: list[dict] | None = None,
        client_meta: dict | None = None,
        wait_for_completion: bool = False,
        disconnect_event: "asyncio.Event | None" = None,
    ) -> Slot:
        """Accept a fresh inference request.

        - If thread_id is empty, auto-derive from prompt-prefix-hash (prefix-hash fix).
        - If grace window is open for this thread+model → enqueue at FIFO HEAD
          + restart grace timer (max_grace_extensions cap applies).
        - Otherwise → normal FIFO enqueue.

        Raises RuntimeError if the manager is shutting down (v0.2.1).
        """
        # C3 fix: refuse new submissions after shutdown signal so
        # callers fail fast instead of hanging on a completion_future that
        # will never resolve (worker is dead, queue.close() clears staging).
        # Raises queue.QueueClosed to match existing test_manager contract +
        # the documented queue-closed exception type.
        if self._stop_event.is_set():
            from turbohaul.queue import QueueClosed
            raise QueueClosed(
                "TurbohaulManager is shutting down — new submissions are refused."
            )

        if not thread_id:
            thread_id = derive_thread_id_prefix_hash(prompt, model_tag)

        slot = Slot.new(
            model_tag=model_tag,
            prompt=prompt,
            thread_id=thread_id,
            context=context,
            client_meta=client_meta,
        )
        # caller-attached disconnect_event (lazy-init in
        # the route's own loop). None for non-HTTP callers
        # (BootInventory orphan-replay, internal probes, tests pre-attach).
        slot.disconnect_event = disconnect_event

        # Attach a future BEFORE enqueue so worker_loop can resolve it on completion.
        if wait_for_completion:
            slot.completion_future = asyncio.get_running_loop().create_future()

        # Grace-window matched-thread shortcut
        if self.grace.matches(thread_id, model_tag):
            await self.queue.enqueue_head(slot)
            # restart_for_followup may return False if at extension cap; that's fine,
            # the request still queues at head once, but the slot will pop next cycle.
            self.grace.restart_for_followup()
        else:
            await self.queue.enqueue(slot)

        # Audit — Note: slot-write stays on state_db_session; audit-write goes
        # through the pool wrapped in asyncio.to_thread (sync-only guard).
        with state_db_session(self.boot.storage.state_db_path) as conn:
            upsert_slot(
                conn,
                {
                    "slot_id": slot.slot_id,
                    "model_tag": slot.model_tag,
                    "thread_id": slot.thread_id,
                    "state": slot.state.value,
                    "client_meta": slot.client_meta,
                },
            )

        def _audit_submit() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                record_audit_event(
                    audit_conn,
                    "submit",
                    {"thread_id_prefix": (thread_id or "")[:8], "model_tag": model_tag},
                    slot_id=slot.slot_id,
                )

        await asyncio.to_thread(_audit_submit)

        # telemetry — capture request arrival + queue state
        try:
            self._telemetry.on_request_arrival(slot)
            self._telemetry.on_queue_state(self.queue.depth(), slot)
        except Exception:
            pass  # observe-only: never break the hot path

        return slot

    # === Status snapshot =====================================================

    def status_snapshot(self) -> dict:
        """/status payload per v0.2 §9.3."""
        depth = self.queue.depth()

        active_info: dict | None = None
        loading_info: dict | None = None
        if self._active_slot is not None:
            slot = self._active_slot
            # FE LOADING transition fix: split status into ACTIVE vs the
            # pre-active transitional states (STAGED / PRE_LOADING /
            # LOADING / READY). Before this split, FE saw active=null
            # for the whole 5-30s cold-load window — reads as a hang.
            state_v = slot.state.value
            if state_v == "ACTIVE" or state_v == "ACTIVE_MATCH":
                if self._active_handle is not None:
                    active_info = {
                        "slot_id": slot.slot_id,
                        "model_tag": slot.model_tag,
                        "state": state_v,
                        # Redaction: only first 8 chars of thread_id exposed (v0.2 §11.1)
                        "thread_id_prefix": (slot.thread_id or "")[:8],
                        "pid": self._active_handle.pid,
                        "port": self._active_handle.port,
                    }
            elif state_v in {"STAGED", "PRE_LOADING", "LOADING", "READY"}:
                elapsed = 0.0
                started = getattr(slot, "started_loading_at", None) or getattr(slot, "received_at", None)
                if started is not None:
                    elapsed = max(0.0, time.monotonic() - started)
                loading_info = {
                    "slot_id": slot.slot_id,
                    "model_tag": slot.model_tag,
                    "state": state_v,
                    "thread_id_prefix": (slot.thread_id or "")[:8],
                    "elapsed_s": round(elapsed, 1),
                    "pid": self._active_handle.pid if self._active_handle else None,
                    "port": self._active_handle.port if self._active_handle else None,
                }

        grace_info: dict | None = None
        if not self.grace.expired():
            grace_info = {
                "remaining_s": int(self.grace.remaining_s()),
                "extension_count": self.grace.extension_count,
                "max_extensions": self.grace.max_extensions,
                "thread_id_prefix": (self.grace.thread_id or "")[:8] if self.grace.thread_id else "",
                "model_tag": self.grace.model_tag,
            }

        idle_info: dict | None = None
        # /status idle snapshot reflects the manager-level
        # _idle_* holder (which IS the warm sidecar), not the legacy
        # IdleHotTimer (which only tracks the model name).
        if (
            self._idle_handle is not None
            and self._idle_expires_at is not None
            and time.monotonic() < self._idle_expires_at
        ):
            idle_info = {
                "remaining_s": int(self._idle_expires_at - time.monotonic()),
                "model_tag": self._idle_model_tag,
            }
        elif not self.idle.expired():
            # Backward compat: when idle_seconds=0 (test mode) the warm
            # holder is not used and self.idle still tracks "last model".
            idle_info = {
                "remaining_s": int(self.idle.remaining_s()),
                "model_tag": self.idle.model_tag,
            }

        # Cache the vram ref ONCE so the null-check + list() see the same value.
        # (status_snapshot is a sync def with no await, so single-threaded asyncio
        # already makes it atomic vs the supervisor's _vram_free_mib write — this is
        # belt-and-suspenders against a future await/second-writer ever sneaking in.)
        vram_cache = self._vram_free_mib
        return {
            "queue": {
                "acceptance_buffer_depth": depth["acceptance_buffer_depth"],
                "staging_queue_depth": depth["staging_queue_depth"],
                "staging_queue_max": depth["staging_queue_max"],
            },
            "active": active_info,
            "loading": loading_info,
            "grace": grace_info,
            "idle_hot": idle_info,
            # client-disconnect eviction observability.
            "evictions": {
                "total_lifetime": self._eviction_count,
                "last_evicted_at": self._last_evicted_at_iso,
            },
            #  background sweeper that finalizes the
            # state-row for the disconnect-eviction path evictions (deferred from the hot path per
            # the hot path). Sweeper runs every background_sweep_interval_s.
            "background_sweeper": {
                "last_sweep_iso": self._last_sweep_iso,
                "slots_finalized_lifetime": self._slots_finalized_lifetime,
            },
            "parallel_slots": {
                # Design #1: live in-flight rider count when a fan-out is active
                # (best-effort, await-free same-loop read), else 1 if a handle is
                # warm, else 0. `max` is the active sidecar's --parallel width
                # when known (handle.parallel), falling back to the process-count
                # config knob.
                "used": (
                    len(self._inflight)
                    if self._inflight
                    else (1 if self._active_handle else 0)
                ),
                "max": (
                    getattr(self._active_handle, "parallel", None)
                    or self.runtime.queue.max_parallel_sidecars
                ),
            },
            # Live inference monitor: tok/s + progress, written await-free by the
            # LiveSlotsPoller (idle default = single idle_generation() shape).
            # Counts/rates only — no prompt/response/IP/full-thread-id (the
            # 8-char generation_id is non-reversible). At cap>=2 this is the
            # back-compat ALIAS = the most-recently-active resident's generation
            # (the supervisor mirrors it); the per-resident blocks ride residents[].
            "generation": self.live_generation or idle_generation(),
            # multi-slot observability. ``residents`` = the live
            # per-model sidecars (EMPTY at cap<=1: the legacy singleton is excluded —
            # active/loading/grace above carry that state). ``vram`` = per-GPU free
            # MiB cached off the hot path by the supervisor (null at cap<=1 / probe-
            # down). BOTH are await-free + lock-free (status_snapshot stays sync).
            "residents": self._residents_snapshot(),
            "vram": list(vram_cache) if vram_cache is not None else None,
            "vram_total_mib": list(self._vram_total_mib) if self._vram_total_mib is not None else None,
        }

    def _residents_snapshot(self) -> "list[dict]":
        """Await-free per-resident view for /status (cap>=2 multi-slot observability).

        EMPTY at cap<=1 (``_SINGLETON_RESIDENT_KEY`` + DEAD residents excluded — the
        legacy active/loading/grace/idle_hot fields carry the single-sidecar state).
        Snapshots ``_residents`` via ``list()`` so a concurrent dispatcher mutation
        can't 'dict changed size during iteration'; reads each resident's scalars
        directly (sole-writer driver discipline). LOCK-FREE — never acquires
        ``_registry_lock`` (mirrors status_snapshot's hot-path-safe contract). Each
        entry carries the resident's live generation block (from ``live_generations``)
        so the FE shows per-model tok/s without a second round-trip."""
        out: list[dict] = []
        for k, r in list(self._residents.items()):
            if k == _SINGLETON_RESIDENT_KEY or r.state is ResidentState.DEAD:
                continue
            handle = r.handle
            pid = handle.pid if handle is not None else r.booting_pid
            idle_in = None
            if (
                r.state is ResidentState.IDLE_EVICTABLE
                and r.idle_expires_at is not None
            ):
                idle_in = max(0, int(r.idle_expires_at - time.monotonic()))
            out.append({
                "model_tag": r.model_tag,
                "state": r.state.value,
                "port": r.port,
                "pid": pid,
                "spawn_seq": r.spawn_seq,
                "reserved_need_mib": r.reserved_need_mib,
                "parallel": r.parallel,
                "main_gpu": r.main_gpu,
                "split_mode": r.split_mode,
                "inflight": len(r.inflight),
                "idle_expires_in_s": idle_in,
                "generation": self.live_generations.get(r.model_tag),
            })
        return out

    # === Port allocation =====================================================

    def _alloc_port(self) -> int:
        """Lowest free sidecar port in ``[default_port_base, +100)``.

        A port is "held" if any LIVE resident in ``self._residents`` reports it
        via ``Resident.port`` (the Phase-1 authoritative listen port). Phase-0
        keeps ``MAX_PARALLEL_SIDECARS == 1`` and the lone resident's ``port``
        stays at its placeholder (``None``) because the FSM still drives the
        listen port through ``_active_handle.port``, not the registry scalar —
        so with one resident the window is empty of holds and this returns
        ``default_port_base`` verbatim (== the deployed hard-coded 11500). The
        scan only does real work once Phase-1 binds resident ports for a second
        concurrent sidecar; introducing it now means the spawn path is already
        port-registry aware with ZERO behaviour change at max=1.
        """
        base = self.boot.runtime.default_port_base
        held = {
            r.port
            for r in self._residents.values()
            if r.port is not None
        }
        for port in range(base, base + 100):
            if port not in held:
                return port
        # Window exhausted (>=100 live residents on contiguous ports). Phase-0
        # can never reach this (single resident); Phase-1's dispatcher gates
        # spawns on MAX_PARALLEL_SIDECARS long before 100 ports are claimed, so
        # this is a defensive fallback, not a live path. Return base so the
        # caller's spawn attempt fails fast on a real bind collision rather than
        # silently picking an out-of-window port.
        return base

    # === Spawn-sequence (live-monitor generation_id) =========================

    def _active_resident(self) -> "Resident | None":
        """The resident that owns the CURRENT active sidecar.

        Phase-0 (``MAX_PARALLEL_SIDECARS == 1``) has exactly one resident under
        ``_SINGLETON_RESIDENT_KEY``, so the active resident is unambiguously that
        singleton. Phase-1 keys residents by ``model_tag`` and resolves the
        active one from the FSM's current model; until then return the singleton
        (or ``None`` if the registry is somehow empty, so callers degrade-open).
        """
        return self._residents.get(_SINGLETON_RESIDENT_KEY)

    def _bump_spawn_seq(self) -> None:
        """Advance the spawn counter for a new active handle (worker_loop only).

        Single chokepoint: bumps the legacy global ``_spawn_seq`` AND mirrors it
        onto the active resident's ``spawn_seq`` so the registry view never
        diverges from the global. At max=1 the resident value equals the global
        exactly, so the live-monitor generation_id is unchanged. Keeping the
        global write here preserves every existing read site that still reads
        ``_spawn_seq`` directly (zero churn, identical behaviour); the per-read
        migration to ``_active_spawn_seq`` happens incrementally.
        """
        self._spawn_seq += 1
        resident = self._active_resident()
        if resident is not None:
            resident.spawn_seq = self._spawn_seq

    def _active_spawn_seq(self) -> int:
        """The active resident's spawn_seq (live-monitor generation_id input).

        Returns the active resident's mirrored ``spawn_seq``; falls back to the
        legacy global ``_spawn_seq`` if no resident is registered. At max=1 the
        two are identical (``_bump_spawn_seq`` keeps them in lock-step), so the
        unified generation_id computed by the metrics poller and the streaming
        tee is byte-for-byte unchanged from the deployed manager.
        """
        resident = self._active_resident()
        if resident is not None:
            return resident.spawn_seq
        return self._spawn_seq

    def _spawn_seq_for_model(self, model_tag: "str | None") -> int:
        """spawn_seq of the resident actually serving ``model_tag`` (live-monitor
        generation_id input for the streaming text tee).

        At cap>=2 the dispatcher path never bumps the singleton's spawn_seq, so
        ``_active_spawn_seq`` (which resolves the singleton) stays 0 while the
        metrics supervisor hashes the model_tag resident's bumped spawn_seq. The
        text-plane tee must use THIS value so its generation_id matches the
        LiveOutputBuffer key the supervisor publishes as the anchor -- otherwise
        the live pane subscribes to an unfed buffer and shows nothing. Falls back to ``_active_spawn_seq`` at cap<=1, where no
        model_tag-keyed resident exists -> byte-identical generation_id.
        """
        r = self._live_resident_for(model_tag)
        if r is not None:
            return r.spawn_seq
        return self._active_spawn_seq()

    # === state-migration mirror chokepoints ==================
    # Each writes the legacy manager-global scalar (AUTHORITATIVE — zero churn
    # for existing readers) AND mirrors it onto the active resident, exactly
    # like ``_bump_spawn_seq`` does for spawn_seq. At MAX_PARALLEL_SIDECARS == 1
    # the resident value tracks the global 1:1 (byte-identical); a later phase's dispatcher
    # flips the resident copy to authoritative + migrates the read sites.

    def _set_active_handle(self, handle: "SidecarHandle | None") -> None:
        self._active_handle = handle
        r = self._active_resident()
        if r is not None:
            r.handle = handle

    def _set_active_slot(self, slot: "Slot | None") -> None:
        self._active_slot = slot
        r = self._active_resident()
        if r is not None:
            r.active_slot = slot

    def _set_idle_holder(
        self,
        handle: "SidecarHandle | None",
        model_tag: str | None,
        expires_at: float | None,
    ) -> None:
        self._idle_handle = handle
        self._idle_model_tag = model_tag
        self._idle_expires_at = expires_at
        r = self._active_resident()
        if r is not None:
            r.idle_handle = handle
            r.idle_model_tag = model_tag
            r.idle_expires_at = expires_at

    def _clear_idle_holder(self) -> None:
        self._set_idle_holder(None, None, None)

    def _set_idle_expires_at(self, expires_at: float | None) -> None:
        self._idle_expires_at = expires_at
        r = self._active_resident()
        if r is not None:
            r.idle_expires_at = expires_at

    def _set_latest_keep_alive(self, value: int | None) -> None:
        self._latest_keep_alive_s = value
        r = self._active_resident()
        if r is not None:
            r.latest_keep_alive_s = value

    # === Worker loop (full FSM-driven cycle) =================================

    async def worker_loop(self) -> None:
        """Drive the FSM forever: pop → spawn → active → complete → grace → pop → idle.

        Per v0.2 §6. Subprocess interactions are dependency-injected via ctor (spawn_fn,
        health_fn, sigterm_fn, vram_fn, complete_fn). Default implementations call the
        real subprocess_mgr functions. Tests inject mocks.
        """
        log.info("worker_loop started")
        # cap>=2 routes to the multi-slot dispatcher; the cap<=1
        # body below stays byte-identical (the 493-forked gate). The cap is the
        # CONFIG KNOB (default 1; flipped to 2 in prod at cutover, and overridden
        # to 2 by the new concurrency tests) — NOT the module constant — so every
        # existing fixture + the deployed runtime stay single-slot.
        if self.runtime.queue.max_parallel_sidecars >= 2:
            return await self._dispatch_loop()
        while not self._stop_event.is_set():
            # Model-affinity hint for pop_next: the model currently warm in the
            # idle holder (preferred) or the active slot. This is a READ of the
            # manager scalars; a fire-and-forget _teardown_idle_holder task may
            # null _idle_handle/_idle_model_tag concurrently, but the read pair
            # below is AWAIT-FREE (atomic in the single-threaded event loop) and
            # `warm` is only a HINT -- a stale value just falls back to FIFO. No
            # new lock and no new mutator are introduced. warm=None => strict
            # FIFO (back-compat). _idle_model_tag is the confirmed attribute
            # (manager.__init__) for the idle holder's model tag.
            if self._idle_handle is not None:
                warm = self._idle_model_tag
            elif self._active_slot is not None:
                warm = self._active_slot.model_tag
            else:
                warm = None
            slot = await self.queue.pop_next(warm_model_tag=warm)
            if slot is None:
                # Inline idle-tick + fire-and-forget
                # plus an identity-guarded debounce.
                # Identity guard: capture `expires` into a local; only reset _idle_expires_at
                # if it is STILL the same object we observed. Prevents the race where a
                # concurrent reset (request promotion repopulates _idle_expires_at to a
                # fresh T+120 window) would otherwise be wiped by our stale-T0 debounce
                # → teardown fires on a legitimate fresh window → warm holder killed
                # mid-promotion. PL #16848 mandate.
                expires = self._idle_expires_at
                if (
                    self._idle_handle is not None
                    and expires is not None
                    and time.monotonic() >= expires
                ):
                    if self._idle_expires_at is expires:  # identity check
                        self._set_idle_expires_at(None)
                        # Fire-and-forget — don't block worker_loop on
                        # the 5s SIGTERM grace + wait4 of the llama-server child.
                        # add done callback to log failures instead of silently swallowing.
                        _task = asyncio.create_task(
                            self._teardown_idle_holder("idle_expired")
                        )
                        _task.add_done_callback(
                            lambda t: t.exception() and log.error(
                                "idle teardown failed: %s", t.exception()
                            )
                        )
                await asyncio.sleep(0.05)
                continue

            # client-disconnect eviction handling.
            if slot.is_evicted:
                self._fail_completion_future(
                    slot,
                    SlotEvictedError(
                        f"slot {slot.slot_id} evicted: client disconnect"
                    ),
                )
                # Audit-emit via the pool path; NO sync
                # state_db_session(mark_slot_ended) on the hot path —
                # SQLite fsync 1-3s stalls would bypass the pool entirely.
                # State-row finalization defers to terminal-park / the background sweeper
                # background sweeper RC stub.
                try:
                    await self._audit_event_only_async(
                        slot.slot_id,
                        "slot_evicted",
                        {
                            "reason": "client_disconnect",
                            "time_in_queue_s": time.monotonic() - slot.created_at,
                        },
                    )
                except Exception:
                    log.exception(
                        "slot_evicted audit emit failed (best-effort)"
                    )
                # /status metric bookkeeping
                self._eviction_count += 1
                self._last_evicted_at_iso = datetime.now(
                    timezone.utc,
                ).isoformat()
                # Inline mirror + identity guard — same
                # idle-tick block on the eviction branch so consecutive
                # evictions don't starve idle expiry.
                expires = self._idle_expires_at
                if (
                    self._idle_handle is not None
                    and expires is not None
                    and time.monotonic() >= expires
                ):
                    if self._idle_expires_at is expires:
                        self._set_idle_expires_at(None)
                        # add done callback to log failures instead of silently swallowing.
                        _task2 = asyncio.create_task(
                            self._teardown_idle_holder("idle_expired")
                        )
                        _task2.add_done_callback(
                            lambda t: t.exception() and log.error(
                                "idle teardown failed: %s", t.exception()
                            )
                        )
                continue

            try:
                await self._process_slot(slot)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("slot %s processing failed", slot.slot_id)
                self._fail_completion_future(slot, e)
                # C2 fix: teardown active sidecar BEFORE force_cold
                # to prevent PID leak. _force_cold only updates DB state; without
                # teardown the spawned llama-server keeps running and the
                # single-slot invariant breaks until boot_orphan_reaper at next
                # restart. Best-effort — don't let teardown failure mask the
                # original exception that triggered this path.
                if self._active_handle is not None:
                    try:
                        await self._teardown(slot, "worker-uncaught-exception")
                    except Exception:
                        log.exception(
                            "teardown during worker exception failed (best-effort)"
                        )
                await self._force_cold(slot, "worker-uncaught-exception")
        log.info("worker_loop exited")

    # === multi-slot dispatcher (cap>=2 path) ================
    # worker_loop branches into _dispatch_loop when max_parallel_sidecars>=2. The
    # cap<=1 path above is UNTOUCHED (byte-identical, the 493-forked gate). The
    # dispatcher is the SOLE writer of the _residents dict + each resident's
    # ``state``, ALWAYS under _registry_lock. Each resident is driven by its own
    # long-lived _drive_resident task (sole writer of that r.* lock-free in ACTIVE).
    # The request-route (submit/chat_completion) NEVER takes _registry_lock.

    def _model_residents(self) -> "list[Resident]":
        """Live model_tag-keyed residents (EXCLUDES the legacy singleton + DEAD).
        Await-free ``list()`` snapshot so a concurrent mutation can't error."""
        return [
            r
            for k, r in list(self._residents.items())
            if k != _SINGLETON_RESIDENT_KEY and r.state is not ResidentState.DEAD
        ]

    def _dispatch_warm_hint(self) -> "str | None":
        """pop_next affinity hint at cap>=2: the most-recently-active live
        resident's model_tag. HINT only — stale value falls back to FIFO, never
        starves a routable follower."""
        live = self._model_residents()
        if not live:
            return None
        return max(live, key=lambda r: r.last_active_monotonic).model_tag

    async def _dispatch_loop(self) -> None:
        """The cap>=2 dispatcher: pop -> route-or-reserve -> loop. Never blocks on
        an ACTIVE wait (the per-resident drivers own that)."""
        log.info(
            "dispatcher started (max_parallel_sidecars=%d)",
            self.runtime.queue.max_parallel_sidecars,
        )
        while not self._stop_event.is_set():
            slot = await self.queue.pop_next(
                warm_model_tag=self._dispatch_warm_hint()
            )
            if slot is None:
                await asyncio.sleep(0.05)
                continue
            if slot.is_evicted:
                self._handle_evicted_slot(slot)
                continue
            try:
                await self._route_or_reserve(slot)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("dispatch of slot %s failed", slot.slot_id)
                self._fail_completion_future(
                    slot, RuntimeError("dispatch routing failed")
                )
        log.info("dispatcher exited")

    def _handle_evicted_slot(self, slot: Slot) -> None:
        """Client-disconnect eviction (mirrors worker_loop's is_evicted branch).
        The per-resident idle tick is owned by the drivers, so it is NOT here."""
        self._fail_completion_future(
            slot,
            SlotEvictedError(f"slot {slot.slot_id} evicted: client disconnect"),
        )
        self._spawn_bg(self._audit_evicted_slot(slot))
        self._eviction_count += 1
        self._last_evicted_at_iso = datetime.now(timezone.utc).isoformat()  # noqa: UP017

    async def _audit_evicted_slot(self, slot: Slot) -> None:
        try:
            await self._audit_event_only_async(
                slot.slot_id,
                "slot_evicted",
                {
                    "reason": "client_disconnect",
                    "time_in_queue_s": time.monotonic() - slot.created_at,
                },
            )
            # telemetry — client disconnect
            try:
                elapsed = time.monotonic() - slot.created_at
                self._telemetry.on_client_disconnect(slot, "client_disconnect", elapsed)
            except Exception:
                pass
        except Exception:
            log.exception("slot_evicted audit emit failed (best-effort)")

    def _live_resident_for(self, model_tag: "str | None") -> "Resident | None":
        """The live (non-DEAD) model_tag-keyed resident, or None."""
        if model_tag is None:
            return None
        r = self._residents.get(model_tag)
        if r is None or r.state is ResidentState.DEAD:
            return None
        return r

    def _lru_idle_evictable(self) -> "Resident | None":
        """Least-recently-active resident that is IDLE_EVICTABLE with NO active
        slot and NO inflight riders. Busy residents are NEVER evictable."""
        cands = [
            r
            for r in self._model_residents()
            if r.state is ResidentState.IDLE_EVICTABLE
            and r.active_slot is None
            and not r.inflight
        ]
        if not cands:
            return None
        return min(cands, key=lambda r: r.last_active_monotonic)

    def _spawn_bg(self, coro) -> "asyncio.Task":
        """Fire-and-forget a load-bearing background coroutine while holding a STRONG
        reference. asyncio keeps only a weak ref to a bare ``create_task``
        result, so an unreferenced teardown/requeue/reap task can be GC-cancelled
        mid-flight. The task is parked in ``self._bg_tasks`` and self-removes on
        completion."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    def _defer_unroutable(self, slot: Slot) -> None:
        """MISS+full with no idle victim: bounded re-queue. A per-slot defer
        counter caps the spins so an unroutable slot can't busy-loop forever; on
        cap exhaustion the caller's future fails (503-equivalent) instead of
        starving. enqueue_head preserves FIFO for when a resident frees."""
        n = getattr(slot, "_dispatch_defer_count", 0) + 1
        slot._dispatch_defer_count = n
        if n > _MAX_DISPATCH_DEFERS:
            self._fail_completion_future(
                slot,
                RuntimeError(
                    f"no capacity for model {slot.model_tag!r} after "
                    f"{_MAX_DISPATCH_DEFERS} defers (all residents busy)"
                ),
            )
            return
        self._spawn_bg(self._requeue_after_backoff(slot))

    async def _requeue_after_backoff(self, slot: Slot) -> None:
        """Re-enqueue an unroutable slot at the HEAD after a short backoff so the
        dispatcher can't burn its whole defer budget in a sub-second busy-loop
        while residents are transiently busy. Detached, so it MUST fail the slot's
        future on ANY failure (cancel during the sleep at shutdown, or enqueue_head
        raising on a closing queue) rather than silently dropping it — otherwise the
        client hangs until upstream timeout (B4). Mirrors _route_or_reserve's
        except-fails-the-future safety net."""
        try:
            await asyncio.sleep(0.05)
            await self.queue.enqueue_head(slot)
        except asyncio.CancelledError:
            self._fail_completion_future(
                slot, RuntimeError("requeue cancelled during shutdown")
            )
            raise
        except Exception as e:  # noqa: BLE001 -- detached path must not swallow
            log.exception("requeue-after-backoff failed for slot %s", slot.slot_id)
            self._fail_completion_future(slot, e)

    async def _requeue_slots_or_fail(self, slots: "list[Slot]") -> None:
        """Re-enqueue a BATCH of drained inbox slots at the head IN ORDER. ONE task
        awaiting sequentially preserves FIFO among them — fanning out one _spawn_bg
        per slot would let N tasks race on the queue lock and scramble the order
        (a fairness bug). Every slot that can't be re-enqueued — queue closing at
        shutdown (enqueue_head raises) or cancel — has its future FAILED, never
        dropped (the B3-drain analogue of the B4 _requeue_after_backoff safety net).
        On cancel, the remaining batch is failed too so nothing hangs."""
        for i, slot in enumerate(slots):
            try:
                await self.queue.enqueue_head(slot)
            except asyncio.CancelledError:
                for s in slots[i:]:
                    self._fail_completion_future(
                        s, RuntimeError("inbox-drain requeue cancelled during shutdown")
                    )
                raise
            except Exception as e:  # noqa: BLE001 -- detached path must not swallow
                log.exception("inbox-drain requeue failed for slot %s", slot.slot_id)
                self._fail_completion_future(slot, e)

    async def _route_or_reserve(self, slot: Slot) -> None:
        """ONE atomic _registry_lock critical section: HIT route / MISS+capacity
        reserve / MISS+full LRU-evict-then-reserve / else bounded-defer. The
        IDLE_EVICTABLE->ACTIVE flip and the inbox.put are CO-LOCATED under the lock
        (closes the lost-slot race); the 5s nvidia-smi is scoped to the reserve
        branch only (PL D2)."""
        async with self._registry_lock:
            r = self._live_resident_for(slot.model_tag)
            if r is not None:
                # HIT: reclaim from idle + hand to the driver, all under the lock.
                if r.state is ResidentState.IDLE_EVICTABLE:
                    r.state = ResidentState.ACTIVE
                r.last_active_monotonic = time.monotonic()
                if r.inbox is not None:
                    r.inbox.put_nowait(slot)
                return
            cap = self.runtime.queue.max_parallel_sidecars
            if len(self._model_residents()) >= cap:
                victim = self._lru_idle_evictable()
                if victim is None:
                    self._defer_unroutable(slot)
                    return
                self._begin_evict_locked(victim)
            await self._reserve_and_start_locked(slot)

    async def _reserve_and_start_locked(self, slot: Slot) -> None:
        """CALLER HOLDS _registry_lock. Read the model footprint, run the
        cross-resident VRAM gate, alloc a port, insert a RESERVED_LOADING
        placeholder (reserving its budget against concurrent reserves), and start
        its driver. The slow spawn+health happen OUTSIDE the lock inside the driver
        (the placeholder already reserves the budget)."""
        need, parallel, main_gpu, split_mode, sleep_idle_s = self._read_model_footprint(
            slot.model_tag
        )
        if not self._vram_admits_locked(need, parallel, main_gpu, split_mode):
            # Refuse (cross-resident over-commit) — mirror the safety-gate refusal.
            log.warning(
                "cross-resident VRAM gate refused spawn for %s "
                "(need=%dMiB parallel=%d gpu=%d split=%s)",
                slot.model_tag, need, parallel, main_gpu, split_mode,
            )
            self._fail_completion_future(
                slot,
                RuntimeError(
                    f"cross-resident VRAM gate refused {slot.model_tag!r}: "
                    f"need {need} MiB would over-commit GPU {main_gpu}"
                ),
            )
            return
        port = self._alloc_port()
        r = Resident(
            model_tag=slot.model_tag,
            port=port,
            state=ResidentState.RESERVED_LOADING,
            reserved_need_mib=need,
            parallel=parallel,
            main_gpu=main_gpu,
            split_mode=split_mode,
            sleep_idle_seconds=sleep_idle_s,
            last_active_monotonic=time.monotonic(),
            grace=GraceTimer(
                grace_seconds=self.runtime.queue.grace_seconds,
                max_extensions=self.runtime.queue.max_grace_extensions,
            ),
            idle=IdleHotTimer(
                idle_seconds=self.runtime.queue.idle_hot_load_seconds
            ),
            inbox=asyncio.Queue(),
        )
        r.inbox.put_nowait(slot)  # the slot that triggered the reservation
        self._residents[slot.model_tag] = r
        r.driver_task = asyncio.create_task(self._drive_resident(r))
        r.driver_task.add_done_callback(
            lambda t, rr=r: self._on_driver_done(rr, t)
        )

    def _read_model_footprint(
        self, model_tag: "str | None"
    ) -> "tuple[int, int, int, str, int]":
        """(reserved_need_mib, parallel, main_gpu, split_mode, sleep_idle_seconds)
        from the manifest. Sync file read; the dispatcher calls it under the lock
        so the placeholder's reserved budget is exact. Missing manifest ->
        (0,1,0,'layer',0) = degrade-open for the footprint (the per-spawn
        all_safety_gates still runs in the driver). sleep_idle_seconds=0 means
        'use global default' — the driver falls back to
        runtime.queue.idle_hot_load_seconds."""
        try:
            m = read_manifest(self.boot.storage.manifests_path, model_tag)
        except FileNotFoundError:
            return 0, 1, 0, "layer", 0
        flags = m.llama_server_flags or {}
        gguf_mib = int((m.gguf_size_bytes or 0) // (1024 * 1024))
        ctx = int(flags.get("ctx_size") or m.context_size or 0)
        kv_quant = flags.get("cache_type_k") or "f16"
        kv_mib = estimate_kv_cache_mib(ctx, m.gguf_size_bytes or 0, kv_quant)
        parallel = max(1, int(flags.get("parallel", 1) or 1))
        # par_extra: the marginal per-slot compute floor for parallel>1, ON TOP of
        # the model footprint (the red-team's fix: reserve the FULL body+KV, not
        # just the compute floor).
        par_extra = (parallel - 1) * PER_SLOT_COMPUTE_FLOOR_MIB
        expected_vram_mib = int((m.expected_vram_bytes or 0) // (1024 * 1024))
        # cpu_moe / n_cpu_moe offload expert weights to HOST RAM, so the gguf+kv
        # heuristic (which counts EVERY weight as GPU-resident) grossly over-reserves an
        # expert-offload model and wrong-refuses a co-resident that actually fits
        # (live-E2E 2026-06-25: 35b n-cpu-moe @500K reserved 29.9GiB vs a proven
        # 19.4GiB). For those configs the operator's MEASURED expected_vram_bytes is the
        # only accurate footprint -> trust it. Normal models keep the conservative
        # max(declared, gguf+kv). The driver's per-spawn all_safety_gates re-checks LIVE
        # free VRAM before the sidecar binds either way.
        cpu_moe = bool(
            flags.get("cpu_moe") or int(flags.get("n_cpu_moe", 0) or 0) > 0
        )
        if cpu_moe and expected_vram_mib > 0:
            need = expected_vram_mib + par_extra
        else:
            need = max(expected_vram_mib, gguf_mib + kv_mib) + par_extra
        main_gpu = int(flags.get("main_gpu", 0) or 0)
        split_mode = str(flags.get("split_mode", "layer") or "layer")
        sleep_idle_s = int(flags.get("sleep_idle_seconds") or 0)
        return need, parallel, main_gpu, split_mode, sleep_idle_s

    def _vram_admits_locked(
        self, need: int, parallel: int, main_gpu: int, split_mode: str
    ) -> bool:
        """CALLER HOLDS _registry_lock. Cross-resident over-commit gate.

        Co-residence is supported ONLY for single-GPU-pinned
        (``split_mode='none'``) models on DISTINCT cards. A layer/row/tensor-split
        sibling spans every visible GPU, so no "free distinct card" can be
        guaranteed AND the aggregate-budget reserve math would double-count an
        already-loaded sibling (B1). Until per-card layer-split accounting lands,
        refuse-blind any co-residence that isn't tensor-isolated on distinct cards.
        The FIRST resident (no sibling) admits regardless of split_mode — it keeps
        the legacy degrade-open(parallel:1)/refuse-blind(parallel>1) doctrine via
        the per-spawn all_safety_gates in the driver.

        B1: for the admitted (split=none/distinct-card) shape the reserve only
        charges siblings whose VRAM is NOT YET reflected in the live nvidia-smi
        probe — i.e. those still RESERVED_LOADING (spawned, weights not loaded) on
        THIS card. An ACTIVE/GRACE/IDLE_EVICTABLE sibling is already loaded =>
        already absent from the free reading; also subtracting its reserved_need_mib
        double-charges it and wrong-refuses the steady state (two warm models is the
        whole point of the feature). With this gate every admitted sibling is on a distinct
        card so this normally contributes 0; the same-card term is defensive against
        a future relaxation."""
        siblings = self._model_residents()
        new_split = (split_mode or "layer").lower()
        if siblings:
            # Refuse-blind: incoming must be tensor-isolated...
            if new_split != "none":
                return False
            for r in siblings:
                # ...and every existing sibling must be tensor-isolated on a
                # DIFFERENT card (else its weights occupy this card too).
                if (r.split_mode or "layer").lower() != "none":
                    return False
                if r.main_gpu == main_gpu:
                    return False
        free_fit, _min_card, _n = _vram_budget(split_mode, main_gpu)
        if free_fit is None:
            # nvidia-smi unreadable.
            if siblings:
                return False  # co-residence without a probe = refuse-blind
            return True  # lone spawn -> driver's all_safety_gates owns the doctrine
        # Reserve ONLY still-booting (not-yet-in-probe) siblings on THIS card (B1).
        reserve = 0
        for r in siblings:
            if r.state is ResidentState.RESERVED_LOADING and r.main_gpu == main_gpu:
                reserve += r.reserved_need_mib
        return (free_fit - reserve) >= need

    def _begin_evict_locked(self, r: "Resident") -> None:
        """CALLER HOLDS _registry_lock. LRU-evict (or driver-death reap): drain any
        inbox slots back to the queue, mark DEAD, deregister from _residents (so
        capacity frees + dispatcher stops routing), then teardown the captured
        handle on a DETACHED task (keyed off the captured ref, not the dict entry).

        B3: the inbox drain happens BEFORE the DEAD early-return — a driver that
        already died via its own finally set DEAD without draining (or only partly),
        and this reaper path is the only other drain site, so an already-DEAD
        resident must still surrender its queued slots. The queue hand-off is
        one-shot, so re-draining an already-empty inbox is a harmless no-op."""
        # Drain unstarted inbox slots back to the main queue (not lost) — EVERY call.
        # This method is sync (under _registry_lock) so it can't await; collect the
        # slots IN ORDER and hand them to ONE ordered bg re-enqueue task (not one
        # task per slot, which would race the queue lock and scramble FIFO).
        if r.inbox is not None and not r.inbox.empty():
            drained: list[Slot] = []
            while not r.inbox.empty():
                try:
                    drained.append(r.inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if drained:
                self._spawn_bg(self._requeue_slots_or_fail(drained))
        if r.state is ResidentState.DEAD:
            return
        r.state = ResidentState.DEAD
        if self._residents.get(r.model_tag) is r:
            del self._residents[r.model_tag]
        self._spawn_bg(self._evict_teardown(r))

    async def _evict_teardown(self, r: "Resident") -> None:
        """Detached: claim torn_down (exactly-once vs the driver finally) then
        teardown r's live handle off the hot path. B2: if r is still RESERVED_LOADING
        (sidecar spawned but handle not yet published) the handle is None and
        ``booting_pid`` is the ONLY reference to the live process — reap by pid so
        an LRU/death teardown of a booting resident can't leak it."""
        async with self._registry_lock:
            if r.torn_down:
                return
            r.torn_down = True
            handle = r.handle
            r.handle = None
            booting_pid = r.booting_pid
            r.booting_pid = None
        if handle is not None and handle.is_alive():
            await self._reap_resident_handle(handle)
        elif booting_pid is not None:
            await self._reap_booting_pid(booting_pid)

    async def _reap_booting_pid(self, pid: int) -> None:
        """SIGTERM-then-REAP a sidecar that spawned but whose handle was never
        published (driver cancelled/failed mid ``_wait_healthy``). ``booting_pid`` is
        the only reference to that process — and ``_live_handle_pids`` actively
        PROTECTS it from the orphan reaper — so without this it leaks VRAM+PID forever
        (B2). The sidecar is OUR child (same-process spawn at cap>=2), so we MUST
        ``waitpid`` it or it lingers as a ZOMBIE/defunct entry that keeps the PID
        allocated. Bounded: WNOHANG-poll for a grace window,
        escalate to SIGKILL, final blocking reap. Best-effort, off the event loop (the
        sleeps run in the worker thread, never blocking the loop).

        PID-RECYCLE GUARD (PL pre-cutover polish #4): a child that exits becomes a
        zombie holding its PID until reaped — but a COMPETING reaper (the lost
        ``SidecarHandle``'s ``Popen`` being GC-reaped by CPython after the cancel
        unwinds its frame) can reap it first, freeing the PID for the OS to recycle to
        an UNRELATED process. So before EVERY signal we re-confirm pid is still our
        ALIVE child via ``waitpid(WNOHANG)``: ``ChildProcessError`` (ECHILD) => not
        ours / already reaped => STOP (never signal a recycled pid); ``(pid, _)`` =>
        ours, just exited => reaped here => done; ``(0, 0)`` => ours, alive => safe to
        signal. This narrows the recycle window to the microseconds between the check
        and the kill (the standard best-effort bound for raw-pid reaping)."""
        def _own_live_child() -> bool:
            """True iff pid is our ALIVE child (safe to signal). False if it is gone,
            already reaped (zombie collected here), or no longer ours (ECHILD)."""
            try:
                wpid, _ = os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                return False  # not ours / already reaped -> do NOT signal a recycled pid
            return wpid == 0  # 0 = still running; pid = exited+reaped just now (done)

        def _term_and_reap() -> None:
            if not _own_live_child():
                return
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            for _ in range(30):  # ~3s grace for SIGTERM, re-checking ownership each iter
                if not _own_live_child():
                    return  # exited (reaped) or no longer ours
                time.sleep(0.1)
            if not _own_live_child():
                return
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            with contextlib.suppress(ChildProcessError, ProcessLookupError, OSError):
                os.waitpid(pid, 0)  # final blocking reap of the SIGKILL'd child
        try:
            await asyncio.to_thread(_term_and_reap)
        except Exception:
            log.exception("booting_pid reap failed (best-effort) pid=%d", pid)

    async def _reap_resident_handle(self, handle) -> None:
        """Drained-SIGTERM a resident's handle via the DI _sigterm seam (consistent
        with the legacy _teardown; async + non-blocking + mockable). Best-effort."""
        try:
            await self._sigterm(
                handle,
                drained_window_s=float(
                    self.runtime.queue.drained_sigterm_window_active_s
                ),
                is_active=False,
                cold_window_s=float(
                    self.runtime.queue.drained_sigterm_window_cold_s
                ),
            )
        except Exception:
            log.exception("resident handle reap failed (best-effort)")

    def _on_driver_done(self, r: "Resident", task: "asyncio.Task") -> None:
        """done_callback (SYNC — cannot await): schedule the supervisor reaper on
        EVERY driver exit, including CANCELLED. B2: a driver cancelled during
        RESERVED_LOADING leaves a live sidecar (handle still None, ``booting_pid``
        set) — bailing on ``task.cancelled()`` meant the reaper never ran and the
        process leaked. The driver finally already reaps + claims torn_down, so the
        reaper is exactly-once-safe (it no-ops when torn_down is already set) and
        only ADDS the failed-future / booting_pid safety net on the cancel path.
        Normal completion (idle-evict self-teardown) likewise no-ops."""
        cancelled = task.cancelled()
        exc = None if cancelled else task.exception()
        if cancelled:
            log.warning("driver for %s cancelled — scheduling reap", r.model_tag)
        elif exc is not None:
            log.warning("driver for %s died: %r — scheduling reap", r.model_tag, exc)
        self._spawn_bg(self._reap_dead_resident(r))

    async def _reap_dead_resident(self, r: "Resident") -> None:
        """Supervisor reaper: fail every pending future the dead driver owned
        (active_slot + ALL inflight riders) + unblock streaming routes, then
        mark DEAD + deregister + teardown (exactly-once via torn_down)."""
        # Always fail the pending futures (the driver abandoned them on death);
        # teardown stays exactly-once via the torn_down claim in _begin_evict_locked.
        async with self._registry_lock:
            pend = [r.active_slot, *list(r.inflight)]
        for s in pend:
            if s is None:
                continue
            if s.completion_future is not None and not s.completion_future.done():
                self._fail_completion_future(
                    s, RuntimeError(f"resident {r.model_tag!r} driver died")
                )
            ev = getattr(s, "stream_done_event", None)
            if ev is not None and not ev.is_set():
                ev.set()
        async with self._registry_lock:
            self._begin_evict_locked(r)

    def _idle_window_seconds(self, keep_alive_s: "int | None", default: int) -> int:
        """Per-resident idle-hot window from the latest keep_alive intent (mirrors
        the cap<=1 grace->idle math: None->default, <0->KEEP_ALIVE_MAX_S cap,
        else min(keep,cap); 0 disables idle)."""
        if keep_alive_s is None:
            return default
        if keep_alive_s < 0:
            return KEEP_ALIVE_MAX_S
        return min(keep_alive_s, KEEP_ALIVE_MAX_S)

    async def _spawn_for_resident(
        self, r: "Resident", slot: Slot
    ) -> "SidecarHandle | None":
        """Spawn r's sidecar OUTSIDE _registry_lock (the placeholder already
        reserved the budget). Per-spawn host safety gate (per-model), spawn,
        capture booting_pid under the lock (closes the reaper window), health-wait,
        then publish r.handle + state=ACTIVE under the lock. On failure: fail the
        slot future and return None (the finally/supervisor reaps the resident)."""
        argv: list[str] = []
        try:
            manifest = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
            argv = flags_to_argv(manifest.llama_server_flags)
            gguf_path = (
                self.boot.storage.blob_store_path
                / "sha256"
                / manifest.gguf_blob_sha256[:2]
                / manifest.gguf_blob_sha256
            )
        except FileNotFoundError:
            gguf_path = self.boot.storage.blob_store_path / "missing.gguf"
        # Per-model host safety gate (RAM/IO/load + the per-spawn VRAM/KV gate). The
        # CROSS-resident over-commit gate already ran under the lock at reserve.
        if self.runtime.queue.safety_enabled:
            gate_ok = await self._run_spawn_safety_gate(slot)
            if not gate_ok:
                self._fail_completion_future(
                    slot, RuntimeError("safety gates refused spawn")
                )
                return None
        handle = self._spawn(
            self.boot.runtime.llama_server_binary,
            gguf_path,
            r.port,
            slot.model_tag,
            argv,
            binary_fd=self._binary_fd,
        )
        async with self._registry_lock:
            r.booting_pid = handle.pid  # in the reaper union before handle is set
        slot.port = handle.port
        slot.pid = handle.pid
        healthy = await self._wait_healthy(
            r.port, self.runtime.queue.loading_health_timeout_s,
            is_alive=handle.is_alive,
        )
        if not healthy:
            self._fail_completion_future(
                slot, RuntimeError("loading-fail-health-timeout")
            )
            # Reap the spawned-but-unhealthy sidecar + clear booting_pid, else it
            # lingers in _live_handle_pids (so the orphan reaper SKIPS it) =
            # permanent PID+VRAM leak. Mirrors the cap<=1 path's _teardown here.
            async with self._registry_lock:
                r.booting_pid = None
            if handle.is_alive():
                await self._reap_resident_handle(handle)
            return None
        async with self._registry_lock:
            r.handle = handle
            r.booting_pid = None
            r.state = ResidentState.ACTIVE
            r.spawn_seq += 1  # live-monitor: new active handle on this resident
        return handle

    async def _run_spawn_safety_gate(self, slot: Slot) -> bool:
        """Run all_safety_gates for slot.model_tag exactly as the cap<=1 path does
        (manifest-derived params). Returns True on all-pass, False on any refusal
        (logged + audited)."""
        mv = mc = mg = 0
        mq = "f16"
        mnk = False
        mp = 1
        msm = "layer"
        mmg = 0
        mcm = False
        try:
            m = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
            mv = m.expected_vram_bytes or 0
            mg = m.gguf_size_bytes or 0
            mc = m.llama_server_flags.get("ctx_size") or m.context_size or 0
            mq = m.llama_server_flags.get("cache_type_k") or "f16"
            mnk = bool(m.llama_server_flags.get("no_kv_offload", False))
            mp = int(m.llama_server_flags.get("parallel", 1) or 1)
            msm = str(m.llama_server_flags.get("split_mode", "layer") or "layer")
            mmg = int(m.llama_server_flags.get("main_gpu", 0) or 0)
            mcm = bool(
                m.llama_server_flags.get("cpu_moe")
                or int(m.llama_server_flags.get("n_cpu_moe", 0) or 0) > 0
            )
        except FileNotFoundError:
            mv = 0
        gates = await asyncio.to_thread(all_safety_gates,
            min_free_ram_mib=self.runtime.queue.safety_min_free_ram_mib,
            min_free_vram_mib=self.runtime.queue.safety_min_free_vram_mib,
            max_load_per_core=self.runtime.queue.safety_max_load_per_core,
            max_iowait_percent=self.runtime.queue.safety_max_iowait_percent,
            manifest_expected_vram_bytes=mv,
            iowait_sample_window_s=self.runtime.queue.safety_iowait_sample_window_s,
            ctx_size=mc, gguf_size_bytes=mg, kv_cache_quant=mq,
            no_kv_offload=mnk, parallel=mp, split_mode=msm, main_gpu=mmg,
            cpu_moe_offload=mcm,
        )
        failed = [g for g in gates if not g.ok]
        if failed:
            log.warning(
                "safety gates refused spawn for %s: %s",
                slot.slot_id, "; ".join(f"{g.name}: {g.detail}" for g in failed),
            )
            return False
        return True

    async def _drive_resident(self, r: "Resident") -> None:
        """Long-lived per-resident driver (cap>=2). Spawns r's sidecar once, then
        serves slots from r.inbox through ACTIVE->GRACE, parking in IDLE_EVICTABLE
        between requests until idle expiry / eviction / death. SOLE writer of r.*
        lock-free in ACTIVE; takes _registry_lock only for the idle<->active<->dead
        transitions the dispatcher also touches."""
        handle = None
        slot = None  # bound before the first await so the finally can fail it
        try:
            slot = await r.inbox.get()  # the reservation's first slot
            handle = await self._spawn_for_resident(r, slot)
            if handle is None:
                return
            # Per-model idle timeout: read from the manifest's sleep_idle_seconds
            # (threaded through Resident.sleep_idle_seconds). -1 = pin/keep-warm
            # (never idle-unload), 0 = fall back to global default, N>0 = that
            # model's idle timeout in seconds. This is what evicts the 35b sub-agent
            # model too early.
            if r.sleep_idle_seconds == -1:
                per_model_idle = KEEP_ALIVE_MAX_S  # pin-warm, never idle-unload
            elif r.sleep_idle_seconds > 0:
                per_model_idle = r.sleep_idle_seconds
            else:
                per_model_idle = self.runtime.queue.idle_hot_load_seconds
            while not self._stop_event.is_set():
                async with self._registry_lock:
                    if r.state is ResidentState.DEAD:
                        break
                    r.state = ResidentState.ACTIVE
                    r.active_slot = slot
                    r.last_active_monotonic = time.monotonic()
                await self._serve_on_resident(r, slot, handle)
                idle_window = self._idle_window_seconds(
                    r.latest_keep_alive_s, per_model_idle
                )
                async with self._registry_lock:
                    if r.state is ResidentState.DEAD:
                        break
                    r.active_slot = None
                    r.last_active_monotonic = time.monotonic()
                    if idle_window <= 0:
                        self._begin_evict_locked(r)
                        break
                    r.state = ResidentState.IDLE_EVICTABLE
                    r.idle_expires_at = time.monotonic() + idle_window
                try:
                    slot = await asyncio.wait_for(
                        r.inbox.get(), timeout=idle_window
                    )
                except TimeoutError:
                    async with self._registry_lock:
                        if (
                            r.state is ResidentState.IDLE_EVICTABLE
                            and r.inbox.empty()
                        ):
                            self._begin_evict_locked(r)
                            break
                        if not r.inbox.empty():
                            slot = r.inbox.get_nowait()
                            continue
                        continue
        except asyncio.CancelledError:
            raise
        finally:
            # EXACTLY-ONCE teardown claim (vs _evict_teardown). Whoever claims
            # torn_down owns the FULL cleanup: reap the live process (handle OR the
            # still-booting pid — B2), drain unstarted inbox riders back to the queue
            # (B3), and fail the anchor slot if it died before the serve loop owned it.
            booting_pid = None
            pending_slots: list[Slot] = []
            async with self._registry_lock:
                claim = not r.torn_down
                if claim:
                    r.torn_down = True
                    handle = r.handle
                    r.handle = None
                    booting_pid = r.booting_pid
                    r.booting_pid = None
                    if r.inbox is not None:
                        while not r.inbox.empty():
                            try:
                                pending_slots.append(r.inbox.get_nowait())
                            except asyncio.QueueEmpty:
                                break
                    if self._residents.get(r.model_tag) is r:
                        r.state = ResidentState.DEAD
                        del self._residents[r.model_tag]
            if claim:
                if handle is not None and handle.is_alive():
                    await self._reap_resident_handle(handle)
                elif booting_pid is not None:
                    await self._reap_booting_pid(booting_pid)  # B2: handle never set
                # Re-queue inbox riders that never started — another resident serves
                # (or fail their future if the queue is closing at shutdown). Async
                # context here (lock released), so await the ordered batch directly
                # to preserve FIFO among the riders.
                await self._requeue_slots_or_fail(pending_slots)
                # Fail the anchor slot if it died before the serve loop took
                # ownership (cancel/spawn-fail during RESERVED_LOADING) — the
                # supervisor's active_slot/inflight sweep can't see it then.
                # _spawn_for_resident already fails it on the health-timeout path,
                # so the done() guard in _fail_completion_future keeps this idempotent.
                if (
                    slot is not None
                    and slot is not r.active_slot
                    and slot not in r.inflight
                ):
                    self._fail_completion_future(
                        slot,
                        RuntimeError(
                            f"resident {r.model_tag!r} driver exited before serve"
                        ),
                    )

    async def _serve_on_resident(
        self, r: "Resident", slot: Slot, handle
    ) -> None:
        """Serve one anchor slot on r's warm handle: ACTIVE (parallel fan-out /
        streaming / non-streaming complete) -> drain -> GRACE (ACTIVE_MATCH warm
        reuse within the grace window). Writes r.* directly (sole-writer). Does NOT
        handle idle handoff (the driver loop owns IDLE_EVICTABLE)."""
        n_parallel = max(1, getattr(handle, "parallel", 1))
        if slot.state is SlotState.STAGED:
            transition(slot, SlotState.LOADING)
        transition(slot, SlotState.ACTIVE)
        slot.started_active_at = time.monotonic()
        await self._audit_async(slot, "active")
        r.latest_keep_alive_s = (slot.client_meta or {}).get("keep_alive_s")
        is_streaming = (
            isinstance(slot.client_meta, dict)
            and bool(slot.client_meta.get("stream", False))
            and slot.stream_ready_event is not None
            and slot.stream_done_event is not None
        )
        if n_parallel > 1:
            await self._fan_out_on_resident(r, slot, handle, n_parallel)
        elif is_streaming:
            slot.stream_handle = handle
            slot.stream_ready_event.set()
            try:
                await asyncio.wait_for(slot.stream_done_event.wait(), timeout=3600.0)
            except TimeoutError:
                log.warning("streaming slot %s exceeded 3600s", slot.slot_id)
            if slot.completion_future is not None and not slot.completion_future.done():
                slot.completion_future.set_result({"_streamed": True})
        else:
            result = await self._complete_fn(slot, handle)
            if slot.completion_future is not None and not slot.completion_future.done():
                slot.completion_future.set_result(result)
        # GRACE + ACTIVE_MATCH warm reuse within the grace window (per-resident).
        transition(slot, SlotState.GRACE)
        slot.grace_started_at = time.monotonic()
        r.grace.start(slot.thread_id, slot.model_tag)
        await self._audit_async(slot, "grace_enter")
        deadline = time.monotonic() + self.runtime.queue.grace_seconds
        while time.monotonic() < deadline and not self._stop_event.is_set():
            matched = await self.queue.pop_matched_thread(
                slot.thread_id, slot.model_tag
            )
            if matched is not None:
                matched.port = handle.port
                matched.pid = handle.pid
                r.active_slot = matched
                try:
                    transition(matched, SlotState.ACTIVE_MATCH)
                    transition(matched, SlotState.ACTIVE)
                    r.latest_keep_alive_s = (
                        matched.client_meta or {}
                    ).get("keep_alive_s")
                    m_stream = (
                        isinstance(matched.client_meta, dict)
                        and matched.client_meta.get("stream", False)
                        and matched.stream_ready_event is not None
                        and matched.stream_done_event is not None
                    )
                    if m_stream:
                        matched.stream_handle = handle
                        matched.stream_ready_event.set()
                        try:
                            await asyncio.wait_for(
                                matched.stream_done_event.wait(), timeout=3600.0
                            )
                        except TimeoutError:
                            pass
                        if matched.completion_future is not None and not matched.completion_future.done():
                            matched.completion_future.set_result({"_streamed": True})
                    else:
                        res = await self._complete_fn(matched, handle)
                        if matched.completion_future is not None and not matched.completion_future.done():
                            matched.completion_future.set_result(res)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 -- per-slot isolation
                    self._fail_completion_future(matched, e)
                    log.exception("active_match completion failed %s", matched.slot_id)
                transition(matched, SlotState.GRACE)
                if r.grace.restart_for_followup():
                    deadline = time.monotonic() + self.runtime.queue.grace_seconds
                transition(matched, SlotState.POPPED)
                with state_db_session(self.boot.storage.state_db_path) as conn:
                    mark_slot_ended(conn, matched.slot_id, "active_match_completed")
                r.active_slot = slot
                continue
            await asyncio.sleep(0.05)
        transition(slot, SlotState.POPPED)
        assert not r.inflight, "Design #1 invariant: riders drain before GRACE exit"

    async def _fan_out_on_resident(
        self, r: "Resident", anchor: Slot, handle, n_parallel: int
    ) -> None:
        """Per-resident CONTINUOUS concurrent serve, up to ``n_parallel`` in-flight,
        riding ``r.inflight`` (IN-PLACE -- never rebind).

        LAYER-2 (per-model parallelism): same-model requests are routed by the
        dispatcher to THIS resident's ``r.inbox`` (model-pure), NOT left in the global
        staging queue -- so riders are pulled from ``r.inbox`` and the pipe is kept full
        up to ``n_parallel`` WHILE any slot is still generating, so a request arriving
        mid-burst joins immediately (llama-server ``--parallel N`` serves them). Each
        slot runs its own stream/non-stream completion, so a mixed-mode burst is fine.
        Drains every in-flight slot before returning so GRACE/teardown stays safe
        (Design #1 invariant: ``r.inflight`` empty on exit)."""
        r.inflight[:] = []
        tasks: dict[asyncio.Task, Slot] = {}

        def _launch(s: Slot) -> None:
            s.stream_handle = handle
            r.inflight.append(s)
            s_stream = (
                isinstance(s.client_meta, dict)
                and bool(s.client_meta.get("stream", False))
                and s.stream_ready_event is not None
                and s.stream_done_event is not None
            )
            if s_stream:
                s.stream_ready_event.set()
                t = asyncio.create_task(self._await_streamed_slot(s))
            else:
                t = asyncio.create_task(self._complete_one_slot(s, handle))
            tasks[t] = s

        def _admit_from_inbox() -> None:
            # Pull queued same-model riders (r.inbox is model-pure: the dispatcher only
            # routes THIS resident's model here) up to n_parallel concurrent in-flight.
            while len(tasks) < n_parallel:
                try:
                    extra = r.inbox.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if extra.is_evicted:
                    self._fail_completion_future(
                        extra, SlotEvictedError(f"slot {extra.slot_id} evicted")
                    )
                    continue
                _launch(extra)

        try:
            _launch(anchor)
            _admit_from_inbox()
            while tasks:
                done, _pending = await asyncio.wait(
                    set(tasks), return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    s = tasks.pop(t)
                    if s in r.inflight:
                        r.inflight.remove(s)
                    exc = t.exception()
                    if exc is not None:
                        self._fail_completion_future(s, exc)
                # Keep the pipe full: admit any riders that arrived mid-burst.
                _admit_from_inbox()
        finally:
            # Driver cancelled / unexpected raise mid-burst: cancel the still-running
            # per-slot tasks and fail their futures so no client hangs and no task is
            # orphaned. Normal exit leaves ``tasks`` empty -> no-op.
            if tasks:
                for t, s in list(tasks.items()):
                    t.cancel()
                    self._fail_completion_future(
                        s, RuntimeError(f"resident {r.model_tag!r} fan-out interrupted")
                    )
            r.inflight.clear()

    async def _complete_one_slot(self, s: Slot, handle) -> None:
        """Serve one NON-streaming slot on the shared handle + resolve its future."""
        try:
            res = await self._complete_fn(s, handle)
            if s.completion_future is not None and not s.completion_future.done():
                s.completion_future.set_result(res)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- per-slot isolation
            self._fail_completion_future(s, e)

    async def _await_streamed_slot(self, s: Slot) -> None:
        """Wait for a STREAMING slot's HTTP handler to finish, then resolve its future."""
        try:
            if s.stream_done_event is not None:
                await asyncio.wait_for(s.stream_done_event.wait(), timeout=3600.0)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            pass
        if s.completion_future is not None and not s.completion_future.done():
            s.completion_future.set_result({"_streamed": True})

    async def _fan_out_and_drain(
        self, anchor: Slot, handle, n_parallel: int,
    ) -> None:
        """Per-model concurrent dispatch (Design #1): serve up to ``n_parallel``
        same-model STREAMING requests concurrently on ONE shared sidecar.

        Reached ONLY from the ACTIVE branch when the active model's manifest
        declares parallel>1 (``n_parallel == handle.parallel``). parallel:1
        models never enter here -- they take the verbatim serial block, which
        stays byte-identical to today.

        Shape: a ONE-SHOT fan-out (admit up to ``n_parallel - 1`` extra riders
        from the queue exactly once -- never continuously refilling, so it
        cannot livelock the model swap) followed by a per-rider DRAIN barrier.
        Returns only when EVERY rider's route has genuinely finished its stream
        (``self._inflight`` empty), so the caller can safely advance
        ACTIVE->GRACE and (later) tear the sidecar down.

        Single-mutator discipline preserved: worker_loop is the sole writer of
        ``self._inflight``; routes only ever set their OWN
        ``slot.stream_done_event``. No lock, and the drain waiter tasks are
        owned + awaited here (cancelled on every exit), never detached.
        """
        # The anchor's mode decides the whole fan-out: STREAMING riders are
        # served by their own HTTP route (stream_ready/stream_done events);
        # NON-STREAMING riders by concurrent manager-owned _complete_fn calls.
        # Hermes sub-agents are non-streaming, so the non-streaming path is the
        # one that matters most here. Riders must match the anchor's mode
        # (the admit loop enforces it; mixed-mode riders go back to the queue).
        anchor_streaming = bool(
            isinstance(anchor.client_meta, dict)
            and anchor.client_meta.get("stream", False)
        )
        # (1) The anchor is rider 0 -- already ACTIVE on `handle`.
        # MUTATE IN PLACE, never rebind self._inflight. The lone
        # resident's ``inflight`` is the SAME list object (Resident(inflight=
        # self._inflight)); rebinding would orphan that view. ``[:] =`` keeps the
        # one canonical list so r.inflight never diverges (the dispatcher reads r.inflight).
        self._inflight[:] = [anchor]
        anchor.stream_handle = handle
        if anchor_streaming and anchor.stream_ready_event is not None:
            anchor.stream_ready_event.set()
        keep_alives: list[int | None] = [
            (anchor.client_meta or {}).get("keep_alive_s")
        ]

        # (2) ONE-SHOT admit loop: pull up to n_parallel-1 MORE same-model
        # streaming riders and attach them to the SAME sidecar handle.
        while len(self._inflight) < n_parallel:
            extra = await self.queue.pop_next(warm_model_tag=anchor.model_tag)
            if extra is None:
                break  # queue has no more admissible riders right now
            extra_streaming = (
                isinstance(extra.client_meta, dict)
                and bool(extra.client_meta.get("stream", False))
            )
            if (
                extra.model_tag != anchor.model_tag
                or extra_streaming != anchor_streaming
            ):
                # Wrong model OR a different streaming-mode than the anchor: not
                # a rider for THIS fan-out. Push it back to the HEAD (the serial
                # path / next fan-out handles it) and stop -- this is one-shot.
                await self.queue.enqueue_head(extra)
                break
            if extra.is_evicted:
                # Client disconnected before admit -> do NOT burn a slot on it.
                self._fail_completion_future(
                    extra,
                    SlotEvictedError("client disconnected before fan-out admit"),
                )
                await self._audit_async(extra, "evicted_pre_fanout")
                continue
            # Promote the rider onto the shared sidecar. Drift-guard mirrors the
            # ACTIVE_MATCH promotion: if the slot's state drifted, reset its pid
            # to None FIRST (so _force_cold can NEVER sigterm the shared sidecar)
            # then cold-park it and drop it from the batch.
            # ⚠ SAFETY SEQUENCE MIRRORED in _admit_nonstreaming_riders (the
            # continuous-admit refill). Keep the pid=None-BEFORE-_force_cold
            # ordering identical in BOTH sites if you ever touch one.
            extra.port = handle.port
            extra.pid = handle.pid
            try:
                transition(extra, SlotState.LOADING)
                transition(extra, SlotState.ACTIVE)
            except InvalidTransition as drift_err:
                log.warning(
                    "fan-out rider %s state drift (%s) -- dropping; %s",
                    extra.slot_id, extra.state.value, drift_err,
                )
                extra.pid = None  # MUST precede _force_cold: never reap shared sidecar
                self._fail_completion_future(extra, drift_err)
                await self._force_cold(
                    extra, f"fanout_rider_drift:{extra.state.value}"
                )
                continue
            extra.started_active_at = time.monotonic()
            extra.stream_handle = handle
            # Append to the rider set BEFORE signalling ready, so any observer
            # that sees the route unblock also sees the slot in self._inflight
            # (consistent state; keeps the drain-before-swap invariant exact).
            self._inflight.append(extra)
            keep_alives.append((extra.client_meta or {}).get("keep_alive_s"))
            await self._audit_async(extra, "fanout_rider_active")
            if anchor_streaming and extra.stream_ready_event is not None:
                extra.stream_ready_event.set()

        # keep_alive aggregate = MAX across riders (deterministic; the longest
        # warm-hold any rider asked for wins). Mirrors the serial anchor capture.
        _ka = [k for k in keep_alives if k is not None]
        self._set_latest_keep_alive(max(_ka) if _ka else None)

        # (3) DISPATCH + DRAIN, branched on the anchor's mode. Non-streaming
        # riders (hermes sub-agents) are served by concurrent manager-owned
        # _complete_fn calls; streaming riders by their own routes (the barrier
        # below). Every admitted rider matches the anchor's mode.
        if not anchor_streaming:
            await self._drain_nonstreaming_riders(anchor, handle, n_parallel)
            return

        # (3-stream) DRAIN BARRIER. Each rider's route sets its OWN stream_done_event in
        # its finally (normal end, client-disconnect CancelledError, typed httpx
        # error, OR the route's own STREAM_TIMEOUT_S). We wait on the GENUINE
        # done-event -- never force-removing a rider while its route's httpx may
        # still be open (red-team must-fix: teardown gates on real route close,
        # not an accounting flag). A per-rider 3600s safety cap matches the
        # serial path's wait_for; on a cap, ONLY that rider is cooperatively
        # unwound -- siblings keep streaming. Returns when self._inflight is [].
        # Keyed by slot_id (Slot is a non-frozen dataclass and thus unhashable,
        # so it cannot be a dict key directly).
        waiters: dict[str, asyncio.Task] = {}
        for s in self._inflight:
            assert s.stream_done_event is not None
            waiters[s.slot_id] = asyncio.create_task(
                asyncio.wait_for(s.stream_done_event.wait(), timeout=3600.0)
            )
        try:
            pending = set(waiters.values())
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for s in list(self._inflight):
                    w = waiters[s.slot_id]
                    if w not in done:
                        continue
                    # Distinguish a genuine stream_done from the 3600s cap.
                    if not w.cancelled() and isinstance(
                        w.exception(), asyncio.TimeoutError
                    ):
                        log.warning(
                            "fan-out rider %s hit 3600s drain cap; cooperative "
                            "unwind (siblings unaffected)", s.slot_id,
                        )
                        if (
                            s.stream_done_event is not None
                            and not s.stream_done_event.is_set()
                        ):
                            s.stream_done_event.set()
                    if (
                        s.completion_future is not None
                        and not s.completion_future.done()
                    ):
                        s.completion_future.set_result({"_streamed": True})
                    if s is not anchor:
                        # Rider terminal cleanup: the ANCHOR continues through
                        # _process_slot's normal GRACE/teardown flow, but a rider
                        # is done the instant its own route closes. Decouple it
                        # from the shared sidecar (pid=None so _force_cold can
                        # never reap the handle) and walk it to COLD so no zombie
                        # ACTIVE slot lingers.
                        s.pid = None
                        await self._force_cold(s, "fanout_rider_drained")
                    self._inflight.remove(s)
        finally:
            # On NORMAL completion self._inflight is already empty. On an
            # exception/cancel mid-drain it may still hold live riders -- set
            # their stream_done_event so their routes unwind (a truncated stream
            # when the sidecar is later reaped on the error path is acceptable
            # degraded behaviour, never a hang or zombie).
            for s in self._inflight:
                if (
                    s.stream_done_event is not None
                    and not s.stream_done_event.is_set()
                ):
                    s.stream_done_event.set()
            # No orphaned waiters: cancel + await every still-pending task.
            for w in waiters.values():
                if not w.done():
                    w.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await w
            self._inflight.clear()  # in-place, keep r.inflight alias canonical

    async def _admit_nonstreaming_riders(
        self, anchor: Slot, handle, n_parallel: int,
    ) -> list[Slot]:
        """Admit MORE same-model NON-streaming riders onto
        the shared sidecar, up to ``n_parallel`` total in-flight. Returns the
        slots newly promoted (each appended to ``self._inflight`` and ACTIVE on
        ``handle``); ``[]`` when none are admissible right now.

        Mirrors the one-shot admit promotion in ``_fan_out_and_drain`` (same
        drift-guard, evicted-skip, audit) but is called REPEATEDLY by the drain
        so a rider that arrives mid-burst joins the free ``--parallel`` slot
        immediately instead of waiting out the anchor (fixes the 4.5s/9.1s
        serialization the one-shot admit left when req2 hadn't yet reached the
        queue at anchor launch).

        Single-mutator preserved: runs in the worker_loop task -- the sole
        writer of ``self._inflight``. A wrong-model OR streaming pop is pushed
        back to the queue HEAD and STOPS the refill; that same force-FIFO-head
        path is how the queue lets a starved OTHER-model request (e.g. the main
        27b, after ``max_other_model_wait_s``) break the same-model batch so the
        model can still swap back -- so this can never livelock the swap.
        """
        admitted: list[Slot] = []
        while len(self._inflight) < n_parallel:
            extra = await self.queue.pop_next(warm_model_tag=anchor.model_tag)
            if extra is None:
                break  # nothing queued for this model right now
            extra_streaming = (
                isinstance(extra.client_meta, dict)
                and bool(extra.client_meta.get("stream", False))
            )
            if extra.model_tag != anchor.model_tag or extra_streaming:
                # Not a rider for THIS non-streaming fan-out: push back to HEAD
                # (the next worker_loop pop / fresh fan-out / model swap handles
                # it) and STOP -- one bounded refill, never a queue busy-spin.
                await self.queue.enqueue_head(extra)
                break
            if extra.is_evicted:
                self._fail_completion_future(
                    extra,
                    SlotEvictedError("client disconnected before fan-out admit"),
                )
                await self._audit_async(extra, "evicted_pre_fanout")
                continue
            # Promote onto the shared sidecar (identical to the one-shot path).
            extra.port = handle.port
            extra.pid = handle.pid
            try:
                transition(extra, SlotState.LOADING)
                transition(extra, SlotState.ACTIVE)
            except InvalidTransition as drift_err:
                log.warning(
                    "fan-out refill rider %s state drift (%s) -- dropping; %s",
                    extra.slot_id, extra.state.value, drift_err,
                )
                extra.pid = None  # MUST precede _force_cold: never reap shared sidecar
                self._fail_completion_future(extra, drift_err)
                await self._force_cold(
                    extra, f"fanout_rider_drift:{extra.state.value}"
                )
                continue
            extra.started_active_at = time.monotonic()
            extra.stream_handle = handle
            self._inflight.append(extra)
            await self._audit_async(extra, "fanout_rider_active")
            admitted.append(extra)
        return admitted

    async def _drain_nonstreaming_riders(
        self, anchor: Slot, handle, n_parallel: int,
    ) -> None:
        """Non-streaming concurrent dispatch (Design #1): fire ``_complete_fn``
        for EVERY rider at once -- each is its own httpx POST to the sidecar's
        ``--parallel`` slots, so the engine serves them concurrently via
        continuous batching. Await all, resolve each ``completion_future`` with
        its result (or fail ONLY that rider on a typed sidecar error -- siblings
        keep going), terminal-clean the riders, and clear ``self._inflight``.
        Returns only when every rider has completed (drain-before-swap holds).
        This is the path hermes sub-agents take (they call non-streaming).

        While draining, keep the sidecar's ``--parallel``
        slots full by admitting same-model riders that arrive mid-burst -- the
        loop wakes on a completion OR a short poll (``_FANOUT_ADMIT_POLL_S``),
        then refills up to ``n_parallel`` via ``_admit_nonstreaming_riders``. So
        N sub-agents fired ~together run ``n_parallel``-at-a-time (the rest
        queue) instead of serializing, while the queue's same-model batch cap
        still lets the model swap back.
        """
        # One concurrent completion task per rider, keyed by slot_id (Slot is a
        # non-frozen dataclass and thus unhashable).
        ctasks: dict[str, asyncio.Task] = {
            s.slot_id: asyncio.create_task(self._complete_fn(s, handle))
            for s in self._inflight
        }
        try:
            # Admit any same-model riders ALREADY queued at drain start. Covers
            # the one-shot admit race: req2 reaches the queue micro-seconds after
            # the anchor's one-shot admit ran, so it would otherwise idle a free
            # slot until the anchor finished.
            for s in await self._admit_nonstreaming_riders(
                anchor, handle, n_parallel
            ):
                ctasks[s.slot_id] = asyncio.create_task(
                    self._complete_fn(s, handle)
                )
            pending = set(ctasks.values())
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=_FANOUT_ADMIT_POLL_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for s in list(self._inflight):
                    t = ctasks[s.slot_id]
                    if t not in done:
                        continue
                    if t.cancelled():
                        pass  # cancelled on teardown; future handled in finally
                    elif t.exception() is not None:
                        # A typed sidecar error fails ONLY this rider's caller.
                        self._fail_completion_future(s, t.exception())
                    else:
                        result = t.result()
                        if (
                            s.completion_future is not None
                            and not s.completion_future.done()
                        ):
                            s.completion_future.set_result(result)
                    if s is not anchor:
                        # Rider terminal cleanup (the anchor continues through
                        # _process_slot's GRACE/teardown). Decouple from the
                        # shared sidecar (pid=None) and walk to COLD.
                        s.pid = None
                        await self._force_cold(s, "fanout_rider_drained")
                    self._inflight.remove(s)
                # Refill any slot freed this tick (and admit a mid-burst arrival
                # that landed while a slot was free) up to n_parallel.
                for s in await self._admit_nonstreaming_riders(
                    anchor, handle, n_parallel
                ):
                    t = asyncio.create_task(self._complete_fn(s, handle))
                    ctasks[s.slot_id] = t
                    pending.add(t)
        finally:
            # Cancel + await any still-pending task; fail any unresolved rider
            # future so no caller hangs. Clear the rider set.
            for s in list(self._inflight):
                t = ctasks.get(s.slot_id)
                if t is not None and not t.done():
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await t
                if (
                    s.completion_future is not None
                    and not s.completion_future.done()
                ):
                    self._fail_completion_future(
                        s, RuntimeError("fan-out drain aborted"),
                    )
            self._inflight.clear()  # in-place, keep r.inflight alias canonical

    async def _process_slot(self, slot: Slot) -> None:
        """Drive one slot through STAGED → LOADING → ACTIVE → GRACE → POPPED."""
        self._set_active_slot(slot)
        # Clear cross-slot keep_alive leakage. A previous slot's
        # ACTIVE_MATCH chain may have left a value here even though that
        # slot's grace→idle consumed it; defensive reset keeps the invariant
        # "value reflects this anchor cycle only" honest.
        self._set_latest_keep_alive(None)

        try:
            # Build llama-server argv from manifest if available; tolerate missing
            # manifest for testing convenience.
            argv: list[str] = []
            manifest_found = True
            try:
                manifest = read_manifest(self.boot.storage.manifests_path, slot.model_tag)
                argv = flags_to_argv(manifest.llama_server_flags)
                gguf_path = (
                    self.boot.storage.blob_store_path
                    / "sha256"
                    / manifest.gguf_blob_sha256[:2]
                    / manifest.gguf_blob_sha256
                )
            except FileNotFoundError:
                manifest_found = False
                gguf_path = self.boot.storage.blob_store_path / "missing.gguf"

            # Polish: pre-validate the
            # manifest BEFORE the teardown_idle_holder branch when a warm
            # holder for a DIFFERENT model is at risk. A bogus model_tag
            # with no manifest used to fall into the "different model"
            # path and tear down the warm holder, then the spawn would
            # inevitably hang in LOADING (missing.gguf) and health-check
            # timeout. The warm hold was lost for nothing.
            # Now: if manifest is missing AND an idle holder exists for a
            # DIFFERENT model, bail fast (LOADING -> LOADING_FAIL -> POPPED)
            # WITHOUT touching self._idle_handle. Otherwise (no holder OR
            # same-model holder which we would inherit anyway) fall through
            # to the legacy missing.gguf -> spawn-then-LOADING_FAIL path so
            # existing tests that rely on the manifest-missing tolerance
            # continue to work.
            holder_at_risk = (
                not manifest_found
                and self._idle_handle is not None
                and self._idle_model_tag != slot.model_tag
            )
            if holder_at_risk:
                transition(slot, SlotState.LOADING)
                await self._audit_async(slot, "stage_to_loading")
                transition(slot, SlotState.LOADING_FAIL)
                await self._audit_async(slot, "manifest_not_found")
                transition(slot, SlotState.POPPED)
                _mfn_conn = open_state_db(self.boot.storage.state_db_path)
                try:
                    mark_slot_ended(
                        _mfn_conn, slot.slot_id, "manifest_not_found",
                    )
                finally:
                    _mfn_conn.close()
                self._fail_completion_future(
                    slot,
                    RuntimeError(
                        f"model_tag {slot.model_tag!r} has no manifest "
                        f"(idle holder for {self._idle_model_tag!r} preserved)"
                    ),
                )
                return

            # allocate the listen port from the resident registry instead
            # of hard-coding default_port_base. At MAX_PARALLEL_SIDECARS == 1
            # this returns default_port_base (11500) verbatim -- behaviour is
            # byte-for-byte identical to the deployed manager; the indirection
            # is what lets a Phase-1 second sidecar land on the next free port.
            port = self._alloc_port()

            # STAGED → LOADING
            transition(slot, SlotState.LOADING)
            await self._audit_async(slot, "stage_to_loading")

            # Warm-inherit path. If the previous slot left
            # a warm sidecar holding the same model_tag and the idle
            # window has not expired, reuse the handle and skip spawn +
            # health-wait (sidecar is already healthy by construction).
            # capture idle_handle in local var BEFORE the check to
            # avoid TOCTOU — another coroutine could change _idle_handle
            # between the check and the warm_inherit block.
            _idle_h = self._idle_handle
            _idle_mt = self._idle_model_tag
            _idle_exp = self._idle_expires_at
            warm_inherit = (
                _idle_h is not None
                and _idle_mt == slot.model_tag
                and _idle_exp is not None
                and time.monotonic() < _idle_exp
            )
            if warm_inherit:
                handle = _idle_h
                self._clear_idle_holder()
                slot.port = handle.port
                slot.pid = handle.pid
                self._set_active_handle(handle)
                self._bump_spawn_seq()  # live-monitor: mark new active handle + mirror to active resident (sole writer = worker_loop)
                await self._audit_async(slot, "idle_hot_inherit")
                # Skip spawn + health wait; jump straight to LOADING -> ACTIVE.
                healthy = True
            else:
                # Different model OR no idle holder. If a stale holder
                # exists for a different model, tear it down before
                # spawning new -- immediate switch.
                if self._idle_handle is not None:
                    await self._teardown_idle_holder("model_swap")
                # Safety guardrails: pre-spawn host checks
                # (VRAM headroom + RAM + CPU load + IO wait). Refuse here
                # rather than spawning into an OOM / IO-stuck host.
                if self.runtime.queue.safety_enabled:
                    manifest_vram = 0
                    manifest_ctx = 0
                    manifest_gguf_bytes = 0
                    manifest_kv_quant = "f16"
                    manifest_no_kv_offload = False
                    manifest_parallel = 1
                    manifest_split_mode = "layer"
                    manifest_main_gpu = 0
                    manifest_cpu_moe = False
                    try:
                        m_for_vram = read_manifest(
                            self.boot.storage.manifests_path,
                            slot.model_tag,
                        )
                        manifest_vram = m_for_vram.expected_vram_bytes or 0
                        manifest_gguf_bytes = m_for_vram.gguf_size_bytes or 0
                        # ctx_size: prefer llama_server_flags.ctx_size (what
                        # actually gets passed to llama-server CLI); fall
                        # back to manifest.context_size.
                        manifest_ctx = (
                            m_for_vram.llama_server_flags.get("ctx_size")
                            or m_for_vram.context_size
                            or 0
                        )
                        # KV quant: derive from cache_type_k (cache_type_v
                        # assumed to match; if different, picks the larger).
                        manifest_kv_quant = (
                            m_for_vram.llama_server_flags.get("cache_type_k")
                            or "f16"
                        )
                        # --no-kv-offload: KV in host RAM, so the kv_cache_fit
                        # gate must NOT count it against VRAM (it re-checks RAM).
                        manifest_no_kv_offload = bool(
                            m_for_vram.llama_server_flags.get("no_kv_offload", False)
                        )
                        # parallel: extra concurrent llama.cpp slots add a flat
                        # per-slot compute floor to the VRAM gate (safety.py).
                        manifest_parallel = int(
                            m_for_vram.llama_server_flags.get("parallel", 1) or 1
                        )
                        # GPU placement (per-model manifest flags): split_mode
                        # drives the VRAM gate's aggregate-vs-single-card budget;
                        # main_gpu picks the card for split_mode:none. Absent ->
                        # "layer" = llama.cpp default (aggregate across all GPUs).
                        manifest_split_mode = str(
                            m_for_vram.llama_server_flags.get("split_mode", "layer")
                            or "layer"
                        )
                        manifest_main_gpu = int(
                            m_for_vram.llama_server_flags.get("main_gpu", 0) or 0
                        )
                        # cpu_moe / n_cpu_moe offload experts to RAM -> the closed-form
                        # body=gguf over-counts; the cpu-moe gate branch trusts the
                        # manifest's measured expected_vram for those configs (parity
                        # with the cap>=2 driver path).
                        manifest_cpu_moe = bool(
                            m_for_vram.llama_server_flags.get("cpu_moe")
                            or int(
                                m_for_vram.llama_server_flags.get("n_cpu_moe", 0) or 0
                            )
                            > 0
                        )
                    except FileNotFoundError:
                        manifest_vram = 0
                    gates = await asyncio.to_thread(all_safety_gates,
                        min_free_ram_mib=self.runtime.queue.safety_min_free_ram_mib,
                        min_free_vram_mib=self.runtime.queue.safety_min_free_vram_mib,
                        max_load_per_core=self.runtime.queue.safety_max_load_per_core,
                        max_iowait_percent=self.runtime.queue.safety_max_iowait_percent,
                        manifest_expected_vram_bytes=manifest_vram,
                        iowait_sample_window_s=self.runtime.queue.safety_iowait_sample_window_s,
                        ctx_size=manifest_ctx,
                        gguf_size_bytes=manifest_gguf_bytes,
                        kv_cache_quant=manifest_kv_quant,
                        no_kv_offload=manifest_no_kv_offload,
                        parallel=manifest_parallel,
                        split_mode=manifest_split_mode,
                        main_gpu=manifest_main_gpu,
                        cpu_moe_offload=manifest_cpu_moe,
                    )
                    failed = [g for g in gates if not g.ok]
                    if failed:
                        # Build a single error message + emit audit detail.
                        detail = "; ".join(
                            f"{g.name}: {g.detail}" for g in failed
                        )
                        log.warning(
                            "safety gates refused spawn for slot %s: %s",
                            slot.slot_id, detail,
                        )
                        transition(slot, SlotState.LOADING_FAIL)
                        await self._audit_async(slot, "safety_gate_refused")
                        await self._audit_event_only_async(
                            slot.slot_id,
                            "safety_gate_detail",
                            {"failed": [
                                {"name": g.name, "detail": g.detail}
                                for g in failed
                            ]},
                        )
                        transition(slot, SlotState.POPPED)
                        # No sidecar spawned; _teardown is a no-op + audit-only.
                        # Just mark slot ended + fail caller future.
                        _sg_conn = open_state_db(
                            self.boot.storage.state_db_path
                        )
                        try:
                            mark_slot_ended(
                                _sg_conn, slot.slot_id, "safety_gate_refused",
                            )
                        finally:
                            _sg_conn.close()
                        self._fail_completion_future(
                            slot,
                            RuntimeError(
                                f"safety gates refused spawn: {detail}",
                            ),
                        )
                        return
                handle = self._spawn(
                    self.boot.runtime.llama_server_binary,
                    gguf_path,
                    port,
                    slot.model_tag,
                    argv,
                    binary_fd=self._binary_fd,
                )
                slot.port = handle.port
                slot.pid = handle.pid
                self._set_active_handle(handle)
                self._bump_spawn_seq()  # live-monitor: mark new active handle + mirror to active resident (sole writer = worker_loop)
                # LOADING → ACTIVE (or LOADING_FAIL → POPPED)
                healthy = await self._wait_healthy(
                    port, self.runtime.queue.loading_health_timeout_s,
                    is_alive=handle.is_alive,
                )
            if not healthy:
                transition(slot, SlotState.LOADING_FAIL)
                await self._audit_async(slot, "loading_fail_health_timeout")
                transition(slot, SlotState.POPPED)
                await self._teardown(slot, "loading-fail-health-timeout")
                self._fail_completion_future(slot, RuntimeError("loading-fail-health-timeout"))
                return

            # Per-model concurrent-dispatch admission cap (Design #1). Pinned on
            # the handle at spawn from the actual --parallel argv, so it is
            # drift-proof vs a fresh manifest read across a warm-inherit reuse
            # cycle. NOTE: the cap is handle.parallel DIRECTLY -- NOT
            # max_parallel_sidecars (that config is the separate multi-PROCESS
            # stub, default 1; clamping to it would disable fan-out entirely).
            # For parallel:1 models this is 1, so the serial path below is taken
            # verbatim and byte-identical to today.
            n_parallel = max(1, getattr(handle, "parallel", 1))

            transition(slot, SlotState.ACTIVE)
            slot.started_active_at = time.monotonic()
            await self._audit_async(slot, "active")
            # Capture anchor's keep_alive intent for grace→idle decision.
            self._set_latest_keep_alive((slot.client_meta or {}).get("keep_alive_s"))

            # telemetry — slot assign + prefill start
            try:
                self._telemetry.on_slot_assign(slot)
                self._telemetry.on_prefill_start(slot)
            except Exception:
                pass

            # Branch on streaming mode.
            #
            # Non-streaming (existing): await self._complete_fn(slot, handle) to
            # post chat-completion, set completion_future, advance to GRACE.
            #
            # Streaming: client_meta["stream"] is True AND submit_for_streaming()
            # pre-armed slot.stream_ready_event + slot.stream_done_event. We
            # SKIP _complete_fn (the route owns the httpx streaming connection).
            # Instead: hand the SidecarHandle to the route via slot.stream_handle,
            # signal stream_ready_event so the route can open its httpx.stream(),
            # then BLOCK here on stream_done_event until the route reports the
            # stream has finished (normal close, client disconnect, or error).
            # Slot stays in ACTIVE the entire time so ACTIVE_MATCH cannot promote
            # a second submission against the same sidecar (preserving the
            # critical catch — single-slot invariant preserved).
            is_streaming = (
                isinstance(slot.client_meta, dict)
                and bool(slot.client_meta.get("stream", False))
                and slot.stream_ready_event is not None
                and slot.stream_done_event is not None
            )
            if n_parallel > 1:
                # Design #1: concurrent fan-out. Serve up to n_parallel
                # same-model riders (streaming OR non-streaming) on this ONE
                # shared sidecar, then drain them all before advancing to GRACE.
                # Reached ONLY for parallel:N models; parallel:1 falls to the
                # verbatim serial block below (byte-identical to today).
                await self._fan_out_and_drain(slot, handle, n_parallel)
            elif is_streaming:
                # Hand the sidecar handle to the route, signal ready.
                slot.stream_handle = handle
                slot.stream_ready_event.set()
                # Wait for the route to finish streaming. Stream timeout is
                # bounded — same default as non-streaming complete_fn — but
                # very long in practice (1h+ for slow-thinking models on big
                # context). If the route never signals, worker_loop unblocks
                # via timeout and proceeds to GRACE (slot already drained).
                try:
                    await asyncio.wait_for(
                        slot.stream_done_event.wait(),
                        timeout=3600.0,  # 1 hour cap; routes typically signal in seconds
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "streaming slot %s exceeded 3600s waiting for stream_done_event; "
                        "advancing to GRACE anyway",
                        slot.slot_id,
                    )
                # Resolve the completion_future so any caller awaiting the
                # slot (e.g. tests or programmatic await) is unblocked.
                if (
                    slot.completion_future is not None
                    and not slot.completion_future.done()
                ):
                    slot.completion_future.set_result({"_streamed": True})
            else:
                # Non-streaming path (existing behaviour, unchanged).
                # Completion (Phase 3 wires httpx forward; Phase 2 default is noop)
                result = await self._complete_fn(slot, handle)
                if slot.completion_future is not None and not slot.completion_future.done():
                    slot.completion_future.set_result(result)

            # Drain-before-swap hard gate (Design #1): every fan-out rider must
            # have finished before the anchor advances to GRACE / the sidecar is
            # torn down. _fan_out_and_drain returns only when self._inflight is
            # empty; for parallel:1 it is never populated, so this is a no-op.
            assert not self._inflight, (
                "Design #1 invariant: riders must drain before ACTIVE->GRACE"
            )
            # ACTIVE → GRACE
            transition(slot, SlotState.GRACE)
            slot.grace_started_at = time.monotonic()
            self.grace.start(slot.thread_id, slot.model_tag)
            await self._audit_async(slot, "grace_enter")

            # telemetry — completion
            try:
                self._telemetry.on_completion(slot, "grace_enter")
            except Exception:
                pass

            # Wait for grace window OR promote a matched staging slot via
            # ACTIVE_MATCH (warm-slot reuse). Per v0.2 §6 FSM; this transition
            # cascades same-(thread_id, model_tag) follow-up requests through
            # the warm llama-server without re-spawn.
            # Wired in v0.2.1 as the active-match path (was a Phase 3 stub).
            deadline = time.monotonic() + self.runtime.queue.grace_seconds
            while time.monotonic() < deadline and not self._stop_event.is_set():
                # Atomic find + remove in one lock acquire
                matched = await self.queue.pop_matched_thread(
                    slot.thread_id, slot.model_tag
                )
                if matched is not None:
                    matched.port = handle.port
                    matched.pid = handle.pid
                    self._set_active_slot(matched)
                    # State-drift guard. If matched.state
                    # drifted between find_matched_thread + here (concurrent
                    # reconcile, retry path, etc.), transition raises
                    # InvalidTransition which would crash worker_loop. Wrap
                    # the promotion + park-on-drift instead of propagate.
                    try:
                        transition(matched, SlotState.ACTIVE_MATCH)
                        await self._audit_async(matched, "active_match_promoted")
                        transition(matched, SlotState.ACTIVE)
                        # Each matched-follow-up's keep_alive overrides
                        # the anchor's for the next grace→idle calculation. Mirrors
                        # Ollama "timer resets on request receipt" rule.
                        self._set_latest_keep_alive(
                            (matched.client_meta or {}).get("keep_alive_s")
                        )
                    except InvalidTransition as drift_err:
                        log.warning(
                            "active_match state drift: slot %s in %s — terminal-park; %s",
                            matched.slot_id, matched.state.value, drift_err,
                        )
                        self._fail_completion_future(matched, drift_err)
                        await self._force_cold(
                            matched,
                            f"active_match_state_drift:{matched.state.value}",
                        )
                        self._set_active_slot(slot)
                        continue
                    matched.started_active_at = time.monotonic()
                    completed_ok = True
                    # Round 8 fix: streaming-path warm-reuse. When a streaming submit lands on an
                    # already-active matched slot, the HTTP route owns the upstream connection via
                    # stream_handle. Worker MUST NOT call _complete_fn (would open a 2nd sidecar
                    # connection and violate the single-slot invariant). Hand
                    # off via stream_ready_event and block on stream_done_event until route drains.
                    # Prior bug: this branch unconditionally called _complete_fn → matched slot's
                    # stream_ready_event was never set → route's SLOT_READY_TIMEOUT_S fired at 600s
                    # every turn ≥ 2 of a Hermes multi-tool-call agent loop.
                    matched_is_streaming = bool(
                        isinstance(matched.client_meta, dict)
                        and matched.client_meta.get("stream", False)
                    )
                    try:
                        if matched_is_streaming:
                            assert (
                                matched.stream_ready_event is not None
                                and matched.stream_done_event is not None
                            ), (
                                f"streaming slot {matched.slot_id} missing events at ACTIVE_MATCH promotion"
                            )
                            matched.stream_handle = handle
                            matched.stream_ready_event.set()
                            try:
                                await asyncio.wait_for(
                                    matched.stream_done_event.wait(),
                                    timeout=3600.0,
                                )
                            except asyncio.TimeoutError:
                                log.warning(
                                    "active_match streaming slot %s exceeded 3600s waiting for stream_done_event",
                                    matched.slot_id,
                                )
                            if matched.completion_future is not None and not matched.completion_future.done():
                                matched.completion_future.set_result({"_streamed": True})
                        else:
                            result2 = await self._complete_fn(matched, handle)
                            if (
                                matched.completion_future is not None
                                and not matched.completion_future.done()
                            ):
                                matched.completion_future.set_result(result2)
                    except asyncio.CancelledError:
                        # Cooperatively unwind route's blocking httpx call
                        # by signaling stream_done before terminal-park (avoids zombie
                        # route + dead slot drift).
                        if (
                            matched_is_streaming
                            and matched.stream_done_event is not None
                            and not matched.stream_done_event.is_set()
                        ):
                            matched.stream_done_event.set()
                        # Cancellation mid-ACTIVE_MATCH
                        # must terminal-park the matched slot so it does not
                        # rot as a zombie ACTIVE row in state.sqlite, then
                        # re-raise so worker_loop's teardown runs cleanly.
                        self._fail_completion_future(
                            matched,
                            asyncio.CancelledError("shutdown during active_match"),
                        )
                        try:
                            transition(matched, SlotState.POPPED)
                        except InvalidTransition:
                            pass
                        await self._audit_async(matched, "active_match_cancelled")
                        try:
                            _am_conn = open_state_db(
                                self.boot.storage.state_db_path
                            )
                            try:
                                mark_slot_ended(
                                    _am_conn,
                                    matched.slot_id,
                                    "active_match_cancelled",
                                )
                            finally:
                                _am_conn.close()
                        except Exception:
                            log.exception(
                                "AM-2 cleanup mark_slot_ended failed for %s",
                                matched.slot_id,
                            )
                        raise
                    except Exception as e:  # noqa: BLE001 -- per-slot isolation
                        completed_ok = False
                        self._fail_completion_future(matched, e)
                        log.exception(
                            "active_match completion failed for slot %s",
                            matched.slot_id,
                        )
                    # On completion failure, skip grace pretense
                    # — go ACTIVE → GRACE → POPPED + mark failed, keep state
                    # machine honest. (transition validates each hop.)
                    transition(matched, SlotState.GRACE)
                    if completed_ok:
                        await self._audit_async(matched, "active_match_to_grace")
                        if self.grace.restart_for_followup():
                            # Also bump per-slot extension_count
                            # (was always 0 in sqlite — only GraceTimer's was).
                            matched.extension_count = self.grace.extension_count
                            deadline = time.monotonic() + self.runtime.queue.grace_seconds
                            await self._audit_event_only_async(
                                matched.slot_id,
                                "grace_extended_via_active_match",
                                {"extension_count": self.grace.extension_count},
                            )
                    else:
                        await self._audit_async(matched, "active_match_failed")
                    # C1 fix: matched slot's request is done; its
                    # sidecar was the anchor's warm process (reused, not its own).
                    # Anchor `slot` remains the GRACE driver until grace expiry.
                    transition(matched, SlotState.POPPED)
                    await self._audit_async(matched, "active_match_completed" if completed_ok else "active_match_failed_terminal")
                    _am_conn = open_state_db(self.boot.storage.state_db_path)
                    try:
                        mark_slot_ended(
                            _am_conn,
                            matched.slot_id,
                            "active_match_completed" if completed_ok else "active_match_failed",
                        )
                    finally:
                        _am_conn.close()
                    self._set_active_slot(slot)  # anchor for teardown bookkeeping
                    continue
                await asyncio.sleep(0.05)

            # GRACE → POPPED (slot lifecycle ends here)
            transition(slot, SlotState.POPPED)
            # Hold the sidecar in idle for follow-up reuse
            # by any same-model_tag request inside idle_hot_load_seconds.
            # When idle_seconds == 0 (test default), this is equivalent to
            # immediate teardown -- preserves "grace-expired" reason on the
            # mark_slot_ended audit (backward-compat with existing tests).
            #
            # Fully-automatic idle handling:
            # honor the latest request's keep_alive intent as IDLE_HOT extension.
            # `_latest_keep_alive_s` was set on the anchor's ACTIVE and refreshed
            # on each ACTIVE_MATCH promotion — so it reflects the most recent
            # request that touched the warm slot (Ollama timer-resets-on-receipt
            # semantics). After consumption it's cleared so a stale value can't
            # leak into the next anchor cycle.
            keep_alive_s = self._latest_keep_alive_s
            default_idle = self.runtime.queue.idle_hot_load_seconds
            if keep_alive_s is None:
                idle_seconds = default_idle
            elif keep_alive_s < 0:
                # Ollama -1 = "pin until VRAM pressure"; we cap at KEEP_ALIVE_MAX_S
                # (never indefinite on single-GPU).
                idle_seconds = KEEP_ALIVE_MAX_S
            else:
                # 0 falls through this expression cleanly → idle disabled.
                idle_seconds = min(keep_alive_s, KEEP_ALIVE_MAX_S)
            ka_clamped = (
                keep_alive_s is not None
                and keep_alive_s >= 0
                and keep_alive_s > KEEP_ALIVE_MAX_S
            )
            # Consumed — clear before any further decisions so the next anchor
            # starts cleanly (defense-in-depth on top of _process_slot reset).
            self._set_latest_keep_alive(None)
            if idle_seconds > 0 and self._active_handle is not None:
                # Hand off the active handle to the manager-level idle holder.
                self._set_idle_holder(
                    self._active_handle,
                    slot.model_tag,
                    time.monotonic() + idle_seconds,
                )
                await self._audit_event_only_async(
                    slot.slot_id,
                    "idle_hot_enter",
                    {
                        "model_tag": slot.model_tag,
                        "idle_seconds": idle_seconds,
                        # Audit: visibility into when
                        # client keep_alive overrode the default + when the cap fired.
                        "keep_alive_requested": keep_alive_s,
                        "keep_alive_clamped": ka_clamped,
                    },
                )
                # Mark the slot ended at the state.sqlite layer -- the slot
                # is done; only the model stays warm. Audit reason names the
                # warm-hold so post-hoc audits can see the difference vs.
                # plain grace-expired teardown.
                _ih_conn = open_state_db(self.boot.storage.state_db_path)
                try:
                    mark_slot_ended(
                        _ih_conn, slot.slot_id, "grace-expired-held-idle"
                    )
                finally:
                    _ih_conn.close()
            else:
                # idle disabled (idle_seconds=0) or no handle -- immediate teardown.
                await self._teardown(slot, "grace-expired")
                self.idle.start(slot.model_tag)
                await self._audit_event_only_async(
                    slot.slot_id,
                    "idle_hot_enter",
                    {"model_tag": slot.model_tag},
                )
        finally:
            # Unwind fix:
            # If unwind reaches here with a live handle, the IDLE_HOT
            # entry did NOT complete (most often CancelledError during
            # shutdown or mid-_complete_fn). MUST teardown the handle,
            # not just drop the reference, or llama-server orphans with
            # the full model in VRAM and no parent reference anywhere.
            #
            # Synthesis:
            # 1. Diagnostic log at entry — surfaces leak path under repro.
            # 2. Do NOT null _active_handle until sigterm SUCCEEDS. If the
            #    sigterm helper raises (drained_sigterm internal failure,
            #    process already dead/zombie, etc.) leaving _active_handle
            #    set lets worker_loop's per-slot exception handler
            #    (line 466-481) fire its safety-net _teardown(). Otherwise
            #    the null-before-success ordering bypassed that safety net.
            handle_to_reap = self._active_handle
            self._set_active_slot(None)
            log.warning(
                "process_slot finally reached: slot=%s active_handle_pid=%s alive=%s idle_match=%s",
                getattr(slot, "slot_id", "?"),
                getattr(handle_to_reap, "pid", None) if handle_to_reap is not None else None,
                handle_to_reap.is_alive() if handle_to_reap is not None else False,
                handle_to_reap is self._idle_handle if handle_to_reap is not None else False,
            )
            # Skip defensive sigterm if the handle was promoted to the
            # IDLE_HOT holder — that promotion is by design; killing it
            # would defeat the warm-hold purpose.
            # removed is_alive() gate — TOCTOU: process could die
            # between the check and _sigterm, causing the sigterm to be
            # skipped entirely. _sigterm handles a dead process gracefully;
            # the safety-net _teardown in worker_loop catches any missed reaps.
            if (
                handle_to_reap is not None
                and handle_to_reap is not self._idle_handle
            ):
                sigterm_ok = False
                try:
                    await asyncio.shield(
                        self._sigterm(
                            handle_to_reap,
                            drained_window_s=float(
                                self.runtime.queue.drained_sigterm_window_active_s
                            ),
                            is_active=False,
                            cold_window_s=float(
                                self.runtime.queue.drained_sigterm_window_cold_s
                            ),
                        )
                    )
                    sigterm_ok = True
                except Exception:
                    log.exception(
                        "cancellation-unwind teardown FAILED — leaving "
                        "_active_handle set so worker_loop safety-net can retry"
                    )
                if sigterm_ok:
                    self._set_active_handle(None)
                # else: keep _active_handle so worker_loop's except handler
                # (which calls _teardown with reason="worker-uncaught-exception")
                # has a second chance to reap. If that ALSO fails, the
                # intra_lifetime_orphan_scan on the next /ensure tick is
                # the final safety net (singleton.py).
            else:
                # Handle absent or promoted to idle_holder — safe to null.
                self._set_active_handle(None)

    async def _teardown(self, slot: Slot, reason: str) -> None:
        """Drained SIGTERM the process group → VRAM verify → orphan reap → audit."""
        if self._active_handle is not None:
            ok, status = await self._sigterm(
                self._active_handle,
                drained_window_s=float(self.runtime.queue.drained_sigterm_window_active_s),
                is_active=False,
                cold_window_s=float(self.runtime.queue.drained_sigterm_window_cold_s),
            )
            # Dynamic expected_drop_mib derived from manifest
            # expected_vram_bytes. Was hardcoded 1024 MiB — let a 921 MiB
            # drop "verify clear" while 17 GiB qwen35b still resident.
            expected_drop_mib = self._compute_expected_drop_mib(slot.model_tag)
            await self._vram_verify(
                expected_drop_mib=expected_drop_mib, timeout_s=30.0,
            )
            # Scan for grandchild orphans left behind by
            # Tom's Fork setsid-detach (killpg never reached them) and
            # reap before the next slot needs the port + VRAM. ~50ms
            # /proc walk; cheap to run on every teardown.
            orphan_reaped = 0
            try:
                orphan_reap_result = boot_orphan_reaper(
                    port_base=self.boot.runtime.default_port_base,
                    known_pids=set(),  # single-slot mode; multi-slot
                                       # A future version will pass live sidecar pids
                )
                orphan_reaped = orphan_reap_result.get("reaped", 0)
            except Exception:
                log.exception(
                    "post-teardown orphan reap failed (best-effort)"
                )
            # Intra-lifetime port-bound reaper. Catches
            # orphans whose parent IS still the running manager (PPid !=
            # 1 so boot_orphan_reaper misses them) — e.g. handle dropped
            # without sigterm via lost reference or finally-clear bug.
            try:
                live_pids = self._live_handle_pids()
                il_result = intra_lifetime_orphan_scan(
                    port_base=self.boot.runtime.default_port_base,
                    known_handle_pids=live_pids,
                )
                if il_result.get("reaped", 0) > 0:
                    log.warning(
                        "intra-lifetime reap caught orphans post-teardown: %s",
                        il_result,
                    )
            except Exception:
                log.exception(
                    "intra-lifetime orphan scan failed (best-effort)"
                )
            # slot-write stays on state_db_session; audit-write goes
            # through the pool wrapped in asyncio.to_thread (sync-only).
            with state_db_session(self.boot.storage.state_db_path) as conn:
                mark_slot_ended(conn, slot.slot_id, reason)

            def _audit_teardown() -> None:
                with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                    record_audit_event(
                        audit_conn,
                        "teardown",
                        {
                            "reason": reason,
                            "sigterm_status": status,
                            "sigterm_ok": ok,
                            "post_teardown_orphans_reaped": orphan_reaped,
                        },
                        slot_id=slot.slot_id,
                    )

            await asyncio.to_thread(_audit_teardown)
            # Clear _active_handle after
            # successful teardown so the outer finally's defensive sigterm
            # net does not double-fire on normal flow. Owner contract:
            # "if you called _teardown you have handed off the handle."
            self._set_active_handle(None)

    async def _teardown_idle_holder(self, reason: str) -> None:
        """Tear down the manager-level idle handle.

        Called when:
        - a slot for a DIFFERENT model_tag arrives (immediate switch path), or
        - the idle timer expires in the worker_loop, or
        - shutdown.
        """
        if self._idle_handle is None:
            return
        held = self._idle_handle
        model_tag = self._idle_model_tag
        self._clear_idle_holder()
        ok, status = await self._sigterm(
            held,
            drained_window_s=float(
                self.runtime.queue.drained_sigterm_window_active_s
            ),
            is_active=False,
            cold_window_s=float(
                self.runtime.queue.drained_sigterm_window_cold_s
            ),
        )
        # Dynamic expected_drop_mib for idle holder teardown.
        expected_drop_mib = self._compute_expected_drop_mib(
            model_tag or ""
        )
        await self._vram_verify(
            expected_drop_mib=expected_drop_mib, timeout_s=30.0,
        )
        try:
            boot_orphan_reaper(
                port_base=self.boot.runtime.default_port_base,
                known_pids=set(),
            )
        except Exception:
            log.exception(
                "idle-holder orphan reap failed (best-effort)"
            )
        # Intra-lifetime port-bound reaper here too.
        try:
            live_pids = self._live_handle_pids()
            intra_lifetime_orphan_scan(
                port_base=self.boot.runtime.default_port_base,
                known_handle_pids=live_pids,
            )
        except Exception:
            log.exception(
                "intra-lifetime orphan scan failed (best-effort)"
            )
        # audit-only write via pool, wrapped to_thread (sync-only).
        def _audit_idle_holder() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                record_audit_event(
                    audit_conn,
                    "teardown_idle_holder",
                    {
                        "reason": reason,
                        "model_tag": model_tag,
                        "sigterm_status": status,
                        "sigterm_ok": ok,
                    },
                )

        await asyncio.to_thread(_audit_idle_holder)

    async def _force_cold(self, slot: Slot, reason: str) -> None:
        """Mark a slot COLD when processing dies mid-flight.

        Walk legal transitions to COLD from any non-terminal
        state instead of silent direct-mutation. fsm.py LEGAL_TRANSITIONS now
        carries STAGED→COLD, LOADING→COLD, LOADING_FAIL→POPPED, POPPED→COLD,
        ACTIVE→GRACE→POPPED→COLD, IDLE_HOT→COLD. Memory and DB state stay
        in sync (no drift where slot.state stays e.g. LOADING in Python
        while sqlite reads state='COLD').

        Defensive teardown of any live handle attached
        to this slot BEFORE forcing COLD. Closes the footgun where a
        caller forgot to teardown (AM-2 active_match_cancelled path
        raises after _force_cold(matched, ...) without sigterm'ing the
        anchor sidecar). Defense-in-depth.

        Annotate audit with pid_source so post-hoc
        tooling can distinguish matched-row-on-anchor-pid (anchor_shared)
        from genuine standalone (self).
        """
        # Identify pid_source, defensive sigterm
        # if the slot owns its own live handle.
        pid_source = "self"
        if (
            self._active_handle is not None
            and slot.pid
            and slot.pid == self._active_handle.pid
            and len(self._inflight) > 1
        ):
            # Design #1 must-fix: the sidecar is serving MULTIPLE concurrent
            # fan-out riders. Force-colding ONE of them (the ANCHOR or a rider)
            # must NEVER sigterm the shared handle while siblings are still
            # streaming. Identity-INDEPENDENT: gate on shared-pid + rider-set
            # size, NOT slot_id (the anchor IS _active_slot, so the slot_id
            # check below would let it fall through to a teardown). Belt-and-
            # suspenders with the admit-path pid=None reset on rider drift.
            pid_source = "fanout_shared_no_teardown"
        elif (
            self._active_slot is not None
            and slot.slot_id != self._active_slot.slot_id
            and self._active_handle is not None
            and slot.pid
            and slot.pid == self._active_handle.pid
        ):
            # matched.pid == anchor.pid via shared warm sidecar
            # (AM-2 drift path). Do NOT teardown — anchor owns it.
            pid_source = "anchor_shared"
        elif (
            slot.pid
            and self._active_handle is not None
            and slot.pid == self._active_handle.pid
        ):
            # Slot owns the active handle. Defensive teardown.
            try:
                await self._sigterm(
                    self._active_handle,
                    drained_window_s=float(
                        self.runtime.queue.drained_sigterm_window_active_s
                    ),
                    is_active=False,
                    cold_window_s=float(
                        self.runtime.queue.drained_sigterm_window_cold_s
                    ),
                )
                self._set_active_handle(None)
            except Exception:
                log.exception(
                    "_force_cold defensive teardown failed (best-effort)"
                )

        # Walk legal hops to COLD per the new FSM table.
        # Worst case: ACTIVE → GRACE → POPPED → COLD (3 hops).
        if not is_terminal(slot.state):
            for _ in range(4):  # bounded — FSM diameter to COLD is 3
                if slot.state == SlotState.COLD:
                    break
                legal = LEGAL_TRANSITIONS.get(slot.state, set())
                if SlotState.COLD in legal:
                    transition(slot, SlotState.COLD)
                    break
                # Step toward COLD via the cheapest-distance hop.
                if SlotState.POPPED in legal:
                    transition(slot, SlotState.POPPED)
                elif SlotState.GRACE in legal:
                    transition(slot, SlotState.GRACE)
                elif SlotState.LOADING_FAIL in legal:
                    transition(slot, SlotState.LOADING_FAIL)
                else:
                    # No legal hop — terminal-park as COLD directly only as
                    # absolute last resort. Log for diagnostics.
                    log.warning(
                        "_force_cold: no legal hop from %s — direct-set COLD",
                        slot.state.value,
                    )
                    slot.state = SlotState.COLD
                    break
        # slot-write stays on state_db_session; audit-write via pool.
        with state_db_session(self.boot.storage.state_db_path) as conn:
            mark_slot_ended(conn, slot.slot_id, reason)

        def _audit_force_cold() -> None:
            with audit_db_session(self.boot.storage.state_db_path) as audit_conn:
                # pid_source annotation
                record_audit_event(
                    audit_conn,
                    "force_cold",
                    {"reason": reason, "pid_source": pid_source},
                    slot_id=slot.slot_id,
                )

        await asyncio.to_thread(_audit_force_cold)

    def _compute_expected_drop_mib(self, model_tag: str) -> int:
        """Dynamic expected_drop_mib derived from manifest.

        Returns max(2048, manifest.expected_vram_bytes/1024**2). Floor
        at 2 GiB absorbs page-cache noise on small models. If manifest
        cannot be read (race during teardown of a just-deleted manifest),
        falls back to 2048 MiB rather than blocking teardown.
        """
        try:
            m = read_manifest(self.boot.storage.manifests_path, model_tag)
            return max(2048, int(m.expected_vram_bytes / (1024 * 1024)))
        except (FileNotFoundError, OSError):
            log.warning(
                "_compute_expected_drop_mib: manifest read failed for %s, "
                "falling back to 2048 MiB",
                model_tag,
            )
            return 2048

    def _live_handle_pids(self) -> set[int]:
        """Set of currently-known live llama-server pids across ALL residents.

        Used by intra_lifetime_orphan_scan + boot_orphan_reaper to tell managed
        sidecars from leaked orphans on our port range. UNION over
        EVERY resident's live handle pid + idle_handle pid + each reserving resident's
        ``booting_pid`` (the spawned-but-handle-not-yet-set pid), read from an
        AWAIT-FREE ``list()`` snapshot of the registry so a concurrent dispatcher
        mutation can't 'dict changed size during iteration'. At cap==1 the lone
        resident mirrors ``_active_handle``/``_idle_handle`` so this equals the legacy
        {_active,_idle} set (byte-identical); at cap>=2 the union-ALL is the critical
        fix that stops a co-resident sidecar being reaped as an orphan during another
        resident's teardown, and ``booting_pid`` closes the still-booting-sibling
        window before its handle is set.
        """
        live: set[int] = set()
        # cap-gate the multi-resident union so at cap<=1 this function is LITERALLY
        # the legacy two-if body below (structural byte-identity, not argued — the
        # entire dispatcher/registry surface is unreachable at cap=1, and the
        # _residents singleton's booting_pid is never written on that path).
        if self.runtime.queue.max_parallel_sidecars >= 2:
            for r in list(self._residents.values()):  # await-free atomic snapshot
                h = r.handle
                if h is not None and h.is_alive():
                    live.add(h.pid)
                ih = r.idle_handle
                if ih is not None and ih.is_alive():
                    live.add(ih.pid)
                if r.booting_pid is not None:
                    live.add(r.booting_pid)
        # Legacy manager-global holders (primary mirror) — belt-and-suspenders so
        # cap==1 stays byte-identical even if a singleton mirror ever lagged.
        # read handle into local var first to avoid TOCTOU between
        # is_alive() check and .pid read (handle could change between reads).
        ah = self._active_handle
        if ah is not None and ah.is_alive():
            live.add(ah.pid)
        ih = self._idle_handle
        if ih is not None and ih.is_alive():
            live.add(ih.pid)
        return live

    async def _audit_async(self, slot: Slot, event_type: str) -> None:
        """Async-context wrapper for `_audit`.

        Note: audit_db_session is sync-only. When called from an
        async function (e.g. `_process_slot`), the sync `_audit` must be
        offloaded to a worker thread or the sync-only guard raises RuntimeError.
        Call this from async code; call `_audit` directly from sync code.
        """
        await asyncio.to_thread(self._audit, slot, event_type)

    async def _audit_event_only_async(
        self,
        slot_id: str,
        event_type: str,
        payload: dict | None = None,
    ) -> None:
        """Async-context wrapper for `_audit_event_only`. See `_audit_async`."""
        await asyncio.to_thread(
            self._audit_event_only, slot_id, event_type, payload
        )

    def _audit(self, slot: Slot, event_type: str) -> None:
        """Audit: upsert current slot row + record event + publish to event bus.

        Note: slot-write stays on state_db_session; audit-write via pool. Both
        calls are sync (this method is `def`, not `async def`), so the
        audit_db_session sync-only guard is satisfied without to_thread.

        Note: do NOT call this directly from async code — use
        `_audit_async` instead (the sync-only guard catches direct calls from a
        running event loop).
        """
        with state_db_session(self.boot.storage.state_db_path) as conn:
            upsert_slot(
                conn,
                {
                    "slot_id": slot.slot_id,
                    "model_tag": slot.model_tag,
                    "thread_id": slot.thread_id,
                    "state": slot.state.value,
                    "port": slot.port,
                    "pid": slot.pid,
                    "extension_count": slot.extension_count,
                    "client_meta": slot.client_meta,
                },
            )
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(conn, event_type, {"state": slot.state.value}, slot_id=slot.slot_id)
        # Publish redacted event to WS subscribers (v0.2 §11.1)
        self.event_bus.publish_nowait(
            {
                "event": event_type,
                "slot_id": slot.slot_id,
                "model_tag": slot.model_tag,
                "state": slot.state.value,
                # Redaction: only first 8 chars of thread_id exposed
                "thread_id_prefix": (slot.thread_id or "")[:8],
            }
        )

    def _audit_event_only(self, slot_id: str, event_type: str, payload: dict | None = None) -> None:
        """Audit: record event ONLY, no slot row mutation.

        Use after teardown when the slot is already COLD in DB and we don't want
        to clobber that state. Note: routed through audit pool (sync call).
        """
        with audit_db_session(self.boot.storage.state_db_path) as conn:
            record_audit_event(conn, event_type, payload or {}, slot_id=slot_id)

    # ===  background sweeper ==============================

    async def _periodic_terminal_park_sweep(self) -> None:
        """Periodically finalize the state-row for the disconnect-eviction path evictions.

        Deferred state-row finalization OFF the
        worker_loop hot path to avoid the 1-3s SQLite fsync stall that
        bypassed the Note audit pool. Audit-emit fires immediately via
        ``_audit_event_only_async``; the state-row finalize lands here.

        Loop: every ``background_sweep_interval_s`` (default 60s — matches
        Note audit pool rhythm), the sweeper queries for slots that are:
          - in STAGED state (never reached ACTIVE)
          - older than ``background_sweep_min_age_s`` (default 24h)
          - have no live ``pid`` (no sidecar attached)
          - have ``ended_at IS NULL`` (not already terminal-parked)
        Each match is mark_slot_ended'd with reason
        ``background_sweeper_evicted``. mark_slot_ended sets state=COLD +
        ended_at=now, so the SELECT predicate becomes false on the next
        sweep — naturally idempotent.

        Triple-gate (state=STAGED + age + pid IS NULL) eliminates false
        positives. An in-flight slot has either a non-NULL pid or has
        already transitioned past STAGED; either disqualifies it from the
        sweep. The 24h staleness floor is defense-in-depth — even if a
        pid-check edge case slips through, a 24h+ STAGED-state slot is
        overwhelmingly likely to be an the disconnect-eviction path eviction casualty.
        """
        interval = max(1, int(self.runtime.queue.background_sweep_interval_s))
        min_age = max(60, int(self.runtime.queue.background_sweep_min_age_s))
        log.info(
            "periodic_terminal_park_sweep started (interval=%ds, min_age=%ds)",
            interval, min_age,
        )
        while not self._stop_event.is_set():
            try:
                await self._run_one_sweep(min_age)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "periodic_terminal_park_sweep iteration failed (best-effort)"
                )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval,
                )
                # _stop_event fired during sleep — exit cleanly
                break
            except asyncio.TimeoutError:
                continue  # normal cadence tick — run next sweep
        log.info("periodic_terminal_park_sweep exited")

    async def _run_one_sweep(self, min_age_s: int) -> int:
        """Run one sweep iteration. Returns count of slots finalized.

        Synchronous SQL inside ``state_db_session`` — OK because this method
        runs on the background sweeper task, NOT the worker_loop hot path.
        """
        from datetime import timedelta
        cutoff_iso = (
            datetime.now(timezone.utc) - timedelta(seconds=min_age_s)
        ).isoformat(timespec="seconds")
        finalized: list[str] = []
        with state_db_session(self.boot.storage.state_db_path) as conn:
            cur = conn.execute(
                """SELECT slot_id FROM slots
                   WHERE state = 'STAGED'
                     AND created_at < ?
                     AND pid IS NULL
                     AND ended_at IS NULL
                   LIMIT 100""",  # batch cap, bounds writer-lock hold time under storm pattern
                (cutoff_iso,),
            )
            stale_slot_ids = [row["slot_id"] for row in cur.fetchall()]
            for slot_id in stale_slot_ids:
                mark_slot_ended(conn, slot_id, "background_sweeper_evicted")
                finalized.append(slot_id)
        # Counters + /status surfacing (single per-run update, not per-row)
        self._slots_finalized_lifetime += len(finalized)
        self._last_sweep_iso = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
        # Single audit per sweep run keeps audit volume bounded under storm.
        try:
            await self._audit_event_only_async(
                None,
                "background_sweeper_run",
                {
                    "slots_finalized": len(finalized),
                    "sweep_ts": self._last_sweep_iso,
                },
            )
        except Exception:
            log.exception(
                "background_sweeper_run audit emit failed (best-effort)"
            )
        return len(finalized)

    # === Shutdown ============================================================

    async def _drain_bg_tasks(self, grace_s: float = 10.0) -> None:
        """Await the detached ``_spawn_bg`` teardown/requeue/reap tasks at shutdown so
        an in-flight ``_evict_teardown`` (sidecar reap) can't leak past process exit
        (fast-follow). Snapshot first (the set self-mutates as tasks finish);
        bounded by ``grace_s``; ``gather(return_exceptions=True)`` so one failing task
        can't abort the drain. A timeout cancels the stragglers (acceptable — they are
        best-effort cleanup and the process is exiting).

        NOTE on timing: the SLOW reap (``_reap_booting_pid``'s up-to-3s SIGTERM-poll +
        SIGKILL) runs INLINE in the driver finally, which the sweep already awaits
        with NO timeout in the DRIVERS step BEFORE this drain — and that finally claims
        ``torn_down`` first, so any bg ``_evict_teardown`` here finds it already claimed
        and no-ops. So this drain only awaits the FAST bg tasks (future-fails, requeues,
        no-op teardowns); the generous 10s grace is pure insurance."""
        pending = [t for t in list(self._bg_tasks) if not t.done()]
        if not pending:
            return
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=grace_s
            )

    async def shutdown(self) -> None:
        """Clean tear-down. Stops worker loop + sweeper + drains queue + closes state db."""
        self._stop_event.set()
        # === shutdown-sweep: OBSERVERS -> dispatcher -> DRIVERS -> bg-drain ===
        # 1) OBSERVERS FIRST (both the cap<=1 single poller AND the cap>=2 resident
        # supervisor) so neither issues a /slots probe against a sidecar being torn
        # down, nor publishes a generation tick after the loop is gone (the Failure
        # Predictor #16 HIGH). Cancelling the supervisor stops ALL N per-resident
        # polls at once.
        for obs in (self._live_poller_task, self._live_supervisor_task):
            if obs is not None and not obs.done():
                obs.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await obs
        # 2) DISPATCHER next: stop routing new work onto residents.
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        # 3) DRIVERS: deterministically cancel every per-resident driver so its
        # finally reaps the handle / booting_pid + drains its inbox (the cap>=2
        # teardown). Earlier the drivers were left to NOTICE _stop_event on their next loop turn;
        # the sweep cancels them so a driver parked in inbox.get()/health-wait tears
        # down NOW (and before queue.close, so drained inbox slots can re-enqueue).
        driver_tasks = [
            r.driver_task
            for r in list(self._residents.values())
            if r.driver_task is not None and not r.driver_task.done()
        ]
        for dt in driver_tasks:
            dt.cancel()
        if driver_tasks:
            # Await the driver teardowns in PARALLEL (gather), not sequentially: each
            # finally may run an up-to-3s SIGTERM-grace reap, so a sequential
            # `for dt: await dt` would serialize N residents to ~N*3s at shutdown.
            # return_exceptions so one failing teardown can't abort the rest (the
            # CancelledError each cancelled driver raises is captured, not propagated)
            # (PL pre-cutover polish #3).
            await asyncio.gather(*driver_tasks, return_exceptions=True)
        # 4) DRAIN the detached bg-tasks (fast-follow): the driver finallys +
        # death-reapers schedule teardown/requeue/reap coroutines through _spawn_bg;
        # await them (bounded) so an in-flight _evict_teardown can't LEAK a sidecar
        # past process exit. Must come AFTER the driver awaits (which is when those
        # reapers get scheduled).
        await self._drain_bg_tasks()
        #  cancel + await the background sweeper task.
        if self._sweeper_task is not None and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._sweeper_task
        # NEMO V2 2.1 fix: close() now returns the slots it cleared; fail
        # their completion_futures so callers get a clean CancelledError
        # instead of hanging until submit_and_wait timeout.
        cleared_slots = await self.queue.close()
        for cleared in cleared_slots:
            self._fail_completion_future(
                cleared,
                asyncio.CancelledError(
                    "manager shutdown -- slot was never processed"
                ),
            )
        # Tear down any idle holder so VRAM is released
        # and llama-server child is reaped on graceful shutdown.
        if self._idle_handle is not None:
            try:
                await self._teardown_idle_holder("shutdown")
            except Exception:
                log.exception(
                    "idle teardown during shutdown failed (best-effort)"
                )
        # Release the TOCTOU-pinned binary fd on shutdown
        if self._binary_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._binary_fd)
            self._binary_fd = None
