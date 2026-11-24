//! `XCL_WAIT_TRACE=1`: what the title waits for when it stops issuing work.
//!
//! With the frame wait budget (`XCL_PSO_FRAME_WAIT_MS`) a retail title presented a few frames of
//! its world and then issued no work for minutes, with none of its threads in a pipeline wait.
//! This trace writes three kinds of lines (all start with `wait-trace:`):
//!
//! * per submitted list that had draws or dispatches skipped: the queue, the submit and frame
//!   number, and per pipeline the skip reason, its key and the root descriptor / table addresses
//!   bound at the skip (the console root signature does not say which are UAVs);
//! * a wait still unfinished after `STUCK_AFTER` – a `Wait32/64BitValueX` polled on the CPU, a
//!   `Wait64BitValueX` bridged to a GPU `queue.Wait`, a title fence whose completed value lags
//!   what was signalled or waited for, a title `SetEventOnCompletion` – once when it gets stuck,
//!   once when it moves again;
//! * while no list was submitted for `IDLE_AFTER`, every `IDLE_REPEAT` the list of stuck waits
//!   and every lagging title fence.
//!
//! Off (unset or `0`) it costs one cached load at each hook.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

use super::console_list::{Layout, Param};

const VAR: &str = "XCL_WAIT_TRACE";
/// a wait unfinished for this long is reported
pub const STUCK_AFTER: Duration = Duration::from_millis(2000);
/// no submit for this long: the title is idle
const IDLE_AFTER: Duration = Duration::from_secs(2);
/// how often the idle list is written while the title stays idle
const IDLE_REPEAT: Duration = Duration::from_secs(2);
/// how often the reporter looks
const TICK: Duration = Duration::from_millis(250);
/// per-list skip lines in all; past it only the totals every `SKIP_TOTALS_PERIOD`
const SKIP_LINES: u64 = 256;
const SKIP_TOTALS_PERIOD: Duration = Duration::from_secs(2);
const PIPELINES_PER_LINE: usize = 6;
const ROOTS_PER_PIPELINE: usize = 8;
/// stuck waits listed in one idle report (the rest are counted)
const STUCK_PER_IDLE: usize = 48;
/// entries kept for lists recorded but not (yet) submitted, and for event waits
const MAX_TRACKED: usize = 4096;

/// Whether the trace is on (`XCL_WAIT_TRACE`); the first call starts the reporter.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = match parse(std::env::var(VAR).ok().as_deref()) {
            Ok(on) => on,
            Err(e) => {
                crate::xlog_error!("D3D12.X: {e}");
                panic!("D3D12.X: {e}");
            }
        };
        if on {
            crate::xlog_info!(
                "wait-trace: on: skipped draws/dispatches per submitted list, waits unfinished after {} ms, the stuck waits every {} s while nothing is submitted for {} s",
                STUCK_AFTER.as_millis(),
                IDLE_REPEAT.as_secs(),
                IDLE_AFTER.as_secs()
            );
            if let Err(e) = std::thread::Builder::new().name("xcl-wait-trace".into()).spawn(reporter) {
                crate::xlog_error!("wait-trace: no reporter thread: {e}");
            }
        }
        on
    })
}

fn parse(value: Option<&str>) -> Result<bool, String> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{VAR}={other:?} must be 0 or 1")),
    }
}

fn start() -> Instant {
    static T: OnceLock<Instant> = OnceLock::new();
    *T.get_or_init(Instant::now)
}

static SUBMITS: AtomicU64 = AtomicU64::new(0);
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// The title presented a frame.
pub fn presented() {
    if on() {
        FRAMES.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------------------
// skipped draws and dispatches
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipWhy {
    /// the frame's pipeline wait budget ran out before the pipeline was built
    Budget,
    /// the pipeline was still being built (no budget involved)
    Building,
    /// the pipeline has no translated console shader
    Untranslated,
}

impl SkipWhy {
    fn tag(self) -> &'static str {
        match self {
            SkipWhy::Budget => "budget",
            SkipWhy::Building => "building",
            SkipWhy::Untranslated => "untranslated",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipKind {
    Draw,
    Dispatch,
}

/// A root argument that is an address: a root CBV/SRV/UAV (64-bit) or a descriptor table (32-bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Root {
    Descriptor(u64),
    Table(u32),
}

impl std::fmt::Display for Root {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Root::Descriptor(va) => write!(f, "d:{va:#x}"),
            Root::Table(at) => write!(f, "t:{at:#x}"),
        }
    }
}

/// The address-valued root arguments of `args` packed by `layout`.
pub fn roots(layout: Option<&Layout>, args: &[u8]) -> Vec<Root> {
    let Some(layout) = layout else { return Vec::new() };
    let word = |at: u32, n: usize| args.get(at as usize..at as usize + n);
    layout
        .params
        .iter()
        .filter_map(|p| match *p {
            Param::Descriptor { at } => word(at, 8).map(|b| Root::Descriptor(u64::from_le_bytes(b.try_into().unwrap()))),
            Param::Table { at } => word(at, 4).map(|b| Root::Table(u32::from_le_bytes(b.try_into().unwrap()))),
            Param::Constants { .. } => None,
        })
        .take(ROOTS_PER_PIPELINE)
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct PipelineSkips {
    pso: usize,
    key: u64,
    why: SkipWhy,
    count: u64,
    roots: Vec<Root>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ListSkips {
    draws: u64,
    dispatches: u64,
    pipelines: Vec<PipelineSkips>,
    /// skips of pipelines past `PIPELINES_PER_LINE` distinct ones
    more: u64,
}

impl ListSkips {
    fn add(&mut self, kind: SkipKind, why: SkipWhy, pso: usize, key: u64, roots: Vec<Root>) {
        match kind {
            SkipKind::Draw => self.draws += 1,
            SkipKind::Dispatch => self.dispatches += 1,
        }
        if let Some(p) = self.pipelines.iter_mut().find(|p| p.pso == pso && p.why == why) {
            p.count += 1;
            for r in roots {
                if p.roots.len() < ROOTS_PER_PIPELINE && !p.roots.contains(&r) {
                    p.roots.push(r);
                }
            }
        } else if self.pipelines.len() < PIPELINES_PER_LINE {
            self.pipelines.push(PipelineSkips { pso, key, why, count: 1, roots });
        } else {
            self.more += 1;
        }
    }

    fn merge(&mut self, other: ListSkips) {
        self.draws += other.draws;
        self.dispatches += other.dispatches;
        self.more += other.more;
        for p in other.pipelines {
            if let Some(q) = self.pipelines.iter_mut().find(|q| q.pso == p.pso && q.why == p.why) {
                q.count += p.count;
            } else if self.pipelines.len() < PIPELINES_PER_LINE {
                self.pipelines.push(p);
            } else {
                self.more += p.count;
            }
        }
    }

    fn describe(&self) -> String {
        let mut parts: Vec<String> = self
            .pipelines
            .iter()
            .map(|p| {
                let roots: Vec<String> = p.roots.iter().map(ToString::to_string).collect();
                format!("pso {:#x} key {:016x} x{} [{}] roots [{}]", p.pso, p.key, p.count, p.why.tag(), roots.join(" "))
            })
            .collect();
        if self.more > 0 {
            parts.push(format!("+{} skips of other pipelines", self.more));
        }
        parts.join("; ")
    }
}

static SKIPS: Mutex<Option<HashMap<usize, ListSkips>>> = Mutex::new(None);
static SKIP_LINES_WRITTEN: AtomicU64 = AtomicU64::new(0);
static SKIPPED_DRAWS: AtomicU64 = AtomicU64::new(0);
static SKIPPED_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SKIPPED_LISTS: AtomicU64 = AtomicU64::new(0);

/// A draw or dispatch recorded on `list` was skipped (call only when `on()`).
pub fn note_skip(list: *mut c_void, kind: SkipKind, why: SkipWhy, pso: usize, roots: Vec<Root>) {
    let key = super::pso_async::trace_key(pso);
    match kind {
        SkipKind::Draw => SKIPPED_DRAWS.fetch_add(1, Ordering::Relaxed),
        SkipKind::Dispatch => SKIPPED_DISPATCHES.fetch_add(1, Ordering::Relaxed),
    };
    let mut g = SKIPS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if m.len() >= MAX_TRACKED && !m.contains_key(&(list as usize)) {
        m.clear();
    }
    m.entry(list as usize).or_default().add(kind, why, pso, key, roots);
}

/// `titles` (mapped to the PC lists `mapped`) went to `queue` (call only when `on()`).
pub unsafe fn submitted(queue: *mut c_void, titles: &[*mut c_void], mapped: &[*mut c_void]) {
    let submit = SUBMITS.fetch_add(1, Ordering::Relaxed) + 1;
    LAST_SUBMIT_MS.store(start().elapsed().as_millis() as u64, Ordering::Relaxed);
    let mut lists: Vec<usize> = titles.iter().chain(mapped.iter()).map(|&l| l as usize).collect();
    lists.sort_unstable();
    lists.dedup();
    let mut found = ListSkips::default();
    {
        let mut g = SKIPS.lock().unwrap();
        let Some(m) = g.as_mut() else { return };
        for l in &lists {
            if let Some(s) = m.remove(l) {
                found.merge(s);
            }
        }
    }
    if found.draws == 0 && found.dispatches == 0 {
        return;
    }
    SKIPPED_LISTS.fetch_add(1, Ordering::Relaxed);
    if SKIP_LINES_WRITTEN.fetch_add(1, Ordering::Relaxed) >= SKIP_LINES {
        return;
    }
    let (writes, waits) = mapped.iter().chain(titles.iter()).fold((0, 0), |acc, &l| {
        let (w, k) = super::list_x::sync_counts(l);
        (acc.0 + w, acc.1 + k)
    });
    let (q, ty) = super::list_x::queue_label(queue);
    crate::xlog_info!(
        "wait-trace: skipped on q{q}/ty{ty} submit #{submit} frame {}: {} draws, {} dispatches in {} lists (Write64 signals {writes}, memory waits {waits} on them): {}",
        FRAMES.load(Ordering::Relaxed),
        found.draws,
        found.dispatches,
        titles.len(),
        found.describe()
    );
}

// ---------------------------------------------------------------------------------------
// waits
// ---------------------------------------------------------------------------------------

static LAST_SUBMIT_MS: AtomicU64 = AtomicU64::new(0);

/// A wait polled on the CPU, registered while it lasts.
struct MemoryWait {
    queue: usize,
    queue_ty: i32,
    va: u64,
    width: u8,
    comparison: u32,
    want: u64,
    mask: u64,
    since: Instant,
}

/// A title `SetEventOnCompletion` not yet reached.
struct EventWait {
    fence: ID3D12Fence,
    value: u64,
    event: usize,
    thread: u32,
    since: Instant,
}

// SAFETY: `ID3D12Fence` is free-threaded.
unsafe impl Send for EventWait {}

#[derive(Default)]
struct Registry {
    next: u64,
    memory: HashMap<u64, MemoryWait>,
    events: Vec<EventWait>,
    /// title fence → (queue of its last Signal, queue of its last Wait)
    fence_queues: HashMap<usize, (Option<usize>, Option<usize>)>,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn registry<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    f(REGISTRY.lock().unwrap().get_or_insert_with(Registry::default))
}

/// A wait on memory polled on the CPU begins (call only when `on()`); `memory_wait_end` with
/// the token when it ends.
pub fn memory_wait_begin(queue: usize, queue_ty: i32, va: u64, width: u8, comparison: u32, want: u64, mask: u64) -> u64 {
    registry(|r| {
        r.next += 1;
        r.memory.insert(r.next, MemoryWait { queue, queue_ty, va, width, comparison, want, mask, since: Instant::now() });
        r.next
    })
}

pub fn memory_wait_end(token: u64, satisfied: bool) {
    let Some(w) = registry(|r| r.memory.remove(&token)) else { return };
    let waited = w.since.elapsed();
    if waited >= STUCK_AFTER {
        crate::xlog_info!(
            "wait-trace: memory wait at {:#x} on q{} ended after {} ms ({})",
            w.va,
            super::list_x::queue_label_id(w.queue),
            waited.as_millis(),
            if satisfied { "satisfied" } else { "timed out, went on" }
        );
    }
}

/// The title asked `fence` to set `event` at `value` (call only when `on()`).
pub unsafe fn note_set_event(fence: *mut c_void, value: u64, event: usize) {
    let Some(f) = ID3D12Fence::from_raw_borrowed(&fence) else { return };
    if f.GetCompletedValue() >= value {
        return;
    }
    let thread = windows_sys::Win32::System::Threading::GetCurrentThreadId();
    let fence = f.clone();
    registry(|r| {
        if r.events.len() >= MAX_TRACKED {
            r.events.remove(0);
        }
        r.events.push(EventWait { fence, value, event, thread, since: Instant::now() });
    });
}

/// A title queue `Signal` (`signal`) or `Wait` on title fence `fence` (call only when `on()`).
pub fn note_title_fence_op(queue: usize, fence: usize, signal: bool) {
    registry(|r| {
        let e = r.fence_queues.entry(fence).or_default();
        if signal {
            e.0 = Some(queue);
        } else {
            e.1 = Some(queue);
        }
    });
}

unsafe fn read_memory(va: u64, width: u8) -> Option<u64> {
    let addr = va as usize;
    if !crate::gpu::shader::readable(addr, width as usize) {
        return None;
    }
    match width {
        4 => Some((*(addr as *const std::sync::atomic::AtomicU32)).load(Ordering::Acquire) as u64),
        8 => Some((*(addr as *const AtomicU64)).load(Ordering::Acquire)),
        _ => None,
    }
}

fn shown(v: Option<u64>) -> String {
    v.map_or_else(|| "unreadable".into(), |v| format!("{v:#x}"))
}

/// One wait the reporter sees. `progress` is what moves while the wait advances (a fence's
/// completed value); `since` is when the wait began, if known – otherwise it is stuck from when
/// `progress` last changed.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Observed {
    key: String,
    progress: u64,
    since: Option<Instant>,
    describe: String,
}

/// Which waits are stuck, from one look to the next (no GPU or title state: unit-tested).
#[derive(Default)]
struct Tracker {
    /// key → (progress, since it had that progress)
    seen: HashMap<String, (u64, Instant)>,
    reported: HashSet<String>,
    idle_written: Option<Instant>,
}

struct Look {
    newly_stuck: Vec<(Duration, String)>,
    moved_on: Vec<String>,
    stuck: Vec<(Duration, String)>,
}

impl Tracker {
    fn look(&mut self, now: Instant, observed: Vec<Observed>) -> Look {
        let mut keys = HashSet::new();
        let mut look = Look { newly_stuck: Vec::new(), moved_on: Vec::new(), stuck: Vec::new() };
        for o in observed {
            keys.insert(o.key.clone());
            let since = match o.since {
                Some(s) => {
                    self.seen.insert(o.key.clone(), (o.progress, s));
                    s
                }
                None => {
                    let e = self.seen.entry(o.key.clone()).or_insert((o.progress, now));
                    if e.0 != o.progress {
                        *e = (o.progress, now);
                        if self.reported.remove(&o.key) {
                            look.moved_on.push(o.describe.clone());
                        }
                    }
                    e.1
                }
            };
            let age = now.saturating_duration_since(since);
            if age >= STUCK_AFTER {
                if self.reported.insert(o.key.clone()) {
                    look.newly_stuck.push((age, o.describe.clone()));
                }
                look.stuck.push((age, o.describe));
            }
        }
        self.seen.retain(|k, _| keys.contains(k));
        let gone: Vec<String> = self.reported.iter().filter(|k| !keys.contains(*k)).cloned().collect();
        for k in gone {
            self.reported.remove(&k);
            look.moved_on.push(k);
        }
        look.stuck.sort_by(|a, b| b.0.cmp(&a.0));
        look
    }

    /// Whether the idle list is due: nothing submitted for `idle`, and none written for `IDLE_REPEAT`.
    fn idle_due(&mut self, now: Instant, idle: Duration) -> bool {
        if idle < IDLE_AFTER {
            self.idle_written = None;
            return false;
        }
        if self.idle_written.is_some_and(|t| now.saturating_duration_since(t) < IDLE_REPEAT) {
            return false;
        }
        self.idle_written = Some(now);
        true
    }
}

/// Every wait the reporter can see now, and the lagging title fences (for the idle list).
unsafe fn observe() -> (Vec<Observed>, Vec<String>) {
    let mut out = Vec::new();
    let (memory, events, fence_queues) = registry(|r| {
        let memory: Vec<(u64, u64, i32, u64, u8, u32, u64, u64, Instant)> =
            r.memory.iter().map(|(&t, w)| (t, w.queue as u64, w.queue_ty, w.va, w.width, w.comparison, w.want, w.mask, w.since)).collect();
        r.events.retain(|e| e.fence.GetCompletedValue() < e.value);
        let events: Vec<(ID3D12Fence, u64, usize, u32, Instant)> = r.events.iter().map(|e| (e.fence.clone(), e.value, e.event, e.thread, e.since)).collect();
        (memory, events, r.fence_queues.clone())
    });
    for (token, queue, ty, va, width, op, want, mask, since) in memory {
        out.push(Observed {
            key: format!("memory {token}"),
            progress: 0,
            since: Some(since),
            describe: format!(
                "Wait{}BitValueX polled on the CPU for q{}/ty{ty}: memory at {va:#x} op {op} want {want:#x} mask {mask:#x}; holds {}",
                width as u32 * 8,
                super::list_x::queue_label_id(queue as usize),
                shown(read_memory(va, width))
            ),
        });
    }
    for (fence, value, event, thread, since) in events {
        let raw = fence.as_raw() as usize;
        out.push(Observed {
            key: format!("event {raw:#x} {value:#x} {event:#x}"),
            progress: 0,
            since: Some(since),
            describe: format!(
                "title SetEventOnCompletion: fence {raw:#x} value {value:#x} event {event:#x} (thread {thread}); fence at {:#x}",
                fence.GetCompletedValue()
            ),
        });
    }
    for a in super::list_x::armed_waits_snapshot() {
        out.push(Observed {
            key: format!("cell {:#x} {:#x} q{}", a.va, a.value, a.waiter),
            progress: 0,
            since: Some(a.armed_at),
            describe: format!(
                "Wait64BitValueX as GPU queue.Wait on q{}/ty{}: cell {:#x} >= {:#x}; cell fence at {:#x}, recorded {:#x}, submitted {:#x} by {}; memory holds {}",
                a.waiter,
                a.waiter_ty,
                a.va,
                a.value,
                a.completed,
                a.recorded,
                a.submitted,
                a.producer.map_or_else(|| "the CPU".into(), |q| format!("q{q}")),
                shown(read_memory(a.va, 8))
            ),
        });
    }
    let mut lagging = Vec::new();
    for f in super::fence_x::snapshot() {
        if f.done >= f.signalled && f.done >= f.waited {
            continue;
        }
        let (sq, wq) = fence_queues.get(&f.fence).copied().unwrap_or((None, None));
        let q = |q: Option<usize>| q.map_or_else(|| "-".into(), |q| format!("q{}", super::list_x::queue_label_id(q)));
        let describe = format!(
            "title fence {:#x}: completed {:#x}, signalled {:#x} (last Signal on {}), waited {:#x} (last Wait on {}), completed-value cell {}",
            f.fence,
            f.done,
            f.signalled,
            q(sq),
            f.waited,
            q(wq),
            if f.cell { "read by the title" } else { "not asked for" }
        );
        lagging.push(describe.clone());
        out.push(Observed { key: format!("fence {:#x}", f.fence), progress: f.done, since: None, describe });
    }
    (out, lagging)
}

fn reporter() {
    let mut tracker = Tracker::default();
    let mut totals_at = Instant::now();
    let mut totals_last = (0, 0, 0);
    loop {
        std::thread::sleep(TICK);
        let now = Instant::now();
        let (observed, lagging) = unsafe { observe() };
        let look = tracker.look(now, observed);
        for (age, d) in &look.newly_stuck {
            crate::xlog_warn!("wait-trace: stuck {} ms: {d}", age.as_millis());
        }
        for d in &look.moved_on {
            crate::xlog_info!("wait-trace: moved on: {d}");
        }
        let idle = start().elapsed().saturating_sub(Duration::from_millis(LAST_SUBMIT_MS.load(Ordering::Relaxed)));
        if tracker.idle_due(now, idle) {
            crate::xlog_warn!(
                "wait-trace: idle {} ms: nothing submitted since submit #{} (frame {}); {} stuck waits, {} lagging title fences",
                idle.as_millis(),
                SUBMITS.load(Ordering::Relaxed),
                FRAMES.load(Ordering::Relaxed),
                look.stuck.len(),
                lagging.len()
            );
            for (age, d) in look.stuck.iter().take(STUCK_PER_IDLE) {
                crate::xlog_warn!("wait-trace:   stuck {} ms: {d}", age.as_millis());
            }
            if look.stuck.len() > STUCK_PER_IDLE {
                crate::xlog_warn!("wait-trace:   +{} more stuck waits", look.stuck.len() - STUCK_PER_IDLE);
            }
            for d in &lagging {
                crate::xlog_warn!("wait-trace:   {d}");
            }
        }
        if now.saturating_duration_since(totals_at) >= SKIP_TOTALS_PERIOD {
            totals_at = now;
            let t = (SKIPPED_LISTS.load(Ordering::Relaxed), SKIPPED_DRAWS.load(Ordering::Relaxed), SKIPPED_DISPATCHES.load(Ordering::Relaxed));
            if t != totals_last && SKIP_LINES_WRITTEN.load(Ordering::Relaxed) > SKIP_LINES {
                crate::xlog_info!(
                    "wait-trace: skips so far: {} submitted lists with skips, {} draws, {} dispatches (per-list lines stopped after {SKIP_LINES})",
                    t.0,
                    t.1,
                    t.2
                );
            }
            totals_last = t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_is_0_or_1() {
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("")), Ok(false));
        assert_eq!(parse(Some("0")), Ok(false));
        assert_eq!(parse(Some(" 1 ")), Ok(true));
        assert!(parse(Some("yes")).is_err());
    }

    #[test]
    fn roots_are_the_address_arguments() {
        let layout = Layout {
            params: vec![Param::Constants { at: 0, count: 2 }, Param::Descriptor { at: 8 }, Param::Table { at: 16 }],
            size: 20,
            inline: 0,
            inline_gfx: 0,
        };
        let mut args = vec![0u8; 20];
        args[8..16].copy_from_slice(&0x1_2345_6780u64.to_le_bytes());
        args[16..20].copy_from_slice(&0x40u32.to_le_bytes());
        assert_eq!(roots(Some(&layout), &args), vec![Root::Descriptor(0x1_2345_6780), Root::Table(0x40)]);
        assert_eq!(roots(None, &args), vec![]);
        assert_eq!(roots(Some(&layout), &args[..12]), vec![]);
    }

    #[test]
    fn list_skips_group_by_pipeline_and_cap_the_line() {
        let mut s = ListSkips::default();
        s.add(SkipKind::Dispatch, SkipWhy::Budget, 1, 0xa, vec![Root::Descriptor(0x100)]);
        s.add(SkipKind::Dispatch, SkipWhy::Budget, 1, 0xa, vec![Root::Descriptor(0x100), Root::Table(8)]);
        s.add(SkipKind::Draw, SkipWhy::Untranslated, 2, 0, vec![]);
        for pso in 3..3 + PIPELINES_PER_LINE as usize {
            s.add(SkipKind::Draw, SkipWhy::Budget, pso, 0, vec![]);
        }
        assert_eq!((s.draws, s.dispatches), (1 + PIPELINES_PER_LINE as u64, 2));
        assert_eq!(s.pipelines.len(), PIPELINES_PER_LINE);
        assert_eq!(s.pipelines[0].count, 2);
        assert_eq!(s.pipelines[0].roots, vec![Root::Descriptor(0x100), Root::Table(8)]);
        assert_eq!(s.more, 2);
        let line = s.describe();
        assert!(line.starts_with("pso 0x1 key 000000000000000a x2 [budget] roots [d:0x100 t:0x8]"), "{line}");
        assert!(line.ends_with("+2 skips of other pipelines"), "{line}");
        let mut merged = ListSkips::default();
        merged.merge(s);
        assert_eq!(merged.pipelines.len(), PIPELINES_PER_LINE);
    }

    fn observed(key: &str, progress: u64, since: Option<Instant>) -> Observed {
        Observed { key: key.into(), progress, since, describe: key.into() }
    }

    #[test]
    fn a_wait_is_reported_once_when_stuck_and_once_when_it_moves_on() {
        let t0 = Instant::now();
        let mut t = Tracker::default();
        let look = t.look(t0, vec![observed("fence", 5, None), observed("cell", 0, Some(t0))]);
        assert!(look.newly_stuck.is_empty() && look.stuck.is_empty());
        let later = t0 + STUCK_AFTER;
        let look = t.look(later, vec![observed("fence", 5, None), observed("cell", 0, Some(t0))]);
        assert_eq!(look.newly_stuck.len(), 2);
        let look = t.look(later + TICK, vec![observed("fence", 5, None), observed("cell", 0, Some(t0))]);
        assert!(look.newly_stuck.is_empty());
        assert_eq!(look.stuck.len(), 2);
        let look = t.look(later + 2 * TICK, vec![observed("fence", 6, None)]);
        assert_eq!(look.moved_on.len(), 2, "the fence advanced and the cell wait is gone");
        assert!(look.stuck.is_empty());
    }

    #[test]
    fn the_idle_list_repeats_while_nothing_is_submitted() {
        let t0 = Instant::now();
        let mut t = Tracker::default();
        assert!(!t.idle_due(t0, IDLE_AFTER - Duration::from_millis(1)));
        assert!(t.idle_due(t0, IDLE_AFTER));
        assert!(!t.idle_due(t0 + TICK, IDLE_AFTER + TICK));
        assert!(t.idle_due(t0 + IDLE_REPEAT, IDLE_AFTER + IDLE_REPEAT));
        assert!(!t.idle_due(t0 + IDLE_REPEAT, Duration::ZERO));
        assert!(t.idle_due(t0 + IDLE_REPEAT, IDLE_AFTER));
    }
}
