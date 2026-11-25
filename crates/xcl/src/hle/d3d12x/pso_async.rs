//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    demand: Mutex<Demand>,
    /// when the build reached the driver's compiler (`driver_build_starts`), None: not yet
    driver_since: Mutex<Option<std::time::Instant>>,
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted);
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
/// Builds in flight now (`build`), and a number per build, for the memory lines below.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static BUILD_NUMBER: AtomicUsize = AtomicUsize::new(0);

const MIB: u64 = 1 << 20;

/// A build that has started: its number, when, and the process's commit charge then.
struct BuildStart {
    number: usize,
    at: std::time::Instant,
    commit: u64,
}

/// Logs a build's start with the process's commit charge and the builds already in flight, so
/// the memory a driver compile holds (during it, and whether it gives it back) can be read from
/// the log: a retail title's world load ran out of commit while XCL's own pools stayed flat.
fn build_starts(preload: bool) -> BuildStart {
    let number = BUILD_NUMBER.fetch_add(1, Ordering::Relaxed) + 1;
    let others = IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
    let commit = crate::hle::kernelx::process_private_bytes();
    let by = if preload { "preload" } else { "bind" };
    crate::xlog_info!("D3D12.X: pipeline build #{number} starts ({by}): commit {} MB, {others} other builds in flight", commit / MIB);
    BuildStart { number, at: std::time::Instant::now(), commit }
}

fn build_ends(s: BuildStart) {
    let others = IN_FLIGHT.fetch_sub(1, Ordering::AcqRel) - 1;
    let commit = crate::hle::kernelx::process_private_bytes();
    let delta = commit as i64 / MIB as i64 - s.commit as i64 / MIB as i64;
    crate::xlog_info!(
        "D3D12.X: pipeline build #{} ends after {} ms: commit {} MB ({delta:+} MB), {others} other builds in flight",
        s.number,
        s.at.elapsed().as_millis(),
        commit / MIB
    );
}

fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    if preload {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    let started = build_starts(preload);
    let built = job();
    build_ends(started);
    complete(p, built);
    if preload {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let mut d = self.demand.lock().unwrap();
        if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Whether a bind is waiting for a pipeline still being built: one of the title's threads is
/// held where a console title never waits (`kernelx::Sleep` does not count that time).
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Stack of every thread that builds a pipeline: DXC and the driver compiler recurse deeply on
/// the biggest translated programs and overflow a title thread's stack.
const BUILD_STACK_BYTES: usize = 64 << 20;

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Arc<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(BUILD_STACK_BYTES).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    build(&p, job, false);
                }
            });
        }
        Mutex::new(tx)
    })
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built ahead (`Preload`) or else at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`),
/// and for every compute pipeline (after those: `Preload`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        demand: Default::default(),
        driver_since: Mutex::new(None),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        let _ = queue().lock().unwrap().send(p);
    } else if used::known(key) || compute {
        if let Some(q) = preload() {
            let mut g = q.list.lock().unwrap();
            if used::known(key) {
                g.0.push_back(p);
            } else {
                g.1.push_back(p);
            }
            drop(g);
            q.cv.notify_one();
        }
    }
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
///
/// After those, every compute pipeline is built ahead too, as it is made: its first bind waits
/// for it on the title's thread (a skipped dispatch would corrupt what it computes), and the
/// driver takes seconds for a big one – on an NVIDIA GPU a retail title's binds waited 3 – 23 s
/// each, 16 of them within 10 s on five of its threads, and its hang watchdog ended it. Compute
/// pipelines are few (2 734 of the title's ~58 000), so building them all costs little; graphics
/// pipelines stay lazy. Each build is kept on disk (`real::driver_cache`) for later runs.
struct Preload {
    /// (of shaders an earlier run used, the other compute pipelines): the first goes first
    list: Mutex<(std::collections::VecDeque<Arc<Pending>>, std::collections::VecDeque<Arc<Pending>>)>,
    cv: Condvar,
}

/// How often a preload worker looks again whether the title still waits for a bind's build
/// before it takes its next pipeline. A worker takes no new build while a bind waits: in a
/// retail title's world load six driver compiles ran at once (two ahead of their use, two of the
/// driver's own background compiles), the title waited for one or two of them, and the
/// process's commit grew 10 GB in 45 s up to the memory guard.
const PRELOAD_YIELD_POLL: std::time::Duration = std::time::Duration::from_millis(20);

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(BUILD_STACK_BYTES).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    // (while a bind waits for its own pipeline, the driver's compiler is left to it)
                    while title_held() {
                        std::thread::sleep(PRELOAD_YIELD_POLL);
                    }
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.0.pop_front().or_else(|| g.1.pop_front()) {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        build(&p, job, true);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn a_bind_that_builds_its_pipeline_builds_it_on_a_big_stack() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadStackLimits, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        const BUILT: usize = 0x5555;
        let standin = 0x7E57_B006usize;
        let binding_thread = std::thread::current().id();
        let (tx, rx) = mpsc::channel();
        // a graphics pipeline: a compute one may be taken first by the workers that build ahead
        submit_lazy(standin, false, Box::new(move || {
            let (mut lo, mut hi) = (0usize, 0usize);
            unsafe { GetCurrentThreadStackLimits(&mut lo, &mut hi) };
            let priority = unsafe { GetThreadPriority(GetCurrentThread()) };
            tx.send((hi - lo, std::thread::current().id(), build_is_wanted(), wanted_here()(), priority)).unwrap();
            BUILT
        }), 0);
        assert!(is_pending(standin), "a lazy pipeline waits for its first bind");
        let (old_priority, ok) = unsafe { (GetThreadPriority(GetCurrentThread()), SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)) };
        assert_ne!(ok, 0);
        let real = resolve_wait_for(standin, None);
        unsafe { SetThreadPriority(GetCurrentThread(), old_priority) };
        assert_eq!(real, BUILT);
        let (stack, builder, build_wanted, compiles_wanted, priority) = rx.recv().unwrap();
        assert!(stack >= BUILD_STACK_BYTES, "built on a {stack}-byte stack");
        assert_ne!(builder, binding_thread, "not on the bind's own stack");
        assert!(build_wanted && compiles_wanted, "the bind waits for this build");
        assert_eq!(priority, THREAD_PRIORITY_BELOW_NORMAL, "at the bind's priority");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// A compute pipeline is built without a bind even when no earlier run used it: its first
    /// bind would otherwise build it on the title's thread. A graphics one made with it is not.
    #[test]
    fn a_compute_pipeline_is_built_ahead_of_its_first_bind() {
        let (compute, graphics) = (0x7E57_7003usize, 0x7E57_7004usize);
        submit_lazy(compute, true, Box::new(|| 0x4323), 0x5EED_0003);
        submit_lazy(graphics, false, Box::new(|| 0x4324), 0x5EED_0004);
        let start = std::time::Instant::now();
        while is_pending(compute) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(compute), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(compute), 0x4323);
        assert!(is_pending(graphics), "graphics pipelines stay lazy");
        assert_eq!(resolve_wait(graphics), 0x4324);
        released(compute);
        released(graphics);
    }
}

#[cfg(test)]
mod frame_budget_tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MS: Duration = Duration::from_millis(1);

    fn budget(limit_ms: u32, window_ms: u32, kinds: FrameWaitKinds, now: Instant) -> FrameBudget {
        FrameBudget::new(MS * limit_ms, MS * window_ms, kinds, now)
    }

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| vars.iter().find(|(n, _)| *n == name).map(|(_, v)| v.to_string())
    }

    fn parse(vars: &[(&str, &str)]) -> Result<Option<FrameBudget>, String> {
        FrameBudget::parse(env(vars), Instant::now())
    }

    fn parse_err(vars: &[(&str, &str)]) -> String {
        match parse(vars) {
            Err(e) => e,
            Ok(_) => panic!("{vars:?} is not an error"),
        }
    }

    const COMPUTE_VARS: [(&str, &str); 2] = [(FRAME_WAIT_COMPUTE_MS_VAR, "12000"), (COMPUTE_WAIT_TOTAL_MS_VAR, "120000")];

    fn joined(parts: &[&[(&'static str, &'static str)]]) -> Vec<(&'static str, &'static str)> {
        parts.concat()
    }

    #[test]
    fn the_frame_budget_is_off_unless_its_milliseconds_are_set() {
        assert!(parse(&[]).unwrap().is_none());
        assert!(parse(&[(FRAME_WAIT_MS_VAR, "0")]).unwrap().is_none());
        assert!(parse(&[(FRAME_WAIT_MS_VAR, " "), (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, "all")]).unwrap().is_none());
    }

    #[test]
    fn a_frame_budget_without_its_window_or_kinds_is_an_error() {
        let ms = (FRAME_WAIT_MS_VAR, "200");
        let missing_window = parse_err(&[ms, (FRAME_WAIT_KINDS_VAR, "graphics")]);
        assert!(missing_window.contains(FRAME_WAIT_WINDOW_MS_VAR), "{missing_window}");
        let zero_window = parse_err(&[ms, (FRAME_WAIT_WINDOW_MS_VAR, "0"), (FRAME_WAIT_KINDS_VAR, "graphics")]);
        assert!(zero_window.contains(FRAME_WAIT_WINDOW_MS_VAR), "{zero_window}");
        let missing_kinds = parse_err(&[ms, (FRAME_WAIT_WINDOW_MS_VAR, "1000")]);
        assert!(missing_kinds.contains(FRAME_WAIT_KINDS_VAR), "{missing_kinds}");
        let wrong_kinds = parse_err(&[ms, (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, "pixel")]);
        assert!(wrong_kinds.contains(FRAME_WAIT_KINDS_VAR), "{wrong_kinds}");
        let wrong_ms = parse_err(&[(FRAME_WAIT_MS_VAR, "lots"), (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, "all")]);
        assert!(wrong_ms.contains(FRAME_WAIT_MS_VAR), "{wrong_ms}");
        let f = parse(&[ms, (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, " Graphics ")]).unwrap().unwrap();
        assert_eq!((f.limit, f.window, f.kinds, f.compute, f.cutoff), (MS * 200, MS * 1000, FrameWaitKinds::Graphics, None, DriverCutoff::default()));
    }

    /// With compute binds covered, both compute variables are required (above 0); without, or
    /// with the budget off, they are an error, as are the cutoffs.
    #[test]
    fn the_compute_budget_and_its_cap_are_required_when_compute_binds_are_covered() {
        let base = [(FRAME_WAIT_MS_VAR, "200"), (FRAME_WAIT_WINDOW_MS_VAR, "1000")];
        for kinds in ["compute", "all"] {
            let kinds = [(FRAME_WAIT_KINDS_VAR, kinds)];
            let with = |extra: &[(&'static str, &'static str)]| joined(&[&base, &kinds, extra]);
            let e = parse_err(&with(&[]));
            assert!(e.contains(FRAME_WAIT_COMPUTE_MS_VAR), "{e}");
            let e = parse_err(&with(&[COMPUTE_VARS[0]]));
            assert!(e.contains(COMPUTE_WAIT_TOTAL_MS_VAR), "{e}");
            let e = parse_err(&with(&[COMPUTE_VARS[1]]));
            assert!(e.contains(FRAME_WAIT_COMPUTE_MS_VAR), "{e}");
            for wrong in ["0", "-1", "lots"] {
                let e = parse_err(&with(&[(FRAME_WAIT_COMPUTE_MS_VAR, wrong), COMPUTE_VARS[1]]));
                assert!(e.contains(FRAME_WAIT_COMPUTE_MS_VAR), "{e}");
                let e = parse_err(&with(&[COMPUTE_VARS[0], (COMPUTE_WAIT_TOTAL_MS_VAR, wrong)]));
                assert!(e.contains(COMPUTE_WAIT_TOTAL_MS_VAR), "{e}");
            }
            let f = parse(&with(&COMPUTE_VARS)).unwrap().unwrap();
            assert_eq!(f.compute, Some(ComputeBudget { frame: MS * 12_000, run_cap: MS * 120_000 }));
        }
        let graphics = joined(&[&base, &[(FRAME_WAIT_KINDS_VAR, "graphics")]]);
        for v in COMPUTE_VARS {
            let e = parse_err(&joined(&[&graphics, &[v]]));
            assert!(e.contains(v.0) && e.contains(FRAME_WAIT_KINDS_VAR), "{e}");
        }
        for v in [COMPUTE_VARS[0], COMPUTE_VARS[1], (DRIVER_CUTOFF_GRAPHICS_VAR, "3000"), (DRIVER_CUTOFF_COMPUTE_VAR, "10000")] {
            let e = parse_err(&[v]);
            assert!(e.contains(v.0) && e.contains(FRAME_WAIT_MS_VAR), "budget off: {e}");
            let e = parse_err(&[(FRAME_WAIT_MS_VAR, "0"), v]);
            assert!(e.contains(v.0), "budget 0: {e}");
        }
    }

    /// The cutoffs are optional (unset: none), above 0, and only for the kinds the budget covers.
    #[test]
    fn the_driver_cutoffs_are_optional() {
        let all = [(FRAME_WAIT_MS_VAR, "200"), (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, "all"), COMPUTE_VARS[0], COMPUTE_VARS[1]];
        assert_eq!(parse(&all).unwrap().unwrap().cutoff, DriverCutoff::default());
        let f = parse(&joined(&[&all, &[(DRIVER_CUTOFF_GRAPHICS_VAR, "3000"), (DRIVER_CUTOFF_COMPUTE_VAR, "10000")]])).unwrap().unwrap();
        assert_eq!(f.cutoff, DriverCutoff { graphics: Some(MS * 3_000), compute: Some(MS * 10_000) });
        assert_eq!((f.cutoff.of(false), f.cutoff.of(true)), (Some(MS * 3_000), Some(MS * 10_000)));
        let f = parse(&joined(&[&all, &[(DRIVER_CUTOFF_COMPUTE_VAR, "10000")]])).unwrap().unwrap();
        assert_eq!(f.cutoff, DriverCutoff { graphics: None, compute: Some(MS * 10_000) });
        for name in [DRIVER_CUTOFF_GRAPHICS_VAR, DRIVER_CUTOFF_COMPUTE_VAR] {
            for wrong in ["0", "soon"] {
                let e = parse_err(&joined(&[&all, &[(name, wrong)]]));
                assert!(e.contains(name), "{e}");
            }
        }
        let graphics = [(FRAME_WAIT_MS_VAR, "200"), (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, "graphics")];
        let e = parse_err(&joined(&[&graphics, &[(DRIVER_CUTOFF_COMPUTE_VAR, "10000")]]));
        assert!(e.contains(DRIVER_CUTOFF_COMPUTE_VAR), "{e}");
        let compute = [(FRAME_WAIT_MS_VAR, "200"), (FRAME_WAIT_WINDOW_MS_VAR, "1000"), (FRAME_WAIT_KINDS_VAR, "compute"), COMPUTE_VARS[0], COMPUTE_VARS[1]];
        let e = parse_err(&joined(&[&compute, &[(DRIVER_CUTOFF_GRAPHICS_VAR, "3000")]]));
        assert!(e.contains(DRIVER_CUTOFF_GRAPHICS_VAR), "{e}");
    }

    #[test]
    fn the_kinds_say_which_binds_the_budget_covers() {
        assert!(FrameWaitKinds::Compute.covers(true) && !FrameWaitKinds::Compute.covers(false));
        assert!(FrameWaitKinds::Graphics.covers(false) && !FrameWaitKinds::Graphics.covers(true));
        assert!(FrameWaitKinds::All.covers(true) && FrameWaitKinds::All.covers(false));
    }

    /// A wait is reserved before it begins: the budget is smaller at once, before the wait ends.
    #[test]
    fn a_wait_is_reserved_before_it_begins() {
        let t0 = Instant::now();
        let f = budget(100, 60_000, FrameWaitKinds::All, t0);
        let a = f.reserve(false, Some(MS * 30), t0);
        assert_eq!(a.grant, MS * 30);
        assert_eq!(f.left(false, t0), MS * 70, "reserved, not yet waited");
        let b = f.reserve(false, None, t0);
        assert_eq!(b.grant, MS * 70, "no more than is left");
        assert_eq!(f.left(false, t0), Duration::ZERO);
        let c = f.reserve(false, None, t0);
        assert!(c.grant.is_zero(), "all of it is reserved: the bind does not wait");
        f.settle(&c, Duration::ZERO);
        assert_eq!(f.left(false, t0), Duration::ZERO);
    }

    /// What a bind reserved and did not wait is given back; a wait longer than its reservation
    /// (the scheduler's lateness) is charged no more than the reservation.
    #[test]
    fn what_a_wait_did_not_use_is_given_back() {
        let t0 = Instant::now();
        let f = budget(100, 60_000, FrameWaitKinds::All, t0);
        let r = f.reserve(true, None, t0);
        assert_eq!(r.grant, MS * 100);
        f.settle(&r, MS * 30);
        assert_eq!(f.left(true, t0), MS * 70);
        let r = f.reserve(true, None, t0);
        f.settle(&r, MS * 500);
        assert_eq!(f.left(true, t0), Duration::ZERO, "spent: nothing is left, not less");
        f.settle(&r, Duration::ZERO);
        assert_eq!(f.left(true, t0), Duration::ZERO, "a reservation is given back once");
    }

    /// A wait going on when the budget is new keeps what is left of its reservation in the new
    /// frame, so the frames' waits in all stay within the budget; what it waited before the
    /// renewal is not charged to the new frame.
    #[test]
    fn a_wait_going_on_keeps_what_is_left_of_its_reservation_in_the_new_frame() {
        let t0 = Instant::now();
        let f = budget(100, 60_000, FrameWaitKinds::All, t0);
        let r = f.reserve(false, None, t0);
        f.presented(t0 + MS * 40);
        assert_eq!(f.left(false, t0 + MS * 40), MS * 40, "60 ms of the wait may still come in this frame");
        f.settle(&r, MS * 50);
        assert_eq!(f.left(false, t0 + MS * 50), MS * 90, "10 ms of the wait were in the new frame");
        let r = f.reserve(false, None, t0 + MS * 50);
        assert_eq!(r.grant, MS * 90);
        f.presented(t0 + MS * 200);
        assert_eq!(f.left(false, t0 + MS * 200), MS * 100, "a reservation past its end holds nothing");
        f.settle(&r, MS * 90);
        assert_eq!(f.left(false, t0 + MS * 200), MS * 100);
    }

    #[test]
    fn the_budget_is_renewed_when_its_window_passes_without_a_present() {
        let t0 = Instant::now();
        let f = budget(100, 1_000, FrameWaitKinds::Compute, t0);
        let r = f.reserve(true, None, t0);
        f.settle(&r, MS * 100);
        assert_eq!(f.left(true, t0 + MS * 999), Duration::ZERO);
        assert_eq!(f.left(true, t0 + MS * 1_000), MS * 100, "the window passed");
        let r = f.reserve(true, None, t0 + MS * 1_000);
        f.settle(&r, MS * 100);
        assert_eq!(f.left(true, t0 + MS * 1_500), Duration::ZERO, "a new window starts at the renewal");
        f.presented(t0 + MS * 1_800);
        let r = f.reserve(true, None, t0 + MS * 1_800);
        f.settle(&r, MS * 100);
        assert_eq!(f.left(true, t0 + MS * 2_700), Duration::ZERO, "a present starts a new window too");
        assert_eq!(f.left(true, t0 + MS * 2_800), MS * 100);
    }

    fn with_compute(limit_ms: u32, compute_ms: u32, cap_ms: u32, now: Instant) -> FrameBudget {
        let mut f = budget(limit_ms, 60_000, FrameWaitKinds::All, now);
        f.compute = Some(ComputeBudget { frame: MS * compute_ms, run_cap: MS * cap_ms });
        f
    }

    /// Compute binds wait from their own budget, graphics binds from `XCL_PSO_FRAME_WAIT_MS`; a
    /// spent compute frame budget skips (it does not borrow the other); once the per-run cap is
    /// spent compute binds wait within `XCL_PSO_FRAME_WAIT_MS`.
    #[test]
    fn compute_binds_have_their_own_budget_until_its_run_cap_is_spent() {
        let t0 = Instant::now();
        let f = with_compute(100, 300, 500, t0);
        assert_eq!((f.left(true, t0), f.left(false, t0)), (MS * 300, MS * 100));
        let c = f.reserve(true, None, t0);
        assert_eq!((c.pool, c.grant), (Pool::Compute, MS * 300));
        assert_eq!(f.left(false, t0), MS * 100, "graphics keeps its budget");
        assert!(f.reserve(true, None, t0).grant.is_zero(), "the compute frame budget is spent: skip");
        f.settle(&c, MS * 300);
        assert_eq!(f.run_compute_spent(), MS * 300);
        f.presented(t0 + MS);
        let c = f.reserve(true, None, t0 + MS);
        assert_eq!((c.pool, c.grant), (Pool::Compute, MS * 200), "no more than is left of the run cap");
        f.settle(&c, MS * 50);
        assert_eq!(f.run_compute_spent(), MS * 350, "the unused part goes back to the cap too");
        let c = f.reserve(true, None, t0 + MS);
        assert_eq!(c.grant, MS * 150);
        f.settle(&c, MS * 150);
        assert_eq!(f.run_compute_spent(), MS * 500);
        let c = f.reserve(true, None, t0 + MS);
        assert_eq!((c.pool, c.grant), (Pool::Frame, MS * 100), "the cap is spent: the frame budget");
        assert!(f.reserve(false, None, t0 + MS).grant.is_zero(), "shared with graphics now");
        f.settle(&c, MS * 10);
        assert_eq!(f.left(false, t0 + MS), MS * 90);
        assert_eq!(f.run_compute_spent(), MS * 500);
    }

    /// Reservations on many threads at once never come to more than the budget.
    #[test]
    fn reservations_on_many_threads_never_exceed_the_budget() {
        let t0 = Instant::now();
        let f = Arc::new(with_compute(100, 250, 10_000, t0));
        let granted: Vec<(bool, Duration)> = (0..16)
            .map(|k| {
                let f = f.clone();
                std::thread::spawn(move || {
                    let compute = k % 2 == 0;
                    (compute, f.reserve(compute, Some(MS * 40), t0).grant)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        let sum = |c: bool| granted.iter().filter(|g| g.0 == c).map(|g| g.1).sum::<Duration>();
        assert_eq!(sum(false), MS * 100, "graphics: exactly the budget, in all");
        assert_eq!(sum(true), MS * 250, "compute: exactly its own budget, in all");
    }

    /// Binds on several threads at once, each waiting for a pipeline that is not built in
    /// time, wait no more than the budget in all.
    #[test]
    fn parallel_waits_never_exceed_the_budget() {
        const THREADS: usize = 6;
        const LIMIT_MS: u32 = 150;
        const SLACK_PER_THREAD_MS: u32 = 25;
        let f: &'static FrameBudget = Box::leak(Box::new(budget(LIMIT_MS, 60_000, FrameWaitKinds::Compute, Instant::now())));
        let standins: Vec<usize> = (0..THREADS).map(|k| 0x7E57_F100 + k).collect();
        let mut gos = Vec::new();
        for &s in &standins {
            let (go, finish) = mpsc::channel::<()>();
            gos.push(go);
            submit(s, true, Box::new(move || {
                let _ = finish.recv_timeout(Duration::from_secs(10));
                0x6666
            }));
        }
        let barrier = Arc::new(std::sync::Barrier::new(THREADS));
        let waited: Duration = standins
            .iter()
            .map(|&s| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let t = Instant::now();
                    assert_eq!(resolve_wait_in(s, None, Some(f)), (s, true));
                    t.elapsed()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .sum();
        assert!(waited <= MS * (LIMIT_MS + SLACK_PER_THREAD_MS * THREADS as u32), "{THREADS} binds waited {waited:?} in all");
        assert_eq!(f.left(true, Instant::now()), Duration::ZERO, "the budget was waited out");
        for go in gos {
            let _ = go.send(());
        }
        for &s in &standins {
            wait_until_built(s);
            let _ = released(s);
        }
    }

    /// A bind does not wait for a pipeline the driver has been building longer than the cutoff
    /// (and spends no budget on it); for one the driver started on just now it waits no longer
    /// than to the cutoff; a cutoff of the other kind does not bound it.
    #[test]
    fn a_pipeline_past_the_driver_cutoff_is_not_waited_for() {
        let (old, fresh, graphics) = (0x7E57_F201usize, 0x7E57_F202usize, 0x7E57_F203usize);
        let mut gos = Vec::new();
        let mut in_driver = Vec::new();
        for (s, compute) in [(old, true), (fresh, true), (graphics, false)] {
            let (go, finish) = mpsc::channel::<()>();
            let (started, driver) = mpsc::channel::<()>();
            gos.push(go);
            in_driver.push(driver);
            submit(s, compute, Box::new(move || {
                driver_build_starts();
                let _ = started.send(());
                let _ = finish.recv_timeout(Duration::from_secs(10));
                0x6767
            }));
        }
        for d in &in_driver {
            d.recv_timeout(Duration::from_secs(10)).expect("in the driver");
        }
        let mut f = budget(5_000, 60_000, FrameWaitKinds::All, Instant::now());
        f.cutoff = DriverCutoff { graphics: Some(MS * 400), compute: Some(MS * 300) };
        std::thread::sleep(MS * 120);
        let mut f_old = budget(5_000, 60_000, FrameWaitKinds::All, Instant::now());
        f_old.cutoff = DriverCutoff { graphics: None, compute: Some(MS * 100) };
        let skips = FRAME_STATS.cutoff_compute.load(Ordering::Relaxed);
        let t = Instant::now();
        assert_eq!(resolve_wait_in(old, None, Some(&f_old)), (old, true));
        assert!(t.elapsed() < MS * 50, "not waited for ({:?})", t.elapsed());
        assert!(FRAME_STATS.cutoff_compute.load(Ordering::Relaxed) > skips, "counted");
        assert_eq!(f_old.left(true, Instant::now()), MS * 5_000, "no budget spent");
        let t = Instant::now();
        assert_eq!(resolve_wait_in(fresh, None, Some(&f)), (fresh, true));
        let waited = t.elapsed();
        assert!(waited < MS * 1_000, "waited only to the cutoff ({waited:?})");
        assert!(!f.left(true, Instant::now()).is_zero());
        assert_eq!(resolve_wait_in(fresh, None, Some(&f)), (fresh, true), "past the cutoff now");
        let mut graphics_only = budget(150, 60_000, FrameWaitKinds::All, Instant::now());
        graphics_only.cutoff = DriverCutoff { graphics: None, compute: Some(MS * 1) };
        let t = Instant::now();
        assert_eq!(resolve_wait_in(graphics, None, Some(&graphics_only)), (graphics, true));
        assert!(t.elapsed() >= MS * 140, "the compute cutoff does not bound a graphics bind ({:?})", t.elapsed());
        for go in gos {
            let _ = go.send(());
        }
        for s in [old, fresh, graphics] {
            wait_until_built(s);
            assert_eq!(resolve_wait_in(s, None, Some(&f)), (0x6767, false), "built: the real pipeline");
            let _ = released(s);
        }
    }

    fn wait_until_built(standin: usize) {
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(MS * 5);
        }
    }

    /// The first bind waits what is left of the frame's budget, the next ones of the frame do
    /// not wait and go on without the pipeline, a renewed budget is waited from again, and the
    /// build goes on (wanted) until a later bind gets the real pipeline.
    #[test]
    fn binds_wait_for_unbuilt_pipelines_no_longer_than_the_frame_budget_in_all() {
        let (slow, other) = (0x7E57_F001usize, 0x7E57_F002usize);
        let (go, finish) = mpsc::channel::<()>();
        let (wanted_tx, wanted_rx) = mpsc::channel();
        submit(slow, true, Box::new(move || {
            let _ = finish.recv_timeout(Duration::from_secs(10));
            let _ = wanted_tx.send(build_is_wanted());
            0x6161
        }));
        let (go_other, finish_other) = mpsc::channel::<()>();
        submit(other, true, Box::new(move || {
            let _ = finish_other.recv_timeout(Duration::from_secs(10));
            0x6262
        }));
        let f = budget(80, 60_000, FrameWaitKinds::Compute, Instant::now());
        let t = Instant::now();
        assert_eq!(resolve_wait_in(slow, None, Some(&f)), (slow, true), "not built within the budget: the stand-in, skipped");
        let first = t.elapsed();
        assert!(first >= MS * 70 && first < MS * 1_000, "waited {first:?}");
        let t = Instant::now();
        assert_eq!(resolve_wait_in(other, None, Some(&f)), (other, true));
        assert_eq!(resolve_wait_in(slow, None, Some(&f)), (slow, true));
        assert!(t.elapsed() < MS * 40, "the frame's budget is spent: no more waits ({:?})", t.elapsed());
        f.presented(Instant::now());
        let t = Instant::now();
        assert_eq!(resolve_wait_in(slow, None, Some(&f)), (slow, true));
        assert!(t.elapsed() >= MS * 70, "a new frame waits again ({:?})", t.elapsed());
        go.send(()).unwrap();
        go_other.send(()).unwrap();
        assert!(wanted_rx.recv_timeout(Duration::from_secs(10)).unwrap(), "the build is still wanted after its binds went on");
        wait_until_built(slow);
        wait_until_built(other);
        assert_eq!(resolve_wait_in(slow, None, Some(&f)), (0x6161, false), "built in the background: the real pipeline");
        assert_eq!(resolve_wait_in(other, None, Some(&f)), (0x6262, false));
        let _ = released(slow);
        let _ = released(other);
    }

    /// A bind of a pipeline that is built within what is left of the budget gets it.
    #[test]
    fn a_pipeline_built_within_the_frame_budget_is_bound() {
        let standin = 0x7E57_F003usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(MS * 50);
            0x6363
        }));
        let f = budget(5_000, 60_000, FrameWaitKinds::All, Instant::now());
        assert_eq!(resolve_wait_in(standin, None, Some(&f)), (0x6363, false));
        assert!(f.left(true, Instant::now()) < MS * 5_000, "the wait was charged");
        let _ = released(standin);
    }

    /// A budget of the other kind leaves a bind as it was: it waits until its pipeline is built.
    #[test]
    fn a_frame_budget_for_graphics_does_not_bound_a_compute_bind() {
        let standin = 0x7E57_F004usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(MS * 200);
            0x6464
        }));
        let f = budget(10, 60_000, FrameWaitKinds::Graphics, Instant::now());
        assert_eq!(resolve_wait_in(standin, None, Some(&f)), (0x6464, false));
        assert_eq!(f.left(false, Instant::now()), MS * 10, "nothing charged");
        let _ = released(standin);
    }

    /// With `XCL_PSO_WAIT_MS` too, a bind waits the shorter of the two; only that per-pipeline
    /// budget makes the pipeline late (not waited for again).
    #[test]
    fn the_shorter_of_the_frame_and_pipeline_budgets_bounds_a_bind() {
        let standin = 0x7E57_F005usize;
        let (go, finish) = mpsc::channel::<()>();
        submit(standin, false, Box::new(move || {
            let _ = finish.recv_timeout(Duration::from_secs(10));
            0x6565
        }));
        let f = budget(5_000, 60_000, FrameWaitKinds::Graphics, Instant::now());
        let t = Instant::now();
        assert_eq!(resolve_wait_in(standin, Some(MS * 40), Some(&f)), (standin, true));
        assert!(t.elapsed() < MS * 1_000, "the per-pipeline budget bounds it ({:?})", t.elapsed());
        let t = Instant::now();
        assert_eq!(resolve_wait_in(standin, Some(MS * 40), Some(&f)), (standin, true));
        assert!(t.elapsed() < MS * 30, "late: not waited for again ({:?})", t.elapsed());
        go.send(()).unwrap();
        wait_until_built(standin);
        assert_eq!(resolve_wait_in(standin, Some(MS * 40), Some(&f)), (0x6565, false));
        let _ = released(standin);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The key of stand-in `pso`'s shaders (as `XCL_PSO_PRELOAD` and the build log name it; 0: not
/// a stand-in, or no key) – `XCL_WAIT_TRACE`.
pub fn trace_key(pso: usize) -> u64 {
    entry(pso).map_or(0, |p| p.key)
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            if !p.sent.swap(true, Ordering::AcqRel) {
                let _ = queue().lock().unwrap().send(p);
            }
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_checked(pso).0
}

/// `resolve_wait`, and whether the frame's wait budget (`XCL_PSO_FRAME_WAIT_MS`) ran out
/// before the pipeline was built: the result is then the stand-in, and the draw or dispatch
/// that binds it is to be skipped (`note_skipped_draw` / `note_skipped_dispatch`). Always false
/// without that budget.
pub fn resolve_wait_checked(pso: usize) -> (usize, bool) {
    resolve_wait_in(pso, wait_budget(), frame_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
#[cfg(test)]
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    resolve_wait_in(pso, budget, None).0
}

/// `resolve_wait_for`, within what is left of `frame` for a pipeline of the kinds it covers.
fn resolve_wait_in(pso: usize, budget: Option<std::time::Duration>, frame: Option<&FrameBudget>) -> (usize, bool) {
    let Some(p) = entry(pso) else { return (pso, false) };
    p.note_use();
    if let Some(frame) = frame.filter(|f| f.kinds.covers(p.compute)) {
        return resolve_within_frame(&p, pso, budget, frame);
    }
    (resolve_wait_unbudgeted(&p, pso, budget), false)
}

/// `resolve_wait_for` of a pipeline the frame budget does not cover (as before it existed).
fn resolve_wait_unbudgeted(p: &Arc<Pending>, pso: usize, budget: Option<std::time::Duration>) -> usize {
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _held = (!*p.done.lock().unwrap()).then(Held::new);
    if let Some(budget) = budget {
        return resolve_within(p, pso, budget);
    }
    p.want();
    if p.job.lock().unwrap().is_some() {
        build_for_waiting_thread(p);
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// Builds the pipeline a bind waits for on a scoped thread with a `BUILD_STACK_BYTES` stack at
/// the bind's priority, and waits for it: the bind's own (title) thread may have a small stack.
/// When no such thread can be made it builds on the bind's thread, as before.
fn build_for_waiting_thread(p: &Arc<Pending>) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, SetThreadPriority};
    const THREAD_PRIORITY_ERROR_RETURN: i32 = i32::MAX;
    let priority = unsafe { GetThreadPriority(GetCurrentThread()) };
    let take_and_build = || {
        let job = p.job.lock().unwrap().take();
        if let Some(job) = job {
            build(p, job, false);
        }
    };
    std::thread::scope(|s| {
        let spawned = std::thread::Builder::new().name("xcl-pso-bind".into()).stack_size(BUILD_STACK_BYTES).spawn_scoped(s, || {
            if priority != THREAD_PRIORITY_ERROR_RETURN {
                unsafe { SetThreadPriority(GetCurrentThread(), priority) };
            }
            take_and_build();
        });
        if let Err(e) = spawned {
            crate::xlog_warn!("D3D12.X: no thread with a {} MiB stack for a pipeline a bind waits for ({e}): built on the bind's own thread", BUILD_STACK_BYTES >> 20);
            take_and_build();
        }
    });
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        let _ = queue().lock().unwrap().send(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        return pso;
    }
    p.want();
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    if *done {
        return built(p);
    }
    drop(done);
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    pso
}

/// Which binds the frame's wait budget covers (`XCL_PSO_FRAME_WAIT_KINDS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameWaitKinds {
    Compute,
    Graphics,
    All,
}

impl FrameWaitKinds {
    fn parse(text: &str) -> Option<FrameWaitKinds> {
        match text.trim().to_ascii_lowercase().as_str() {
            "compute" => Some(FrameWaitKinds::Compute),
            "graphics" => Some(FrameWaitKinds::Graphics),
            "all" => Some(FrameWaitKinds::All),
            _ => None,
        }
    }

    fn covers(self, compute: bool) -> bool {
        match self {
            FrameWaitKinds::Compute => compute,
            FrameWaitKinds::Graphics => !compute,
            FrameWaitKinds::All => true,
        }
    }
}

/// Experimental (`XCL_PSO_FRAME_WAIT_MS`): how long the binds of one frame may wait in all for
/// pipelines still being built. A retail title's world load binds a long chain of compute
/// pipelines the driver takes 2 – 30 s each to build; waiting for each one stops the title
/// presenting for minutes. With this budget a bind waits at most for what is left of it; once it
/// is spent, a bind goes on with the stand-in and its draw or dispatch is skipped, while the
/// build goes on in the background (wanted, at a normal priority) for a later frame.
///
/// The budget is new at each `PresentX`, and also after `window` of wall time since it was last
/// new: during the stall nothing is presented, and without that the binds would never wait
/// again (or never be bounded).
///
/// A bind reserves its wait before it waits, from a budget all threads share, waits at most
/// what it reserved and gives back what it did not use (`reserve`, `settle`): binds on several
/// threads at once never wait more than the budget in all. (Charged after they ended, as
/// before, 219 waits of one run came to 102 s.) A wait still going on when the budget is new
/// keeps what is left of its reservation in the new frame. The first bind may reserve all that
/// is left, so binds on other threads meanwhile go on without their pipelines.
///
/// Compute binds have a budget of their own (`compute`): a dispatch of a one-time compute pass
/// that is skipped never comes back (in another fork's run, 130 of them lost walls, floors and
/// faces). It lasts until the per-run cap is spent; compute binds then wait within `limit`.
///
/// No bind waits for a pipeline the driver has been building for longer than `cutoff` (unset:
/// no cutoff), nor past that point.
pub struct FrameBudget {
    limit: std::time::Duration,
    window: std::time::Duration,
    kinds: FrameWaitKinds,
    compute: Option<ComputeBudget>,
    cutoff: DriverCutoff,
    spend: Mutex<FrameSpend>,
}

/// The compute binds' own budget (`XCL_PSO_FRAME_WAIT_COMPUTE_MS`, `XCL_PSO_COMPUTE_WAIT_TOTAL_MS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComputeBudget {
    /// per frame
    frame: std::time::Duration,
    /// in all over the run
    run_cap: std::time::Duration,
}

/// How long the driver may have been building a pipeline for a bind still to wait for it
/// (`XCL_PSO_WAIT_DRIVER_CUTOFF_MS_GRAPHICS` / `_COMPUTE`; None: no cutoff).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DriverCutoff {
    graphics: Option<std::time::Duration>,
    compute: Option<std::time::Duration>,
}

impl DriverCutoff {
    fn of(self, compute: bool) -> Option<std::time::Duration> {
        if compute { self.compute } else { self.graphics }
    }

    fn any(self) -> bool {
        self.graphics.is_some() || self.compute.is_some()
    }
}

/// Which budget a wait is reserved from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pool {
    /// `limit`: graphics binds, and compute binds without a budget of their own or once its cap is spent
    Frame,
    /// the compute binds' own budget, and its per-run cap
    Compute,
}

struct FrameSpend {
    /// when the budget was last new
    since: std::time::Instant,
    /// of `limit` this frame: waits that ended and what waits going on still hold
    spent: std::time::Duration,
    /// of the compute budget's `frame`, the same way
    compute_spent: std::time::Duration,
    /// of the compute budget's `run_cap`: waits that ended and what waits going on reserved
    run_compute_spent: std::time::Duration,
    /// the last reservation's id
    last: u64,
    /// the waits going on, by reservation id
    reserved: HashMap<u64, Outstanding>,
}

/// A wait going on.
struct Outstanding {
    pool: Pool,
    start: std::time::Instant,
    grant: std::time::Duration,
    /// of it, charged to the frame now (all of it until the budget is new, then what was left)
    charged: std::time::Duration,
}

/// A bind's reserved wait (`reserve`; to be given back with `settle`). A zero `grant` holds nothing.
#[derive(Debug)]
struct Reservation {
    id: u64,
    pool: Pool,
    grant: std::time::Duration,
}

/// Why the frame budget was last new.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Renewal {
    Present,
    Window,
}

const FRAME_WAIT_MS_VAR: &str = "XCL_PSO_FRAME_WAIT_MS";
const FRAME_WAIT_WINDOW_MS_VAR: &str = "XCL_PSO_FRAME_WAIT_WINDOW_MS";
const FRAME_WAIT_KINDS_VAR: &str = "XCL_PSO_FRAME_WAIT_KINDS";
const FRAME_WAIT_COMPUTE_MS_VAR: &str = "XCL_PSO_FRAME_WAIT_COMPUTE_MS";
const COMPUTE_WAIT_TOTAL_MS_VAR: &str = "XCL_PSO_COMPUTE_WAIT_TOTAL_MS";
const DRIVER_CUTOFF_GRAPHICS_VAR: &str = "XCL_PSO_WAIT_DRIVER_CUTOFF_MS_GRAPHICS";
const DRIVER_CUTOFF_COMPUTE_VAR: &str = "XCL_PSO_WAIT_DRIVER_CUTOFF_MS_COMPUTE";

/// The variables that act only with the frame budget on.
const FRAME_BUDGET_ONLY_VARS: [&str; 4] = [FRAME_WAIT_COMPUTE_MS_VAR, COMPUTE_WAIT_TOTAL_MS_VAR, DRIVER_CUTOFF_GRAPHICS_VAR, DRIVER_CUTOFF_COMPUTE_VAR];

/// A variable's value in whole milliseconds above 0.
fn positive_ms(name: &str, value: &str) -> Result<std::time::Duration, String> {
    value
        .parse::<u64>()
        .ok()
        .filter(|&ms| ms > 0)
        .map(std::time::Duration::from_millis)
        .ok_or_else(|| format!("{name}={value:?} must be a whole number of milliseconds above 0"))
}

impl FrameBudget {
    /// The budget the variables' values describe (`var`: a variable's value, None: unset; None:
    /// `XCL_PSO_FRAME_WAIT_MS` unset or 0).
    fn parse(var: impl Fn(&str) -> Option<String>, now: std::time::Instant) -> Result<Option<FrameBudget>, String> {
        let get = |name: &str| var(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let off = || match FRAME_BUDGET_ONLY_VARS.iter().find(|name| get(name).is_some()) {
            Some(name) => Err(format!("{name} is set, but it acts only with the frame wait budget, and {FRAME_WAIT_MS_VAR} is unset or 0")),
            None => Ok(None),
        };
        let Some(ms) = get(FRAME_WAIT_MS_VAR) else { return off() };
        let ms: u64 = ms.parse().map_err(|_| format!("{FRAME_WAIT_MS_VAR}={ms:?} is not a whole number of milliseconds"))?;
        if ms == 0 {
            return off();
        }
        let window = get(FRAME_WAIT_WINDOW_MS_VAR)
            .ok_or_else(|| format!("{FRAME_WAIT_MS_VAR} is set, so {FRAME_WAIT_WINDOW_MS_VAR} (the wall time in ms after which the budget is new without a present) must be set too"))?;
        let window: u64 = window
            .parse()
            .ok()
            .filter(|&w| w > 0)
            .ok_or_else(|| format!("{FRAME_WAIT_WINDOW_MS_VAR}={window:?} must be a whole number of milliseconds above 0"))?;
        let kinds = get(FRAME_WAIT_KINDS_VAR).ok_or_else(|| format!("{FRAME_WAIT_MS_VAR} is set, so {FRAME_WAIT_KINDS_VAR} (compute, graphics or all) must be set too"))?;
        let kinds = FrameWaitKinds::parse(&kinds).ok_or_else(|| format!("{FRAME_WAIT_KINDS_VAR}={kinds:?} must be compute, graphics or all"))?;
        let required = |name: &str, what: &str| match get(name) {
            Some(v) => positive_ms(name, &v),
            None => Err(format!("{FRAME_WAIT_MS_VAR} is set and {FRAME_WAIT_KINDS_VAR} covers compute binds, so {name} ({what}) must be set too")),
        };
        let compute = if kinds.covers(true) {
            Some(ComputeBudget {
                frame: required(FRAME_WAIT_COMPUTE_MS_VAR, "the compute binds' own wait budget per frame, in ms")?,
                run_cap: required(COMPUTE_WAIT_TOTAL_MS_VAR, "how long compute binds may wait from their own budget over the whole run, in ms")?,
            })
        } else {
            None
        };
        let mut cutoff = DriverCutoff::default();
        for (name, compute_kind, slot) in [(FRAME_WAIT_COMPUTE_MS_VAR, true, None), (COMPUTE_WAIT_TOTAL_MS_VAR, true, None), (DRIVER_CUTOFF_GRAPHICS_VAR, false, Some(false)), (DRIVER_CUTOFF_COMPUTE_VAR, true, Some(true))] {
            let Some(v) = get(name) else { continue };
            if !kinds.covers(compute_kind) {
                return Err(format!("{name} is set, but {FRAME_WAIT_KINDS_VAR}={kinds:?} does not cover {} binds", if compute_kind { "compute" } else { "graphics" }));
            }
            match slot {
                Some(true) => cutoff.compute = Some(positive_ms(name, &v)?),
                Some(false) => cutoff.graphics = Some(positive_ms(name, &v)?),
                None => {}
            }
        }
        let mut f = FrameBudget::new(std::time::Duration::from_millis(ms), std::time::Duration::from_millis(window), kinds, now);
        f.compute = compute;
        f.cutoff = cutoff;
        Ok(Some(f))
    }

    fn new(limit: std::time::Duration, window: std::time::Duration, kinds: FrameWaitKinds, now: std::time::Instant) -> FrameBudget {
        let spend = FrameSpend {
            since: now,
            spent: std::time::Duration::ZERO,
            compute_spent: std::time::Duration::ZERO,
            run_compute_spent: std::time::Duration::ZERO,
            last: 0,
            reserved: HashMap::new(),
        };
        FrameBudget { limit, window, kinds, compute: None, cutoff: DriverCutoff::default(), spend: Mutex::new(spend) }
    }

    /// The budget is new: of the waits going on, only what is left of their reservations is
    /// charged to the new frame.
    fn renew(s: &mut FrameSpend, now: std::time::Instant, why: Renewal) {
        s.since = now;
        s.spent = std::time::Duration::ZERO;
        s.compute_spent = std::time::Duration::ZERO;
        for o in s.reserved.values_mut() {
            o.charged = o.grant.saturating_sub(now.saturating_duration_since(o.start));
            match o.pool {
                Pool::Frame => s.spent += o.charged,
                Pool::Compute => s.compute_spent += o.charged,
            }
        }
        match why {
            Renewal::Present => FRAME_STATS.renewed_by_present.fetch_add(1, Ordering::Relaxed),
            Renewal::Window => FRAME_STATS.renewed_by_window.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Renews the budget when its window has passed.
    fn spend_at(&self, now: std::time::Instant) -> std::sync::MutexGuard<'_, FrameSpend> {
        let mut s = self.spend.lock().unwrap();
        if now.saturating_duration_since(s.since) >= self.window {
            FrameBudget::renew(&mut s, now, Renewal::Window);
        }
        s
    }

    /// The budget a bind of this kind waits from now, and what is left of it.
    fn pool_left(&self, s: &FrameSpend, compute: bool) -> (Pool, std::time::Duration) {
        if let Some(c) = self.compute.filter(|_| compute) {
            let run_left = c.run_cap.saturating_sub(s.run_compute_spent);
            if !run_left.is_zero() {
                return (Pool::Compute, c.frame.saturating_sub(s.compute_spent).min(run_left));
            }
        }
        (Pool::Frame, self.limit.saturating_sub(s.spent))
    }

    /// (tests) What is left now for a bind of this kind.
    #[cfg(test)]
    fn left(&self, compute: bool, now: std::time::Instant) -> std::time::Duration {
        let s = self.spend_at(now);
        self.pool_left(&s, compute).1
    }

    /// Reserves a bind's wait before it waits: what is left for its kind, at most `want`.
    fn reserve(&self, compute: bool, want: Option<std::time::Duration>, now: std::time::Instant) -> Reservation {
        let mut s = self.spend_at(now);
        let (pool, left) = self.pool_left(&s, compute);
        let grant = want.map_or(left, |w| w.min(left));
        if grant.is_zero() {
            return Reservation { id: 0, pool, grant };
        }
        match pool {
            Pool::Frame => s.spent += grant,
            Pool::Compute => {
                s.compute_spent += grant;
                s.run_compute_spent += grant;
            }
        }
        s.last += 1;
        let id = s.last;
        s.reserved.insert(id, Outstanding { pool, start: now, grant, charged: grant });
        Reservation { id, pool, grant }
    }

    /// The bind waited `waited` of its reservation: what it did not use is given back.
    fn settle(&self, r: &Reservation, waited: std::time::Duration) {
        let mut s = self.spend.lock().unwrap();
        let Some(o) = s.reserved.remove(&r.id) else { return };
        let used = waited.min(o.grant);
        let in_frame = (o.start + used).saturating_duration_since(o.start.max(s.since));
        match o.pool {
            Pool::Frame => s.spent = s.spent.saturating_sub(o.charged) + in_frame,
            Pool::Compute => {
                s.compute_spent = s.compute_spent.saturating_sub(o.charged) + in_frame;
                s.run_compute_spent = s.run_compute_spent.saturating_sub(o.grant - used);
            }
        }
    }

    /// Of the compute budget's per-run cap: reserved or waited so far.
    fn run_compute_spent(&self) -> std::time::Duration {
        self.spend.lock().unwrap().run_compute_spent
    }

    /// The title presented a frame: the budget is new.
    fn presented(&self, now: std::time::Instant) {
        FrameBudget::renew(&mut self.spend.lock().unwrap(), now, Renewal::Present);
    }
}

/// What the frame budget did, for the log line every `FRAME_STATS_PERIOD`.
struct FrameStats {
    /// binds that waited for a build (within the budget)
    waits: std::sync::atomic::AtomicU64,
    waited_us: std::sync::atomic::AtomicU64,
    /// binds that went on with the stand-in: the budget was spent (or the driver cutoff passed)
    gave_up_compute: std::sync::atomic::AtomicU64,
    gave_up_graphics: std::sync::atomic::AtomicU64,
    draws_skipped: std::sync::atomic::AtomicU64,
    dispatches_skipped: std::sync::atomic::AtomicU64,
    renewed_by_present: std::sync::atomic::AtomicU64,
    renewed_by_window: std::sync::atomic::AtomicU64,
    /// of `waits`, `waited_us`: compute binds'
    compute_waits: std::sync::atomic::AtomicU64,
    compute_waited_us: std::sync::atomic::AtomicU64,
    /// binds that did not wait: the driver had been building the pipeline longer than the cutoff
    cutoff_graphics: std::sync::atomic::AtomicU64,
    cutoff_compute: std::sync::atomic::AtomicU64,
}

static FRAME_STATS: FrameStats = FrameStats {
    waits: std::sync::atomic::AtomicU64::new(0),
    waited_us: std::sync::atomic::AtomicU64::new(0),
    gave_up_compute: std::sync::atomic::AtomicU64::new(0),
    gave_up_graphics: std::sync::atomic::AtomicU64::new(0),
    draws_skipped: std::sync::atomic::AtomicU64::new(0),
    dispatches_skipped: std::sync::atomic::AtomicU64::new(0),
    renewed_by_present: std::sync::atomic::AtomicU64::new(0),
    renewed_by_window: std::sync::atomic::AtomicU64::new(0),
    compute_waits: std::sync::atomic::AtomicU64::new(0),
    compute_waited_us: std::sync::atomic::AtomicU64::new(0),
    cutoff_graphics: std::sync::atomic::AtomicU64::new(0),
    cutoff_compute: std::sync::atomic::AtomicU64::new(0),
};

impl FrameStats {
    fn snapshot(&self) -> [u64; 8] {
        [
            self.waits.load(Ordering::Relaxed),
            self.waited_us.load(Ordering::Relaxed),
            self.gave_up_compute.load(Ordering::Relaxed),
            self.gave_up_graphics.load(Ordering::Relaxed),
            self.draws_skipped.load(Ordering::Relaxed),
            self.dispatches_skipped.load(Ordering::Relaxed),
            self.renewed_by_present.load(Ordering::Relaxed),
            self.renewed_by_window.load(Ordering::Relaxed),
        ]
    }

    /// (compute waits, their µs, compute binds without their pipeline, dispatches skipped,
    /// graphics / compute cutoff skips)
    fn compute_snapshot(&self) -> [u64; 6] {
        [
            self.compute_waits.load(Ordering::Relaxed),
            self.compute_waited_us.load(Ordering::Relaxed),
            self.gave_up_compute.load(Ordering::Relaxed),
            self.dispatches_skipped.load(Ordering::Relaxed),
            self.cutoff_graphics.load(Ordering::Relaxed),
            self.cutoff_compute.load(Ordering::Relaxed),
        ]
    }
}

const FRAME_STATS_PERIOD: std::time::Duration = std::time::Duration::from_secs(2);

/// The frame budget from the environment (None: off). Panics on a malformed setting: it is read
/// first by `check_frame_budget`, when the D3D12 backend starts.
fn frame_budget() -> Option<&'static FrameBudget> {
    static F: OnceLock<Option<FrameBudget>> = OnceLock::new();
    F.get_or_init(|| match FrameBudget::parse(|name| std::env::var(name).ok(), std::time::Instant::now()) {
        Ok(f) => f,
        Err(e) => {
            crate::xlog_error!("D3D12.X: {e}");
            panic!("D3D12.X: {e}");
        }
    })
    .as_ref()
}

fn cutoff_text(c: Option<std::time::Duration>) -> String {
    c.map_or_else(|| "off".to_string(), |c| format!("{} ms", c.as_millis()))
}

/// Reads the frame budget's settings (a malformed one ends the process here, not at a bind
/// minutes later) and, when it is on, logs it and starts its log line every 2 s.
pub fn check_frame_budget() {
    let Some(f) = frame_budget() else { return };
    crate::xlog_info!(
        "D3D12.X: experimental frame wait budget: binds of {:?} pipelines still being built wait at most {} ms per frame in all (new at each PresentX, or after {} ms without one); past it their draws and dispatches are skipped",
        f.kinds,
        f.limit.as_millis(),
        f.window.as_millis()
    );
    if let Some(c) = f.compute {
        crate::xlog_info!(
            "D3D12.X: experimental compute wait budget: compute binds wait at most {} ms per frame from their own budget ({FRAME_WAIT_COMPUTE_MS_VAR}), {} ms in all over the run ({COMPUTE_WAIT_TOTAL_MS_VAR}); once that is spent they wait within {FRAME_WAIT_MS_VAR}",
            c.frame.as_millis(),
            c.run_cap.as_millis()
        );
    }
    if f.cutoff.any() {
        crate::xlog_info!(
            "D3D12.X: pipeline wait driver cutoff: a bind does not wait for a pipeline the driver has been building longer than {} (graphics) / {} (compute), and its draw or dispatch is skipped",
            cutoff_text(f.cutoff.graphics),
            cutoff_text(f.cutoff.compute)
        );
    }
    let _ = std::thread::Builder::new().name("xcl-pso-frame-wait".into()).spawn(move || {
        let mut last = FRAME_STATS.snapshot();
        let mut last_compute = (FRAME_STATS.compute_snapshot(), f.run_compute_spent());
        loop {
            std::thread::sleep(FRAME_STATS_PERIOD);
            let now = FRAME_STATS.snapshot();
            if now != last {
                let d: Vec<u64> = now.iter().zip(last.iter()).map(|(a, b)| a - b).collect();
                crate::xlog_info!(
                    "D3D12.X: frame wait budget, last 2 s: {} binds waited {} ms in all; {} compute / {} graphics binds went on without their pipeline; {} draws, {} dispatches skipped; renewed {} times by PresentX, {} by the window (totals: {} waits, {} ms, {} draws, {} dispatches skipped)",
                    d[0],
                    d[1] / 1000,
                    d[2],
                    d[3],
                    d[4],
                    d[5],
                    d[6],
                    d[7],
                    now[0],
                    now[1] / 1000,
                    now[4],
                    now[5]
                );
                last = now;
            }
            if f.compute.is_none() && !f.cutoff.any() {
                continue;
            }
            let compute = (FRAME_STATS.compute_snapshot(), f.run_compute_spent());
            if compute == last_compute {
                continue;
            }
            let (c, l) = (compute.0, last_compute.0);
            let cap = f.compute.map_or_else(|| "off".to_string(), |b| format!("{} of {} ms used", compute.1.as_millis(), b.run_cap.as_millis()));
            crate::xlog_info!(
                "D3D12.X: compute wait budget, last 2 s: {} compute binds waited {} ms in all; {} compute binds went on without their pipeline, {} dispatches skipped; {} graphics / {} compute binds skipped at once past the driver cutoff; per-run compute cap: {cap} (totals: {} compute waits, {} ms, {} compute binds without their pipeline, {} graphics / {} compute cutoff skips)",
                c[0] - l[0],
                (c[1] - l[1]) / 1000,
                c[2] - l[2],
                c[3] - l[3],
                c[4] - l[4],
                c[5] - l[5],
                c[0],
                c[1] / 1000,
                c[2],
                c[4],
                c[5]
            );
            last_compute = compute;
        }
    });
}

/// The title presented a frame (`PresentX`): the frame budget is new.
pub fn frame_presented() {
    if let Some(f) = frame_budget() {
        f.presented(std::time::Instant::now());
    }
}

/// A draw was skipped because the frame budget ran out before its pipeline was built.
pub fn note_skipped_draw() {
    FRAME_STATS.draws_skipped.fetch_add(1, Ordering::Relaxed);
}

/// A dispatch was skipped because the frame budget ran out before its pipeline was built.
pub fn note_skipped_dispatch() {
    FRAME_STATS.dispatches_skipped.fetch_add(1, Ordering::Relaxed);
}

/// The pipeline a worker builds on this thread has reached the driver's compiler
/// (`driver_compile` in real.rs): the driver cutoff counts from the first time it does.
pub(super) fn driver_build_starts() {
    BUILDING.with(|b| {
        if let Some(p) = b.borrow().as_ref() {
            p.driver_since.lock().unwrap().get_or_insert_with(std::time::Instant::now);
        }
    });
}

/// `resolve_wait` of a pipeline `frame` covers: the build goes to the workers (never to this
/// thread: a build here cannot be bounded) and is wanted; the bind reserves what is left of the
/// frame's budget for its kind (at most `per_pipeline`, `XCL_PSO_WAIT_MS`, and the time to the
/// driver cutoff), waits at most that and gives back the rest. True: not built by then – the
/// stand-in. A pipeline the driver has been building longer than the cutoff is not waited for
/// at all. Only `per_pipeline` running out makes the pipeline `late`; the frame budget does
/// not, so the next frame's binds wait for it again.
fn resolve_within_frame(p: &Arc<Pending>, pso: usize, per_pipeline: Option<std::time::Duration>, frame: &FrameBudget) -> (usize, bool) {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    };
    if *p.done.lock().unwrap() {
        return (built(p), false);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        let _ = queue().lock().unwrap().send(p.clone());
    }
    p.want();
    let gave_up = || {
        if p.compute {
            FRAME_STATS.gave_up_compute.fetch_add(1, Ordering::Relaxed);
        } else {
            FRAME_STATS.gave_up_graphics.fetch_add(1, Ordering::Relaxed);
        }
        (pso, true)
    };
    if p.late.load(Ordering::Acquire) {
        return gave_up();
    }
    let in_driver = p.driver_since.lock().unwrap().map(|t| t.elapsed());
    let to_cutoff = match (frame.cutoff.of(p.compute), in_driver) {
        (Some(cutoff), Some(in_driver)) if in_driver >= cutoff => {
            if p.compute {
                FRAME_STATS.cutoff_compute.fetch_add(1, Ordering::Relaxed);
            } else {
                FRAME_STATS.cutoff_graphics.fetch_add(1, Ordering::Relaxed);
            }
            return gave_up();
        }
        (Some(cutoff), Some(in_driver)) => Some(cutoff - in_driver),
        _ => None,
    };
    let want = match (per_pipeline, to_cutoff) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    let r = frame.reserve(p.compute, want, std::time::Instant::now());
    if p.compute && frame.compute.is_some() && r.pool == Pool::Frame {
        static CAP_SPENT: std::sync::Once = std::sync::Once::new();
        CAP_SPENT.call_once(|| {
            crate::xlog_info!("D3D12.X: the compute binds' per-run wait cap ({COMPUTE_WAIT_TOTAL_MS_VAR}) is spent: they wait within {FRAME_WAIT_MS_VAR} from now on");
        });
    }
    if r.grant.is_zero() {
        return gave_up();
    }
    let per_pipeline_bounds = per_pipeline == Some(r.grant);
    let start = std::time::Instant::now();
    let finished = {
        let _held = Held::new();
        let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), r.grant, |d| !*d).unwrap();
        *done
    };
    let waited = start.elapsed();
    frame.settle(&r, waited);
    FRAME_STATS.waits.fetch_add(1, Ordering::Relaxed);
    FRAME_STATS.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    if p.compute {
        FRAME_STATS.compute_waits.fetch_add(1, Ordering::Relaxed);
        FRAME_STATS.compute_waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
    }
    if finished {
        return (built(p), false);
    }
    if per_pipeline_bounds {
        p.late.store(true, Ordering::Release);
    }
    gave_up()
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
