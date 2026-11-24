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

    #[test]
    fn the_frame_budget_is_off_unless_its_milliseconds_are_set() {
        let now = Instant::now();
        assert!(FrameBudget::parse(None, None, None, now).unwrap().is_none());
        assert!(FrameBudget::parse(Some("0"), None, None, now).unwrap().is_none());
        assert!(FrameBudget::parse(Some(" "), Some("1000"), Some("all"), now).unwrap().is_none());
    }

    #[test]
    fn a_frame_budget_without_its_window_or_kinds_is_an_error() {
        let now = Instant::now();
        let missing_window = FrameBudget::parse(Some("200"), None, Some("compute"), now).err().unwrap();
        assert!(missing_window.contains(FRAME_WAIT_WINDOW_MS_VAR), "{missing_window}");
        let zero_window = FrameBudget::parse(Some("200"), Some("0"), Some("compute"), now).err().unwrap();
        assert!(zero_window.contains(FRAME_WAIT_WINDOW_MS_VAR), "{zero_window}");
        let missing_kinds = FrameBudget::parse(Some("200"), Some("1000"), None, now).err().unwrap();
        assert!(missing_kinds.contains(FRAME_WAIT_KINDS_VAR), "{missing_kinds}");
        let wrong_kinds = FrameBudget::parse(Some("200"), Some("1000"), Some("pixel"), now).err().unwrap();
        assert!(wrong_kinds.contains(FRAME_WAIT_KINDS_VAR), "{wrong_kinds}");
        let wrong_ms = FrameBudget::parse(Some("lots"), Some("1000"), Some("all"), now).err().unwrap();
        assert!(wrong_ms.contains(FRAME_WAIT_MS_VAR), "{wrong_ms}");
        let f = FrameBudget::parse(Some("200"), Some("1000"), Some(" Graphics "), now).unwrap().unwrap();
        assert_eq!((f.limit, f.window, f.kinds), (MS * 200, MS * 1000, FrameWaitKinds::Graphics));
    }

    #[test]
    fn the_kinds_say_which_binds_the_budget_covers() {
        assert!(FrameWaitKinds::Compute.covers(true) && !FrameWaitKinds::Compute.covers(false));
        assert!(FrameWaitKinds::Graphics.covers(false) && !FrameWaitKinds::Graphics.covers(true));
        assert!(FrameWaitKinds::All.covers(true) && FrameWaitKinds::All.covers(false));
    }

    #[test]
    fn waits_are_charged_to_the_frame_until_a_present_renews_it() {
        let t0 = Instant::now();
        let f = budget(100, 60_000, FrameWaitKinds::All, t0);
        let (left, epoch) = f.left(t0);
        assert_eq!(left, MS * 100);
        f.charge(epoch, MS * 30);
        f.charge(epoch, MS * 50);
        assert_eq!(f.left(t0 + MS).0, MS * 20);
        f.charge(epoch, MS * 500);
        assert_eq!(f.left(t0 + MS).0, Duration::ZERO, "spent: nothing is left, not less");
        f.presented(t0 + MS * 2);
        let (left, renewed) = f.left(t0 + MS * 3);
        assert_eq!(left, MS * 100);
        f.charge(epoch, MS * 40);
        assert_eq!(f.left(t0 + MS * 3).0, MS * 100, "a wait that began before the present is not charged to the new frame");
        f.charge(renewed, MS * 40);
        assert_eq!(f.left(t0 + MS * 3).0, MS * 60);
    }

    #[test]
    fn the_budget_is_renewed_when_its_window_passes_without_a_present() {
        let t0 = Instant::now();
        let f = budget(100, 1_000, FrameWaitKinds::Compute, t0);
        let (_, epoch) = f.left(t0);
        f.charge(epoch, MS * 100);
        assert_eq!(f.left(t0 + MS * 999).0, Duration::ZERO);
        assert_eq!(f.left(t0 + MS * 1_000).0, MS * 100, "the window passed");
        let (_, epoch) = f.left(t0 + MS * 1_000);
        f.charge(epoch, MS * 100);
        assert_eq!(f.left(t0 + MS * 1_500).0, Duration::ZERO, "a new window starts at the renewal");
        f.presented(t0 + MS * 1_800);
        let (_, epoch) = f.left(t0 + MS * 1_800);
        f.charge(epoch, MS * 100);
        assert_eq!(f.left(t0 + MS * 2_700).0, Duration::ZERO, "a present starts a new window too");
        assert_eq!(f.left(t0 + MS * 2_800).0, MS * 100);
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
        assert!(f.left(Instant::now()).0 < MS * 5_000, "the wait was charged");
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
        assert_eq!(f.left(Instant::now()).0, MS * 10, "nothing charged");
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
/// Waits are charged after they end, so binds on several threads at once may wait more than the
/// budget in all (each waits at most what was left when it began).
pub struct FrameBudget {
    limit: std::time::Duration,
    window: std::time::Duration,
    kinds: FrameWaitKinds,
    spend: Mutex<FrameSpend>,
}

struct FrameSpend {
    /// when the budget was last new
    since: std::time::Instant,
    /// bumped at each renewal: a wait that began before it is not charged to the new frame
    epoch: u64,
    spent: std::time::Duration,
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

impl FrameBudget {
    /// The budget the three variables' values describe (None: `XCL_PSO_FRAME_WAIT_MS` unset or 0).
    fn parse(ms: Option<&str>, window_ms: Option<&str>, kinds: Option<&str>, now: std::time::Instant) -> Result<Option<FrameBudget>, String> {
        let Some(ms) = ms.map(str::trim).filter(|v| !v.is_empty()) else { return Ok(None) };
        let ms: u64 = ms.parse().map_err(|_| format!("{FRAME_WAIT_MS_VAR}={ms:?} is not a whole number of milliseconds"))?;
        if ms == 0 {
            return Ok(None);
        }
        let window = window_ms
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("{FRAME_WAIT_MS_VAR} is set, so {FRAME_WAIT_WINDOW_MS_VAR} (the wall time in ms after which the budget is new without a present) must be set too"))?;
        let window: u64 = window
            .parse()
            .ok()
            .filter(|&w| w > 0)
            .ok_or_else(|| format!("{FRAME_WAIT_WINDOW_MS_VAR}={window:?} must be a whole number of milliseconds above 0"))?;
        let kinds = kinds
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("{FRAME_WAIT_MS_VAR} is set, so {FRAME_WAIT_KINDS_VAR} (compute, graphics or all) must be set too"))?;
        let kinds = FrameWaitKinds::parse(kinds).ok_or_else(|| format!("{FRAME_WAIT_KINDS_VAR}={kinds:?} must be compute, graphics or all"))?;
        Ok(Some(FrameBudget::new(std::time::Duration::from_millis(ms), std::time::Duration::from_millis(window), kinds, now)))
    }

    fn new(limit: std::time::Duration, window: std::time::Duration, kinds: FrameWaitKinds, now: std::time::Instant) -> FrameBudget {
        FrameBudget { limit, window, kinds, spend: Mutex::new(FrameSpend { since: now, epoch: 0, spent: std::time::Duration::ZERO }) }
    }

    fn renew(s: &mut FrameSpend, now: std::time::Instant, why: Renewal) {
        s.since = now;
        s.epoch += 1;
        s.spent = std::time::Duration::ZERO;
        match why {
            Renewal::Present => FRAME_STATS.renewed_by_present.fetch_add(1, Ordering::Relaxed),
            Renewal::Window => FRAME_STATS.renewed_by_window.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// What is left of the budget now, and the epoch to charge a wait to (`charge`).
    fn left(&self, now: std::time::Instant) -> (std::time::Duration, u64) {
        let mut s = self.spend.lock().unwrap();
        if now.saturating_duration_since(s.since) >= self.window {
            FrameBudget::renew(&mut s, now, Renewal::Window);
        }
        (self.limit.saturating_sub(s.spent), s.epoch)
    }

    /// A bind waited `waited` from `epoch` on (not charged once the budget has been renewed since).
    fn charge(&self, epoch: u64, waited: std::time::Duration) {
        let mut s = self.spend.lock().unwrap();
        if s.epoch == epoch {
            s.spent = s.spent.saturating_add(waited);
        }
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
    /// binds that went on with the stand-in: the budget was spent
    gave_up_compute: std::sync::atomic::AtomicU64,
    gave_up_graphics: std::sync::atomic::AtomicU64,
    draws_skipped: std::sync::atomic::AtomicU64,
    dispatches_skipped: std::sync::atomic::AtomicU64,
    renewed_by_present: std::sync::atomic::AtomicU64,
    renewed_by_window: std::sync::atomic::AtomicU64,
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
}

const FRAME_STATS_PERIOD: std::time::Duration = std::time::Duration::from_secs(2);

/// The frame budget from the environment (None: off). Panics on a malformed setting: it is read
/// first by `check_frame_budget`, when the D3D12 backend starts.
fn frame_budget() -> Option<&'static FrameBudget> {
    static F: OnceLock<Option<FrameBudget>> = OnceLock::new();
    F.get_or_init(|| {
        let var = |name: &str| std::env::var(name).ok();
        match FrameBudget::parse(var(FRAME_WAIT_MS_VAR).as_deref(), var(FRAME_WAIT_WINDOW_MS_VAR).as_deref(), var(FRAME_WAIT_KINDS_VAR).as_deref(), std::time::Instant::now()) {
            Ok(f) => f,
            Err(e) => {
                crate::xlog_error!("D3D12.X: {e}");
                panic!("D3D12.X: {e}");
            }
        }
    })
    .as_ref()
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
    let _ = std::thread::Builder::new().name("xcl-pso-frame-wait".into()).spawn(|| {
        let mut last = FRAME_STATS.snapshot();
        loop {
            std::thread::sleep(FRAME_STATS_PERIOD);
            let now = FRAME_STATS.snapshot();
            if now == last {
                continue;
            }
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

/// `resolve_wait` of a pipeline `frame` covers: the build goes to the workers (never to this
/// thread: a build here cannot be bounded) and is wanted; the bind waits at most what is left of
/// the frame's budget (and of `per_pipeline`, `XCL_PSO_WAIT_MS`). True: not built by then – the
/// stand-in. Only `per_pipeline` running out makes the pipeline `late`; the frame budget does
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
    let (left, epoch) = frame.left(std::time::Instant::now());
    let (wait, per_pipeline_bounds) = match per_pipeline {
        Some(b) if b <= left => (b, true),
        _ => (left, false),
    };
    if wait.is_zero() {
        return gave_up();
    }
    let start = std::time::Instant::now();
    let finished = {
        let _held = Held::new();
        let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), wait, |d| !*d).unwrap();
        *done
    };
    let waited = start.elapsed();
    frame.charge(epoch, waited);
    FRAME_STATS.waits.fetch_add(1, Ordering::Relaxed);
    FRAME_STATS.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
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
