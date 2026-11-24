//! `XCL_EMPTY_VIEW_STATS=1`: counts of the views in a console heap that get an empty
//! descriptor (num_records 0, or a T# of zeros), so the shader reads zeros where the title
//! meant a resource – by cause, with the bytes they stand for and the first few of each. A
//! summary goes to the log every 2 s while anything changed. Off by default: then every
//! `note` is one cached flag.
//!
//! What is known at view creation is the resource and its description and the heap slot; the
//! shader stage and register that read the slot are not.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Why a view got an empty descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// an SRV of a buffer that is not in unified memory (`real::console_buffer`)
    SrvBufferNotInUma,
    /// a UAV of a buffer that is not in unified memory
    UavBufferNotInUma,
    /// a CBV whose address is not in unified memory
    CbvNotInUma,
    /// an SRV of a texture whose view `real::view_shape` cannot describe
    SrvShapeUnknown,
    /// a UAV of a texture whose view `real::view_shape` cannot describe
    UavShapeUnknown,
    /// an SRV with no resource (a null view the title asked for)
    SrvNoResource,
    /// a UAV with no resource (a null view the title asked for)
    UavNoResource,
    /// a raytracing acceleration structure SRV whose address is not in unified memory
    BvhNotInUma,
}

const CAUSES: [Cause; 8] = [
    Cause::SrvBufferNotInUma,
    Cause::UavBufferNotInUma,
    Cause::CbvNotInUma,
    Cause::SrvShapeUnknown,
    Cause::UavShapeUnknown,
    Cause::SrvNoResource,
    Cause::UavNoResource,
    Cause::BvhNotInUma,
];

/// How many views of each cause are described in the log.
const SAMPLES_PER_CAUSE: usize = 6;

const PERIOD: std::time::Duration = std::time::Duration::from_secs(2);

/// One view that got an empty descriptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    /// the PC resource (raw pointer, 0 = none)
    pub res: usize,
    /// its address where one is known without side effects (a CBV's, a BVH's), else 0
    pub va: u64,
    /// the bytes the view stands for (buffers: the view's or the resource's size; textures:
    /// texels × format size of the resource, approximate)
    pub bytes: u64,
    /// the resource's flags
    pub flags: u32,
    /// the view's DXGI format (the resource's when the view gives none)
    pub format: u32,
    /// the view dimension (D3D12_*_DIMENSION), 0 when not known
    pub dimension: i32,
    /// the slot's PC heap index (`console_heap::Slot::index`)
    pub slot: u32,
}

struct Tally {
    count: AtomicU64,
    bytes: AtomicU64,
    samples: Mutex<Vec<Sample>>,
}

static TALLIES: [Tally; CAUSES.len()] = [const { Tally { count: AtomicU64::new(0), bytes: AtomicU64::new(0), samples: Mutex::new(Vec::new()) } }; CAUSES.len()];

fn tally(cause: Cause) -> &'static Tally {
    &TALLIES[CAUSES.iter().position(|&c| c == cause).expect("every cause is in CAUSES")]
}

pub fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("XCL_EMPTY_VIEW_STATS").is_ok_and(|v| v.trim() == "1"))
}

/// A view got an empty descriptor for `cause` (`sample` describes it; called only when on).
pub fn note(cause: Cause, sample: impl FnOnce() -> Sample) {
    if !enabled() {
        return;
    }
    record(cause, sample());
    start_reporter();
}

fn record(cause: Cause, s: Sample) {
    let t = tally(cause);
    t.count.fetch_add(1, Ordering::Relaxed);
    t.bytes.fetch_add(s.bytes, Ordering::Relaxed);
    let mut g = t.samples.lock().unwrap();
    if g.len() < SAMPLES_PER_CAUSE {
        g.push(s);
    }
}

fn start_reporter() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-empty-views".into()).spawn(|| {
            let mut last = [0u64; CAUSES.len()];
            loop {
                std::thread::sleep(PERIOD);
                let now: Vec<u64> = TALLIES.iter().map(|t| t.count.load(Ordering::Relaxed)).collect();
                if now[..] == last[..] {
                    continue;
                }
                for line in summary(&last) {
                    crate::xlog_info!("{line}");
                }
                last.copy_from_slice(&now);
            }
        });
    });
}

/// The log lines of a summary (`last`: the counts of the summary before).
fn summary(last: &[u64; CAUSES.len()]) -> Vec<String> {
    let mut lines = vec!["D3D12.X: views with an empty descriptor (the shader reads zeros), by cause:".to_string()];
    for (k, cause) in CAUSES.iter().enumerate() {
        let t = &TALLIES[k];
        let count = t.count.load(Ordering::Relaxed);
        if count == 0 {
            continue;
        }
        let samples: Vec<String> = t
            .samples
            .lock()
            .unwrap()
            .iter()
            .map(|s| format!("res {:#x} va {:#x} {} bytes flags {:#x} format {} dim {} slot {}", s.res, s.va, s.bytes, s.flags, s.format, s.dimension, s.slot))
            .collect();
        lines.push(format!(
            "  {cause:?}: {count} (+{} in 2 s), {} MB; first: {}",
            count - last[k],
            t.bytes.load(Ordering::Relaxed) >> 20,
            samples.join("; ")
        ));
    }
    lines
}

/// Bytes of a texture of `d`: texels × the size of `format` (its gfx10 format; 4 bytes when
/// there is none, as for a compressed one).
pub fn texture_bytes(d: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT) -> u64 {
    const UNKNOWN_FORMAT_BYTES: u64 = 4;
    let per = crate::gpu::desc::gfx10_format(format).map_or(UNKNOWN_FORMAT_BYTES, |f| crate::gpu::desc::format_bytes(f).max(1) as u64);
    d.Width.saturating_mul(d.Height as u64).saturating_mul(d.DepthOrArraySize.max(1) as u64).saturating_mul(per)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_cause_is_counted_with_its_bytes_and_first_views() {
        let before = tally(Cause::BvhNotInUma).count.load(Ordering::Relaxed);
        let bytes_before = tally(Cause::BvhNotInUma).bytes.load(Ordering::Relaxed);
        let other_before = tally(Cause::UavNoResource).count.load(Ordering::Relaxed);
        for k in 0..(SAMPLES_PER_CAUSE as u64 + 3) {
            record(Cause::BvhNotInUma, Sample { va: 0x1000 + k, bytes: 1 << 20, ..Default::default() });
        }
        let t = tally(Cause::BvhNotInUma);
        assert_eq!(t.count.load(Ordering::Relaxed) - before, SAMPLES_PER_CAUSE as u64 + 3);
        assert_eq!(t.bytes.load(Ordering::Relaxed) - bytes_before, (SAMPLES_PER_CAUSE as u64 + 3) << 20);
        assert_eq!(t.samples.lock().unwrap().len(), SAMPLES_PER_CAUSE, "only the first few are kept");
        assert_eq!(tally(Cause::UavNoResource).count.load(Ordering::Relaxed), other_before, "other causes are apart");
        let lines = summary(&[0; CAUSES.len()]);
        assert!(lines.iter().any(|l| l.contains("BvhNotInUma") && l.contains("va 0x1000")), "{lines:?}");
    }
}
