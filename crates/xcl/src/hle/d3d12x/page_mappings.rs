//! Command queue CopyPageMappingsBatchX (the older layout's #29, the newer one's #31): the title
//! maps pages of a page pool into a GPU-only reservation. The argument layout and what XCL does
//! with it are in `docs/code/d3d12x-page-mappings.md`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const PAGE: u64 = 0x1_0000;
const BATCH_SIZE: usize = 0x18;
const RANGE_SIZE: usize = 0xC;
const MAX_PAGES_PER_BATCH: u64 = 1 << 20;
const S_OK: usize = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RangeKind {
    Consecutive,
    Repeated,
    Unmapped,
    Kept,
}

impl RangeKind {
    fn from_raw(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Consecutive),
            1 => Some(Self::Repeated),
            2 => Some(Self::Unmapped),
            3 => Some(Self::Kept),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Range {
    pub kind: RangeKind,
    pub pool_page: u32,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Batch {
    pub dst: u64,
    pub ranges: Vec<Range>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PageOp {
    Map { dst: u64, pool_page: u32 },
    Unmap { dst: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecodeError {
    NoBatches,
    NoRanges { batch: usize },
    UnalignedDestination { batch: usize, dst: u64 },
    UnknownKind { batch: usize, range: usize, kind: u32 },
    PoolPageOutOfRange { batch: usize, range: usize, page: u64, pool_pages: u32 },
    TooManyPages { batch: usize },
}

unsafe fn read_u32(p: *const u8, offset: usize) -> u32 {
    p.add(offset).cast::<u32>().read_unaligned()
}

unsafe fn read_u64(p: *const u8, offset: usize) -> u64 {
    p.add(offset).cast::<u64>().read_unaligned()
}

/// Reads `count` batches at `batches` (the title's memory).
pub(super) unsafe fn decode(count: u32, batches: *const u8) -> Result<Vec<Batch>, DecodeError> {
    if count == 0 || batches.is_null() {
        return Err(DecodeError::NoBatches);
    }
    let mut out = Vec::with_capacity(count as usize);
    for b in 0..count as usize {
        let at = batches.add(b * BATCH_SIZE);
        let dst = read_u64(at, 0);
        let n = read_u32(at, 8);
        let ranges = read_u64(at, 0x10) as *const u8;
        if dst % PAGE != 0 {
            return Err(DecodeError::UnalignedDestination { batch: b, dst });
        }
        if n != 0 && ranges.is_null() {
            return Err(DecodeError::NoRanges { batch: b });
        }
        let mut rs = Vec::with_capacity(n as usize);
        for r in 0..n as usize {
            let rp = ranges.add(r * RANGE_SIZE);
            let raw = read_u32(rp, 0);
            let kind = RangeKind::from_raw(raw).ok_or(DecodeError::UnknownKind { batch: b, range: r, kind: raw })?;
            rs.push(Range { kind, pool_page: read_u32(rp, 4), count: read_u32(rp, 8) });
        }
        out.push(Batch { dst, ranges: rs });
    }
    Ok(out)
}

/// The page-by-page changes of `batches` for a pool of `pool_pages` pages (kept pages give none).
pub(super) fn expand(batches: &[Batch], pool_pages: u32) -> Result<Vec<PageOp>, DecodeError> {
    let mut ops = Vec::new();
    for (b, batch) in batches.iter().enumerate() {
        let mut dst = batch.dst;
        let mut pages = 0u64;
        for (r, range) in batch.ranges.iter().enumerate() {
            pages += range.count as u64;
            if pages > MAX_PAGES_PER_BATCH {
                return Err(DecodeError::TooManyPages { batch: b });
            }
            let last = match range.kind {
                _ if range.count == 0 => None,
                RangeKind::Consecutive => Some(range.pool_page as u64 + range.count as u64 - 1),
                RangeKind::Repeated => Some(range.pool_page as u64),
                RangeKind::Unmapped | RangeKind::Kept => None,
            };
            if let Some(page) = last.filter(|&p| p >= pool_pages as u64) {
                return Err(DecodeError::PoolPageOutOfRange { batch: b, range: r, page, pool_pages });
            }
            for i in 0..range.count {
                let at = dst + i as u64 * PAGE;
                match range.kind {
                    RangeKind::Consecutive => ops.push(PageOp::Map { dst: at, pool_page: range.pool_page + i }),
                    RangeKind::Repeated => ops.push(PageOp::Map { dst: at, pool_page: range.pool_page }),
                    RangeKind::Unmapped => ops.push(PageOp::Unmap { dst: at }),
                    RangeKind::Kept => {}
                }
            }
            dst += range.count as u64 * PAGE;
        }
    }
    Ok(ops)
}

/// What the title's page-mapping batches left mapped: destination page → (pool base, pool page).
#[derive(Default)]
pub(super) struct Table {
    pages: BTreeMap<u64, (u64, u32)>,
}

impl Table {
    pub fn apply(&mut self, pool_base: u64, ops: &[PageOp]) {
        for op in ops {
            match *op {
                PageOp::Map { dst, pool_page } => {
                    self.pages.insert(dst, (pool_base, pool_page));
                }
                PageOp::Unmap { dst } => {
                    self.pages.remove(&dst);
                }
            }
        }
    }

    #[cfg(test)]
    pub fn at(&self, dst: u64) -> Option<(u64, u32)> {
        self.pages.get(&(dst & !(PAGE - 1))).copied()
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MirrorCheck {
    CpuViewHasPage,
    CpuViewEmpty,
    NotGpuView,
}

fn check_mirror(dst: u64) -> MirrorCheck {
    match crate::hle::kernelx::mirrors::source(dst as usize) {
        Some(cpu) if crate::hle::kernelx::physical_page_at(cpu) => MirrorCheck::CpuViewHasPage,
        Some(_) => MirrorCheck::CpuViewEmpty,
        None => MirrorCheck::NotGpuView,
    }
}

static CALLS: AtomicU64 = AtomicU64::new(0);
static MAPPED: AtomicU64 = AtomicU64::new(0);
static UNMAPPED: AtomicU64 = AtomicU64::new(0);
static CPU_VIEW_EMPTY: AtomicU64 = AtomicU64::new(0);
static NOT_GPU_VIEW: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);

fn log_summary(calls: u64, count: u32, pool_base: u64, pool_pages: u32) {
    if calls > 1 && !calls.is_power_of_two() {
        return;
    }
    let size = TABLE.lock().unwrap().as_ref().map_or(0, Table::len);
    crate::xlog_info!(
        "D3D12.X: CopyPageMappingsBatchX #{calls}: {count} batches, pool {pool_base:#x} of {pool_pages} pages; so far {} calls refused, {} pages mapped, {} unmapped, {size} mapped now; mapped pages whose CPU view had no page {}, outside a GPU view {}",
        REFUSED.load(Ordering::Relaxed),
        MAPPED.load(Ordering::Relaxed),
        UNMAPPED.load(Ordering::Relaxed),
        CPU_VIEW_EMPTY.load(Ordering::Relaxed),
        NOT_GPU_VIEW.load(Ordering::Relaxed),
    );
}

/// CopyPageMappingsBatchX(batch count, batches, pool base, pool page count). Returns 0 as the stub
/// did, refused calls too: the title ignores the result and its type is not known.
pub(super) unsafe fn copy_page_mappings_batch(count: u32, batches: *const u8, pool_base: u64, pool_pages: u32) -> usize {
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let ops = match decode(count, batches).and_then(|b| expand(&b, pool_pages)) {
        Ok(ops) => ops,
        Err(e) => {
            REFUSED.fetch_add(1, Ordering::Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: CopyPageMappingsBatchX({count} batches, pool {pool_base:#x} of {pool_pages} pages) refused: {e:?}"));
            log_summary(calls, count, pool_base, pool_pages);
            return S_OK;
        }
    };
    for op in &ops {
        match *op {
            PageOp::Map { dst, .. } => {
                MAPPED.fetch_add(1, Ordering::Relaxed);
                match check_mirror(dst) {
                    MirrorCheck::CpuViewHasPage => {}
                    MirrorCheck::CpuViewEmpty => {
                        CPU_VIEW_EMPTY.fetch_add(1, Ordering::Relaxed);
                    }
                    MirrorCheck::NotGpuView => {
                        NOT_GPU_VIEW.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            PageOp::Unmap { .. } => {
                UNMAPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    TABLE.lock().unwrap().get_or_insert_with(Table::default).apply(pool_base, &ops);
    log_summary(calls, count, pool_base, pool_pages);
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    const GARBAGE: u32 = 0xDEAD_BEEF;

    fn range_bytes(kind: u32, page: u32, count: u32) -> Vec<u8> {
        [kind, page, count].iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn batch_bytes(dst: u64, count: u32, ranges: *const u8) -> Vec<u8> {
        let mut b = Vec::with_capacity(BATCH_SIZE);
        b.extend_from_slice(&dst.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&GARBAGE.to_le_bytes());
        b.extend_from_slice(&(ranges as u64).to_le_bytes());
        b
    }

    fn ranges(list: &[(u32, u32, u32)]) -> Vec<u8> {
        list.iter().flat_map(|&(k, p, c)| range_bytes(k, p, c)).collect()
    }

    #[test]
    fn decodes_the_batch_and_range_layout() {
        let r0 = ranges(&[(1, 7, 1), (2, 0, 3)]);
        let r1 = ranges(&[(0, 4, 2), (3, 0, 5)]);
        let mut all = batch_bytes(0x20_0000_0000, 2, r0.as_ptr());
        all.extend(batch_bytes(0x20_0010_0000, 2, r1.as_ptr()));
        let b = unsafe { decode(2, all.as_ptr()) }.unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].dst, 0x20_0000_0000);
        assert_eq!(b[0].ranges, vec![Range { kind: RangeKind::Repeated, pool_page: 7, count: 1 }, Range { kind: RangeKind::Unmapped, pool_page: 0, count: 3 }]);
        assert_eq!(b[1].ranges, vec![Range { kind: RangeKind::Consecutive, pool_page: 4, count: 2 }, Range { kind: RangeKind::Kept, pool_page: 0, count: 5 }]);
    }

    #[test]
    fn every_range_kind_expands_page_by_page() {
        let base = 0x10_0000_0000u64;
        let batch = Batch {
            dst: base,
            ranges: vec![
                Range { kind: RangeKind::Consecutive, pool_page: 10, count: 3 },
                Range { kind: RangeKind::Repeated, pool_page: 5, count: 2 },
                Range { kind: RangeKind::Kept, pool_page: 99, count: 2 },
                Range { kind: RangeKind::Unmapped, pool_page: 99, count: 2 },
            ],
        };
        let ops = expand(&[batch], 16).unwrap();
        assert_eq!(
            ops,
            vec![
                PageOp::Map { dst: base, pool_page: 10 },
                PageOp::Map { dst: base + PAGE, pool_page: 11 },
                PageOp::Map { dst: base + 2 * PAGE, pool_page: 12 },
                PageOp::Map { dst: base + 3 * PAGE, pool_page: 5 },
                PageOp::Map { dst: base + 4 * PAGE, pool_page: 5 },
                PageOp::Unmap { dst: base + 7 * PAGE },
                PageOp::Unmap { dst: base + 8 * PAGE },
            ]
        );
    }

    #[test]
    fn the_table_follows_maps_and_unmaps() {
        let base = 0x10_0000_0000u64;
        let mut t = Table::default();
        t.apply(0x5000_0000, &[PageOp::Map { dst: base, pool_page: 3 }, PageOp::Map { dst: base + PAGE, pool_page: 4 }]);
        assert_eq!(t.at(base + 0x1234), Some((0x5000_0000, 3)));
        t.apply(0x5000_0000, &[PageOp::Unmap { dst: base }, PageOp::Map { dst: base + PAGE, pool_page: 9 }]);
        assert_eq!(t.at(base), None);
        assert_eq!(t.at(base + PAGE), Some((0x5000_0000, 9)));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn refuses_malformed_arguments() {
        assert_eq!(unsafe { decode(0, [0u8; BATCH_SIZE].as_ptr()) }, Err(DecodeError::NoBatches));
        assert_eq!(unsafe { decode(1, std::ptr::null()) }, Err(DecodeError::NoBatches));
        let unaligned = batch_bytes(0x10_0000_1000, 0, std::ptr::null());
        assert_eq!(unsafe { decode(1, unaligned.as_ptr()) }, Err(DecodeError::UnalignedDestination { batch: 0, dst: 0x10_0000_1000 }));
        let no_ranges = batch_bytes(0x10_0000_0000, 1, std::ptr::null());
        assert_eq!(unsafe { decode(1, no_ranges.as_ptr()) }, Err(DecodeError::NoRanges { batch: 0 }));
        let bad = ranges(&[(4, 0, 1)]);
        let bad_kind = batch_bytes(0x10_0000_0000, 1, bad.as_ptr());
        assert_eq!(unsafe { decode(1, bad_kind.as_ptr()) }, Err(DecodeError::UnknownKind { batch: 0, range: 0, kind: 4 }));
    }

    #[test]
    fn pool_pages_are_bounds_checked() {
        let one = |kind, pool_page, count| [Batch { dst: 0, ranges: vec![Range { kind, pool_page, count }] }];
        assert_eq!(expand(&one(RangeKind::Repeated, 8, 1), 8), Err(DecodeError::PoolPageOutOfRange { batch: 0, range: 0, page: 8, pool_pages: 8 }));
        assert_eq!(expand(&one(RangeKind::Consecutive, 6, 3), 8), Err(DecodeError::PoolPageOutOfRange { batch: 0, range: 0, page: 8, pool_pages: 8 }));
        assert_eq!(expand(&one(RangeKind::Consecutive, 6, 2), 8).unwrap().len(), 2);
        assert_eq!(expand(&one(RangeKind::Repeated, 99, 0), 8), Ok(vec![]));
        assert_eq!(expand(&one(RangeKind::Consecutive, 99, 0), 8), Ok(vec![]));
        assert_eq!(expand(&one(RangeKind::Unmapped, 99, 4), 8).unwrap().len(), 4);
        assert_eq!(expand(&one(RangeKind::Kept, u32::MAX, u32::MAX), 8), Err(DecodeError::TooManyPages { batch: 0 }));
    }

    #[test]
    fn a_call_lands_in_the_table_and_a_malformed_one_returns_0() {
        let (pool, dst) = (0x6600_0000_0000u64, 0x6610_0000_0000u64);
        let r = ranges(&[(1, 2, 1)]);
        let b = batch_bytes(dst, 1, r.as_ptr());
        assert_eq!(unsafe { copy_page_mappings_batch(1, b.as_ptr(), pool, 4) }, S_OK);
        assert_eq!(TABLE.lock().unwrap().as_ref().and_then(|t| t.at(dst)), Some((pool, 2)));
        let refused = REFUSED.load(Ordering::Relaxed);
        assert_eq!(unsafe { copy_page_mappings_batch(1, b.as_ptr(), pool, 2) }, S_OK);
        assert_eq!(REFUSED.load(Ordering::Relaxed), refused + 1);
    }

    #[test]
    fn a_batch_with_no_ranges_changes_nothing() {
        let empty = batch_bytes(0x10_0000_0000, 0, std::ptr::null());
        let b = unsafe { decode(1, empty.as_ptr()) }.unwrap();
        assert_eq!(expand(&b, 8), Ok(vec![]));
    }
}
