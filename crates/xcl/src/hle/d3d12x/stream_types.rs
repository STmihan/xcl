//! Pipeline state stream subobject numbering: the PC's and the console D3D12.X's
//! (docs/code/d3d12x-pipeline-stream.md).
//!
//! * `XCL_TRACE_PSO_STREAM=1`: one line per distinct stream given to CreatePipelineState – each
//!   subobject's offset, type, payload size and hash (shaders: stage and blob hash, no bytes).
//! * `XCL_CONSOLE_STREAM_TYPES=1`: a stream in the console's numbering (its MS container in type
//!   24, the PC's AS) is rewritten in the PC's before anything reads it.

use std::collections::HashSet;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::{Mutex, OnceLock};

/// The console's mesh shader subobject (the PC's AS).
const CONSOLE_MS: u32 = 24;

/// Console types read as the PC's: in retail mesh streams read with the PC's payload sizes,
/// every subobject up to the end was one of these (a shift would break that walk).
const CONSOLE_SAME_AS_PC: Range<u32> = 0..23;

/// Console type → the PC's, where they differ. Inferred, not measured: the PC leaves 23 out
/// (VIEW_INSTANCING 22, AS 24, MS 25); the console is read as AS 23, MS 24. Console types from
/// 25 on are not known and leave a stream as it is.
const CONSOLE_TO_PC: [(u32, u32); 2] = [(23, 24), (24, 25)];

/// Types whose payload holds a title pointer (root signature, shaders, stream output, input
/// layout, cached PSO, view instancing): their bytes differ every run, so are not hashed.
const POINTER_PAYLOADS: [u32; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 12, 19, 22, 24, 25];

pub(super) fn console_to_pc(ty: u32) -> Option<u32> {
    if CONSOLE_SAME_AS_PC.contains(&ty) {
        return Some(ty);
    }
    CONSOLE_TO_PC.iter().find(|m| m.0 == ty).map(|m| m.1)
}

fn pc_payload(ty: u32) -> Option<(usize, usize)> {
    super::null::subobject_payload(ty)
}

/// A console type's payload is its PC type's (AS and MS: a D3D12_SHADER_BYTECODE either way).
fn console_payload(ty: u32) -> Option<(usize, usize)> {
    pc_payload(console_to_pc(ty)?)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Sub {
    pub off: usize,
    pub ty: u32,
    pub payload: Range<usize>,
}

/// A stream read with one numbering: its subobjects, and where the walk stopped (offset, type)
/// when a type is not known or its payload passes the end.
#[derive(Debug, Clone)]
pub(super) struct Walk {
    pub subs: Vec<Sub>,
    pub stop: Option<(usize, u32)>,
    pub end: usize,
}

pub(super) fn walk(buf: &[u8], payload: impl Fn(u32) -> Option<(usize, usize)>) -> Walk {
    let mut subs = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = payload(ty) else { return Walk { subs, stop: Some((off, ty)), end: off } };
        let at = off + if palign == 8 { 8 } else { 4 };
        if at + psize > buf.len() {
            return Walk { subs, stop: Some((off, ty)), end: off };
        }
        subs.push(Sub { off, ty, payload: at..at + psize });
        off += (at - off + psize + 7) & !7;
    }
    Walk { subs, stop: None, end: off }
}

pub(super) fn walk_pc(buf: &[u8]) -> Walk {
    walk(buf, pc_payload)
}

pub(super) fn walk_console(buf: &[u8]) -> Walk {
    walk(buf, console_payload)
}

/// The stream in the console's numbering: it reads to the end with it, and its console MS
/// subobject holds a mesh shader container (`is_ms`) – in the PC's numbering that is the AS
/// slot, which no PC stream fills with a mesh shader.
pub(super) fn console_numbered(buf: &[u8], is_ms: impl Fn(&[u8]) -> bool) -> Option<Walk> {
    let w = walk_console(buf);
    if w.stop.is_some() {
        return None;
    }
    let ms = w.subs.iter().find(|s| s.ty == CONSOLE_MS)?;
    is_ms(&buf[ms.payload.clone()]).then_some(w)
}

/// The stream with its types in the PC's numbering (payloads and offsets unchanged).
pub(super) fn to_pc(buf: &[u8], w: &Walk) -> Vec<u8> {
    let mut out = buf.to_vec();
    for s in &w.subs {
        let pc = console_to_pc(s.ty).expect("a console-numbered walk has only known types");
        assert_eq!(console_payload(s.ty), pc_payload(pc), "console type {} and PC type {pc} differ in payload layout", s.ty);
        out[s.off..s.off + 4].copy_from_slice(&pc.to_le_bytes());
    }
    out
}

fn env_on(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty() && v != "0")
}

fn trace_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("XCL_TRACE_PSO_STREAM"))
}

fn remap_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("XCL_CONSOLE_STREAM_TYPES"))
}

/// A D3D12_SHADER_BYTECODE payload's code, when it points at readable memory.
unsafe fn bytecode<'a>(p: &[u8]) -> Option<&'a [u8]> {
    let ptr = u64::from_le_bytes(p.get(0..8)?.try_into().ok()?) as usize;
    let len = u64::from_le_bytes(p.get(8..16)?.try_into().ok()?) as usize;
    crate::gpu::shader::readable(ptr, len).then(|| std::slice::from_raw_parts(ptr as *const u8, len))
}

/// The API stage of a console shader container ("MS", "PS", …).
fn console_stage(code: &[u8]) -> Option<String> {
    crate::gpu::recomp::pso::is_console(code).then(|| crate::gpu::xbox::parse(code).map_or_else(|| "console".to_string(), |sh| sh.stage))
}

unsafe fn is_ms_payload(p: &[u8]) -> bool {
    bytecode(p).and_then(console_stage).is_some_and(|s| s == "MS")
}

/// A stream rewritten in the PC's numbering, with its own desc.
pub(super) struct PcStream {
    _buf: Vec<u8>,
    desc: Box<[usize; 2]>,
}

impl PcStream {
    pub(super) fn desc(&self) -> *const c_void {
        &*self.desc as *const _ as *const c_void
    }
}

/// CreatePipelineState's stream: traced (`XCL_TRACE_PSO_STREAM`), and when it is in the
/// console's numbering and `XCL_CONSOLE_STREAM_TYPES` is set, the PC-numbered copy to read and
/// create instead.
pub(super) unsafe fn prepare(desc: *const c_void) -> Option<PcStream> {
    if !trace_on() && !remap_on() {
        return None;
    }
    if !crate::gpu::shader::readable(desc as usize, 16) {
        return None;
    }
    let (size, base) = (*(desc as *const usize), *((desc as usize + 8) as *const usize));
    if !crate::gpu::shader::readable(base, size) {
        return None;
    }
    let buf = std::slice::from_raw_parts(base as *const u8, size);
    let console = console_numbered(buf, |p| is_ms_payload(p));
    if trace_on() {
        trace(buf, console.as_ref());
    }
    if !remap_on() {
        return None;
    }
    let w = console?;
    let out = to_pc(buf, &w);
    note_remapped(buf, &w);
    let desc = Box::new([out.len(), out.as_ptr() as usize]);
    Some(PcStream { _buf: out, desc })
}

fn types_of(w: &Walk) -> Vec<u32> {
    w.subs.iter().map(|s| s.ty).collect()
}

unsafe fn shader_hashes(buf: &[u8], w: &Walk, numbering: fn(u32) -> Option<u32>) -> String {
    w.subs
        .iter()
        .filter(|s| numbering(s.ty).is_some_and(|t| (1..=6).contains(&t) || t == 24 || t == 25))
        .filter_map(|s| bytecode(&buf[s.payload.clone()]).map(|c| format!("{}:{:016x}", s.ty, crate::gpu::shader::fnv64(c))))
        .collect::<Vec<_>>()
        .join(" ")
}

/// One line per console-numbered pipeline taken in the PC's numbering.
unsafe fn note_remapped(buf: &[u8], w: &Walk) {
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let console = types_of(w);
    let pc: Vec<u32> = console.iter().filter_map(|&t| console_to_pc(t)).collect();
    let ms = w.subs.iter().find(|s| s.ty == CONSOLE_MS).and_then(|s| bytecode(&buf[s.payload.clone()])).map_or(0, crate::gpu::shader::fnv64);
    let shaders = shader_hashes(buf, w, console_to_pc);
    let key = crate::gpu::shader::fnv64(format!("{console:?} {shaders}").as_bytes());
    if !SEEN.lock().unwrap().get_or_insert_with(HashSet::new).insert(key) {
        return;
    }
    crate::xlog_info!("D3D12.X: console-numbered pipeline stream, MS {ms:016x}: types {console:?} read as the PC's {pc:?} (shaders {shaders})");
}

fn walk_summary(w: &Walk, size: usize) -> String {
    match w.stop {
        Some((off, ty)) => format!("stops at {off:#x} on type {ty}"),
        None if w.end == size => format!("ok to {size:#x}"),
        None => format!("ok to {:#x} of {size:#x}", w.end),
    }
}

/// What the PC's reading of a stream holds.
unsafe fn kind(buf: &[u8], pc: &Walk) -> &'static str {
    let shader = |ty: u32| pc.subs.iter().find(|s| s.ty == ty).and_then(|s| bytecode(&buf[s.payload.clone()]));
    let stage = |ty: u32| shader(ty).and_then(console_stage);
    if stage(24).is_some_and(|s| s == "MS") && shader(25).is_none() {
        "MS in the AS slot"
    } else if shader(25).is_some() {
        if stage(25).is_some() { "console mesh" } else { "PC mesh" }
    } else if shader(1).is_some() {
        if stage(1).is_some() { "console graphics" } else { "PC graphics" }
    } else if shader(6).is_some() {
        if stage(6).is_some() { "console compute" } else { "PC compute" }
    } else {
        "other"
    }
}

/// A subobject as the trace shows it: offset, type, payload size, then a shader's stage, size
/// and hash, a few state values, or the payload's hash.
unsafe fn describe(buf: &[u8], s: &Sub, pc_ty: Option<u32>) -> String {
    let p = &buf[s.payload.clone()];
    let head = format!("{:#x}:{}({})", s.off, s.ty, p.len());
    let word = |k: usize| p.get(4 * k..4 * k + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    match pc_ty {
        Some(t) if (1..=6).contains(&t) || t == 24 || t == 25 => match bytecode(p) {
            Some(c) => format!("{head} {} {}B {:016x}", console_stage(c).unwrap_or_else(|| "pc".into()), c.len(), crate::gpu::shader::fnv64(c)),
            None => format!("{head} null"),
        },
        Some(14) => format!("{head} topology {}", word(0)),
        Some(15) => format!("{head} rt {} {:?}", word(8), (0..8).map(word).collect::<Vec<_>>()),
        Some(16) => format!("{head} ds {}", word(0)),
        Some(17) => format!("{head} samples {}/{}", word(0), word(1)),
        Some(t) if POINTER_PAYLOADS.contains(&t) => format!("{head} {}", if p.iter().any(|&b| b != 0) { "ptr" } else { "zero" }),
        _ => format!("{head} {:016x}", crate::gpu::shader::fnv64(p)),
    }
}

/// `XCL_TRACE_PSO_STREAM`: one line per distinct stream (no shader bytes).
unsafe fn trace(buf: &[u8], console: Option<&Walk>) {
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let pc = walk_pc(buf);
    let cw = walk_console(buf);
    let (reading, w, numbering): (&str, &Walk, fn(u32) -> Option<u32>) = match console {
        Some(w) => ("console", w, console_to_pc),
        None if pc.stop.is_none() => ("pc", &pc, Some),
        None => ("console", &cw, console_to_pc),
    };
    let subs = w.subs.iter().map(|s| describe(buf, s, numbering(s.ty))).collect::<Vec<_>>().join(", ");
    let line = format!(
        "{} B [{}], PC walk {}, console walk {}; read as {reading}: {subs}",
        buf.len(),
        kind(buf, &pc),
        walk_summary(&pc, buf.len()),
        walk_summary(&cw, buf.len())
    );
    if !SEEN.lock().unwrap().get_or_insert_with(HashSet::new).insert(crate::gpu::shader::fnv64(line.as_bytes())) {
        return;
    }
    crate::xlog_info!("D3D12.X: PSO stream {line}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::xbox::tests::{fssm, sample, CODE};

    /// DXIL program kinds (`xbox::DXIL_KINDS`)
    const KIND_PS: u32 = 0;
    const KIND_MS: u32 = 13;
    const KIND_AS: u32 = 14;
    /// FSSM hardware / API stage pairs of the synthetic containers
    const NGG_MS: (u8, u8) = (2, 13);
    const PS_PS: (u8, u8) = (5, 1);

    fn container(kind: u32, stage: (u8, u8)) -> Vec<u8> {
        sample(kind, &[fssm(&CODE, &[stage])])
    }

    #[derive(Default)]
    struct Stream(Vec<u8>);

    impl Stream {
        fn raw(&mut self, ty: u32, payload: &[u8], align8: bool) -> &mut Self {
            let at = self.0.len();
            self.0.extend_from_slice(&ty.to_le_bytes());
            self.0.resize(at + if align8 { 8 } else { 4 }, 0);
            self.0.extend_from_slice(payload);
            let end = (self.0.len() + 7) & !7;
            self.0.resize(end, 0);
            self
        }
        fn shader(&mut self, ty: u32, code: &[u8]) -> &mut Self {
            let mut p = (code.as_ptr() as u64).to_le_bytes().to_vec();
            p.extend_from_slice(&(code.len() as u64).to_le_bytes());
            self.raw(ty, &p, true)
        }
        fn word(&mut self, ty: u32, v: u32) -> &mut Self {
            self.raw(ty, &v.to_le_bytes(), false)
        }
    }

    const ROOT_SIGNATURE: u32 = 0;
    const PS: u32 = 2;
    const BLEND: u32 = 8;
    const RASTERIZER: u32 = 10;
    const RENDER_TARGET_FORMATS: u32 = 15;
    const DEPTH_STENCIL_FORMAT: u32 = 16;
    const PC_AS: u32 = 24;
    const PC_MS: u32 = 25;
    const RS_POINTER: u64 = 0x1234_5678_9abc_def0;
    const R8G8B8A8_UNORM: u32 = 28;
    const D32_FLOAT: u32 = 40;

    fn rt_formats() -> Vec<u8> {
        let mut p: Vec<u8> = [R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0].iter().flat_map(|f| f.to_le_bytes()).collect();
        p.extend_from_slice(&1u32.to_le_bytes());
        p
    }

    fn stream(ms_type: u32, ms: &[u8], ps: &[u8]) -> Vec<u8> {
        let mut s = Stream::default();
        s.raw(ROOT_SIGNATURE, &RS_POINTER.to_le_bytes(), true)
            .shader(ms_type, ms)
            .shader(PS, ps)
            .raw(BLEND, &[0; 328], false)
            .raw(RASTERIZER, &[0; 44], false)
            .raw(RENDER_TARGET_FORMATS, &rt_formats(), false)
            .word(DEPTH_STENCIL_FORMAT, D32_FLOAT);
        s.0
    }

    fn is_ms(p: &[u8]) -> bool {
        unsafe { is_ms_payload(p) }
    }

    fn payload_of<'a>(buf: &'a [u8], w: &Walk, ty: u32) -> &'a [u8] {
        &buf[w.subs.iter().find(|s| s.ty == ty).unwrap().payload.clone()]
    }

    #[test]
    fn console_stream_reads_as_the_pcs() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let console = stream(CONSOLE_MS, &ms, &ps);
        let w = console_numbered(&console, is_ms).expect("the MS container in type 24 and nothing the console numbering does not know");
        let pc = to_pc(&console, &w);
        let r = walk_pc(&pc);
        assert_eq!(r.stop, None);
        assert_eq!(r.end, pc.len());
        assert_eq!(types_of(&r), vec![ROOT_SIGNATURE, PC_MS, PS, BLEND, RASTERIZER, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT]);
        assert!(!r.subs.iter().any(|s| s.ty == PC_AS));
        let code = |ty| unsafe { bytecode(payload_of(&pc, &r, ty)) }.unwrap();
        assert_eq!(code(PC_MS), &ms[..]);
        assert_eq!(code(PS), &ps[..]);
        assert_eq!(console_stage(code(PC_MS)).as_deref(), Some("MS"));
        assert_eq!(console_stage(code(PS)).as_deref(), Some("PS"));
        assert_eq!(payload_of(&pc, &r, ROOT_SIGNATURE), RS_POINTER.to_le_bytes());
        assert_eq!(payload_of(&pc, &r, RENDER_TARGET_FORMATS), &rt_formats()[..]);
        assert_eq!(payload_of(&pc, &r, DEPTH_STENCIL_FORMAT), D32_FLOAT.to_le_bytes());
        assert_eq!(pc.len(), console.len());
        let changed: Vec<usize> = (0..pc.len()).filter(|&i| pc[i] != console[i]).collect();
        let ms_off = w.subs.iter().find(|s| s.ty == CONSOLE_MS).unwrap().off;
        assert_eq!(changed, vec![ms_off], "only the MS subobject's type changes (24 → 25)");
    }

    #[test]
    fn console_amplification_reads_as_the_pcs() {
        let (amp, ms, ps) = (container(KIND_AS, NGG_MS), container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(CONSOLE_MS, &ms, &ps));
        s.shader(23, &amp);
        assert_eq!(walk_pc(&s.0).stop.map(|x| x.1), Some(23), "the PC's numbering has no 23");
        let w = console_numbered(&s.0, is_ms).expect("console-numbered");
        let r = walk_pc(&to_pc(&s.0, &w));
        assert_eq!(r.stop, None);
        assert_eq!(types_of(&r), vec![ROOT_SIGNATURE, PC_MS, PS, BLEND, RASTERIZER, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT, PC_AS]);
    }

    #[test]
    fn pc_mesh_stream_is_left_alone() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        assert!(console_numbered(&stream(PC_MS, &ms, &ps), is_ms).is_none(), "a PC MS (25) is not the console's numbering");
    }

    #[test]
    fn pc_amplification_and_mesh_stream_is_left_alone() {
        let (amp, ms, ps) = (container(KIND_AS, NGG_MS), container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(PC_MS, &ms, &ps));
        s.shader(PC_AS, &amp);
        assert!(console_numbered(&s.0, is_ms).is_none());
    }

    #[test]
    fn non_mesh_container_in_24_is_left_alone() {
        let (amp, ps) = (container(KIND_AS, NGG_MS), container(KIND_PS, PS_PS));
        assert!(console_numbered(&stream(PC_AS, &amp, &ps), is_ms).is_none(), "an AS container in 24 is the PC's AS");
        assert!(console_numbered(&stream(PC_AS, &[0x44; 64], &ps), is_ms).is_none(), "a PC blob in 24 is the PC's AS");
    }

    #[test]
    fn console_types_from_25_on_are_unknown() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(CONSOLE_MS, &ms, &ps));
        s.raw(26, &[0; 60], false);
        assert!(console_numbered(&s.0, is_ms).is_none());
        assert_eq!((25..40).filter_map(console_to_pc).count(), 0);
        assert!((0..23).all(|t| console_to_pc(t) == Some(t)));
    }

    #[test]
    fn remapped_payloads_keep_their_layout() {
        for (console, pc) in CONSOLE_TO_PC {
            assert_eq!(console_payload(console), pc_payload(pc));
            assert_eq!(pc_payload(pc), Some((16, 8)));
        }
    }

    #[test]
    fn walk_reports_where_it_stops() {
        let mut s = Stream::default();
        s.word(DEPTH_STENCIL_FORMAT, D32_FLOAT).word(99, 0);
        let w = walk_pc(&s.0);
        assert_eq!(w.stop, Some((8, 99)));
        assert_eq!(types_of(&w), vec![DEPTH_STENCIL_FORMAT]);
    }
}
