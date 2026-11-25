//! Pipeline state stream subobject numbering: the PC's and the console D3D12.X's
//! (docs/code/d3d12x-pipeline-stream.md).
//!
//! * `XCL_TRACE_PSO_STREAM=1`: one line per distinct stream given to CreatePipelineState – each
//!   subobject's offset, type, payload size and hash (shaders: stage and blob hash, no bytes).
//! * `XCL_CONSOLE_STREAM_TYPES` (on by default; `0`: off, `1`: on, anything else is an error): a
//!   stream in the console's numbering (its MS container in type 24, the PC's AS) is rewritten in
//!   the PC's before anything reads it.

use std::collections::HashSet;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::{Mutex, OnceLock};

/// The console's mesh shader subobject (the PC's AS).
const CONSOLE_MS: u32 = 24;

/// Console types read as the PC's: retail mesh streams read to the end with the PC's payload
/// sizes, which is consistent with this but does not tell apart types of the same size.
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

/// The PC's MS
const PC_MS: u32 = 25;

/// A stream's subobjects with the title's type and the PC's.
#[derive(Debug, Clone)]
pub(super) struct Remap {
    /// "console": read to the end in the console's numbering; "MS only": in the PC's (a type
    /// from 25 on, which the console table does not know), with only its MS moved from 24 to 25
    pub reading: &'static str,
    pub subs: Vec<(Sub, u32)>,
}

/// What to do with a stream.
#[derive(Debug, Clone)]
pub(super) enum Plan {
    /// no console MS container in type 24
    Pc,
    Remap(Remap),
    /// a console MS container in type 24, but the stream is not taken as the console's (why)
    Kept(String),
}

/// A stream whose type 24 holds a mesh shader container (`is_ms`) is in the console's
/// numbering: in the PC's, 24 is the AS slot, which no PC stream fills with a mesh shader.
pub(super) fn plan(buf: &[u8], is_ms: impl Fn(&[u8]) -> bool) -> Plan {
    let ms_in_24 = |w: &Walk| w.subs.iter().find(|s| s.ty == CONSOLE_MS).is_some_and(|s| is_ms(&buf[s.payload.clone()]));
    let cw = walk_console(buf);
    if cw.stop.is_none() {
        if !ms_in_24(&cw) {
            return Plan::Pc;
        }
        let subs = cw.subs.into_iter().map(|s| {
            let pc = console_to_pc(s.ty).expect("a console-numbered walk has only known types");
            (s, pc)
        });
        return Plan::Remap(Remap { reading: "console", subs: subs.collect() });
    }
    let pw = walk_pc(buf);
    if !ms_in_24(&pw) {
        return Plan::Pc;
    }
    if let Some((off, ty)) = pw.stop {
        return Plan::Kept(format!("neither numbering reads it to the end (PC: type {ty} at {off:#x})"));
    }
    if pw.subs.iter().any(|s| s.ty == PC_MS) {
        return Plan::Kept("it has a type 25 as well".into());
    }
    let subs = pw.subs.into_iter().map(|s| {
        let pc = if s.ty == CONSOLE_MS { PC_MS } else { s.ty };
        (s, pc)
    });
    Plan::Remap(Remap { reading: "MS only", subs: subs.collect() })
}

/// The stream with its types in the PC's numbering (payloads and offsets unchanged).
pub(super) fn to_pc(buf: &[u8], r: &Remap) -> Vec<u8> {
    let mut out = buf.to_vec();
    for (s, pc) in &r.subs {
        if s.ty != *pc {
            assert_eq!(super::null::subobject_payload(s.ty).or_else(|| console_payload(s.ty)), pc_payload(*pc), "type {} read as {pc}: the payload layout differs", s.ty);
        }
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

const CONSOLE_STREAM_TYPES_VAR: &str = "XCL_CONSOLE_STREAM_TYPES";

/// Whether `XCL_CONSOLE_STREAM_TYPES`'s value turns the console numbering on: unset or `1` on, `0`
/// off, anything else (an empty value too) an error naming the variable.
fn parse_remap(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(v) => Err(format!("{CONSOLE_STREAM_TYPES_VAR}={v:?} must be 0 (off) or 1 (on; unset is on too)")),
    }
}

/// `XCL_CONSOLE_STREAM_TYPES` (on unless it is `0`). Panics on another value: it is read first by
/// `check_remap`, when the D3D12 backend starts.
fn remap_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match parse_remap(std::env::var(CONSOLE_STREAM_TYPES_VAR).ok().as_deref()) {
        Ok(on) => on,
        Err(e) => {
            crate::xlog_error!("D3D12.X: {e}");
            panic!("D3D12.X: {e}");
        }
    })
}

/// Reads `XCL_CONSOLE_STREAM_TYPES` (a malformed value ends the process here, not at the first
/// CreatePipelineState) and logs it when it is off.
pub(super) fn check_remap() {
    if !remap_on() {
        crate::xlog_info!("D3D12.X: {CONSOLE_STREAM_TYPES_VAR}=0: pipeline streams in the console's subobject numbering are passed as they are");
    }
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
/// console's numbering and `XCL_CONSOLE_STREAM_TYPES` is not `0`, the PC-numbered copy to read
/// and create instead.
pub(super) unsafe fn prepare(desc: *const c_void) -> Option<PcStream> {
    prepare_with(desc, trace_on(), remap_on())
}

unsafe fn prepare_with(desc: *const c_void, trace_streams: bool, remap: bool) -> Option<PcStream> {
    if !trace_streams && !remap {
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
    let plan = plan(buf, |p| is_ms_payload(p));
    if trace_streams {
        trace(buf, &plan);
    }
    if !remap {
        return None;
    }
    let r = match plan {
        Plan::Remap(r) => r,
        Plan::Kept(why) => {
            note_kept(buf, &why);
            return None;
        }
        Plan::Pc => return None,
    };
    let out = to_pc(buf, &r);
    note_remapped(buf, &r);
    let desc = Box::new([out.len(), out.as_ptr() as usize]);
    Some(PcStream { _buf: out, desc })
}

fn types_of(w: &Walk) -> Vec<u32> {
    w.subs.iter().map(|s| s.ty).collect()
}

fn is_shader_type(pc: u32) -> bool {
    (1..=6).contains(&pc) || pc == 24 || pc == PC_MS
}

unsafe fn shader_hashes(buf: &[u8], r: &Remap) -> String {
    r.subs
        .iter()
        .filter(|(_, pc)| is_shader_type(*pc))
        .filter_map(|(s, pc)| bytecode(&buf[s.payload.clone()]).map(|c| format!("{pc}:{:016x}", crate::gpu::shader::fnv64(c))))
        .collect::<Vec<_>>()
        .join(" ")
}

unsafe fn ms_hash(buf: &[u8]) -> u64 {
    walk_pc(buf).subs.iter().find(|s| s.ty == CONSOLE_MS).and_then(|s| bytecode(&buf[s.payload.clone()])).map_or(0, crate::gpu::shader::fnv64)
}

fn first_time(seen: &Mutex<Option<HashSet<u64>>>, key: &str) -> bool {
    seen.lock().unwrap().get_or_insert_with(HashSet::new).insert(crate::gpu::shader::fnv64(key.as_bytes()))
}

/// One line per console-numbered pipeline taken in the PC's numbering.
unsafe fn note_remapped(buf: &[u8], r: &Remap) {
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let title: Vec<u32> = r.subs.iter().map(|(s, _)| s.ty).collect();
    let pc: Vec<u32> = r.subs.iter().map(|(_, pc)| *pc).collect();
    let shaders = shader_hashes(buf, r);
    let line = format!("MS {:016x} ({}): types {title:?} read as the PC's {pc:?} (shaders {shaders})", ms_hash(buf), r.reading);
    if first_time(&SEEN, &line) {
        crate::xlog_info!("D3D12.X: console-numbered pipeline stream, {line}");
    }
}

/// One line per stream with a console MS container in type 24 that is left as it is.
unsafe fn note_kept(buf: &[u8], why: &str) {
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let line = format!("MS {:016x} in type 24 left as it is: {why}; types {:?}", ms_hash(buf), types_of(&walk_pc(buf)));
    if first_time(&SEEN, &line) {
        crate::xlog_warn!("D3D12.X: pipeline stream {line}");
    }
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
    if stage(24).is_some_and(|s| s == "MS") && shader(PC_MS).is_none() {
        "MS in the AS slot"
    } else if shader(PC_MS).is_some() {
        if stage(PC_MS).is_some() { "console mesh" } else { "PC mesh" }
    } else if shader(1).is_some() {
        if stage(1).is_some() { "console graphics" } else { "PC graphics" }
    } else if shader(6).is_some() {
        if stage(6).is_some() { "console compute" } else { "PC compute" }
    } else {
        "other"
    }
}

/// A subobject as the trace shows it: offset, type, payload size, then a shader's stage, size
/// and hash, a few state values, any 4-byte payload's value, or the payload's hash.
unsafe fn describe(buf: &[u8], s: &Sub, pc_ty: Option<u32>) -> String {
    let p = &buf[s.payload.clone()];
    let head = format!("{:#x}:{}({})", s.off, s.ty, p.len());
    let word = |k: usize| p.get(4 * k..4 * k + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    match pc_ty {
        Some(t) if is_shader_type(t) => match bytecode(p) {
            Some(c) => format!("{head} {} {}B {:016x}", console_stage(c).unwrap_or_else(|| "pc".into()), c.len(), crate::gpu::shader::fnv64(c)),
            None => format!("{head} null"),
        },
        Some(15) => format!("{head} rt {} {:?}", word(8), (0..8).map(word).collect::<Vec<_>>()),
        Some(17) => format!("{head} samples {}/{}", word(0), word(1)),
        Some(t) if POINTER_PAYLOADS.contains(&t) => format!("{head} {}", if p.iter().any(|&b| b != 0) { "ptr" } else { "zero" }),
        _ if p.len() == 4 => format!("{head} ={:#x}", word(0)),
        _ => format!("{head} {:016x}", crate::gpu::shader::fnv64(p)),
    }
}

/// `XCL_TRACE_PSO_STREAM`: one line per distinct stream (no shader bytes).
unsafe fn trace(buf: &[u8], plan: &Plan) {
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let pc = walk_pc(buf);
    let cw = walk_console(buf);
    let subs: Vec<String> = match plan {
        Plan::Remap(r) => r.subs.iter().map(|(s, t)| describe(buf, s, Some(*t))).collect(),
        _ if pc.stop.is_none() => pc.subs.iter().map(|s| describe(buf, s, Some(s.ty))).collect(),
        _ => cw.subs.iter().map(|s| describe(buf, s, console_to_pc(s.ty))).collect(),
    };
    let reading = match plan {
        Plan::Remap(r) => r.reading,
        _ if pc.stop.is_none() => "pc",
        _ => "console",
    };
    let line = format!(
        "{} B [{}], PC walk {}, console walk {}; read as {reading}: {}",
        buf.len(),
        kind(buf, &pc),
        walk_summary(&pc, buf.len()),
        walk_summary(&cw, buf.len()),
        subs.join(", ")
    );
    if first_time(&SEEN, &line) {
        crate::xlog_info!("D3D12.X: PSO stream {line}");
    }
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

    fn remap(buf: &[u8]) -> Remap {
        match plan(buf, is_ms) {
            Plan::Remap(r) => r,
            other => panic!("not remapped: {other:?}"),
        }
    }

    fn left_alone(buf: &[u8]) -> bool {
        matches!(plan(buf, is_ms), Plan::Pc)
    }

    fn kept(buf: &[u8]) -> bool {
        matches!(plan(buf, is_ms), Plan::Kept(_))
    }

    #[test]
    fn console_stream_reads_as_the_pcs() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let console = stream(CONSOLE_MS, &ms, &ps);
        let r = remap(&console);
        assert_eq!(r.reading, "console");
        let pc = to_pc(&console, &r);
        let w = walk_pc(&pc);
        assert_eq!(w.stop, None);
        assert_eq!(w.end, pc.len());
        assert_eq!(types_of(&w), vec![ROOT_SIGNATURE, PC_MS, PS, BLEND, RASTERIZER, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT]);
        let code = |ty| unsafe { bytecode(payload_of(&pc, &w, ty)) }.unwrap();
        assert_eq!(code(PC_MS), &ms[..]);
        assert_eq!(code(PS), &ps[..]);
        assert_eq!(console_stage(code(PC_MS)).as_deref(), Some("MS"));
        assert_eq!(console_stage(code(PS)).as_deref(), Some("PS"));
        assert_eq!(payload_of(&pc, &w, ROOT_SIGNATURE), RS_POINTER.to_le_bytes());
        assert_eq!(payload_of(&pc, &w, RENDER_TARGET_FORMATS), &rt_formats()[..]);
        assert_eq!(payload_of(&pc, &w, DEPTH_STENCIL_FORMAT), D32_FLOAT.to_le_bytes());
        assert_eq!(pc.len(), console.len());
        let changed: Vec<usize> = (0..pc.len()).filter(|&i| pc[i] != console[i]).collect();
        let ms_off = r.subs.iter().find(|(s, _)| s.ty == CONSOLE_MS).unwrap().0.off;
        assert_eq!(changed, vec![ms_off], "only the MS subobject's type changes (24 → 25)");
    }

    #[test]
    fn console_amplification_reads_as_the_pcs() {
        let (amp, ms, ps) = (container(KIND_AS, NGG_MS), container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(CONSOLE_MS, &ms, &ps));
        s.shader(23, &amp);
        assert_eq!(walk_pc(&s.0).stop.map(|x| x.1), Some(23), "the PC's numbering has no 23");
        let r = remap(&s.0);
        assert_eq!(r.reading, "console");
        let w = walk_pc(&to_pc(&s.0, &r));
        assert_eq!(w.stop, None);
        assert_eq!(types_of(&w), vec![ROOT_SIGNATURE, PC_MS, PS, BLEND, RASTERIZER, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT, PC_AS]);
    }

    #[test]
    fn console_ms_with_types_from_26_on_moves_only_the_ms() {
        const DEPTH_STENCIL2: u32 = 26;
        const RASTERIZER1: u32 = 27;
        const RASTERIZER2: u32 = 28;
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(CONSOLE_MS, &ms, &ps));
        s.raw(DEPTH_STENCIL2, &[0; 60], false).raw(RASTERIZER1, &[0; 44], false).raw(RASTERIZER2, &[0; 40], false);
        assert!(walk_console(&s.0).stop.is_some(), "the console table does not know 25 and up");
        let r = remap(&s.0);
        assert_eq!(r.reading, "MS only");
        let pc = to_pc(&s.0, &r);
        let w = walk_pc(&pc);
        assert_eq!(w.stop, None);
        assert_eq!(
            types_of(&w),
            vec![ROOT_SIGNATURE, PC_MS, PS, BLEND, RASTERIZER, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT, DEPTH_STENCIL2, RASTERIZER1, RASTERIZER2]
        );
        assert_eq!(unsafe { bytecode(payload_of(&pc, &w, PC_MS)) }.unwrap(), &ms[..]);
    }

    #[test]
    fn console_ms_with_a_null_25_is_kept() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(CONSOLE_MS, &ms, &ps));
        s.raw(PC_MS, &[0; 16], true);
        assert!(kept(&s.0));
    }

    #[test]
    fn pc_mesh_stream_is_left_alone() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        assert!(left_alone(&stream(PC_MS, &ms, &ps)), "a PC MS (25) is not the console's numbering");
    }

    #[test]
    fn pc_amplification_and_mesh_stream_is_left_alone() {
        let (amp, ms, ps) = (container(KIND_AS, NGG_MS), container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let mut s = Stream(stream(PC_MS, &ms, &ps));
        s.shader(PC_AS, &amp);
        assert!(left_alone(&s.0));
    }

    #[test]
    fn non_mesh_container_in_24_is_left_alone() {
        let (amp, ps) = (container(KIND_AS, NGG_MS), container(KIND_PS, PS_PS));
        assert!(left_alone(&stream(PC_AS, &amp, &ps)), "an AS container in 24 is the PC's AS");
        assert!(left_alone(&stream(PC_AS, &[0x44; 64], &ps)), "a PC blob in 24 is the PC's AS");
    }

    #[test]
    fn console_table_knows_0_to_24() {
        assert_eq!((25..40).filter_map(console_to_pc).count(), 0);
        assert!((0..23).all(|t| console_to_pc(t) == Some(t)));
        assert_eq!(console_to_pc(23), Some(PC_AS));
        assert_eq!(console_to_pc(24), Some(PC_MS));
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

    fn prepared(buf: &[u8], trace: bool, remap: bool) -> Option<Vec<u8>> {
        let desc: [usize; 2] = [buf.len(), buf.as_ptr() as usize];
        unsafe { prepare_with(desc.as_ptr() as *const c_void, trace, remap) }.map(|s| {
            let d = s.desc() as *const usize;
            unsafe { std::slice::from_raw_parts(*d.add(1) as *const u8, *d).to_vec() }
        })
    }

    #[test]
    fn prepare_rewrites_only_with_the_flag() {
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let console = stream(CONSOLE_MS, &ms, &ps);
        assert!(prepared(&console, false, false).is_none());
        assert!(prepared(&console, true, false).is_none(), "the trace alone changes nothing");
        let pc = prepared(&console, false, true).expect("rewritten");
        assert_eq!(types_of(&walk_pc(&pc)), vec![ROOT_SIGNATURE, PC_MS, PS, BLEND, RASTERIZER, RENDER_TARGET_FORMATS, DEPTH_STENCIL_FORMAT]);
        assert!(prepared(&stream(PC_MS, &ms, &ps), true, true).is_none(), "a PC stream is passed as it is");
    }

    /// The console numbering is on unless `XCL_CONSOLE_STREAM_TYPES=0`; another value is an error.
    #[test]
    fn the_console_numbering_is_on_unless_it_is_0() {
        assert_eq!(parse_remap(None), Ok(true), "on by default");
        assert_eq!(parse_remap(Some("1")), Ok(true));
        assert_eq!(parse_remap(Some("0")), Ok(false));
        for wrong in ["", "2", "yes", " 0", "off"] {
            let e = parse_remap(Some(wrong)).expect_err(wrong);
            assert!(e.contains(CONSOLE_STREAM_TYPES_VAR), "{e}");
        }
        let (ms, ps) = (container(KIND_MS, NGG_MS), container(KIND_PS, PS_PS));
        let console = stream(CONSOLE_MS, &ms, &ps);
        assert!(prepared(&console, false, parse_remap(None).unwrap()).is_some(), "unset: rewritten");
        assert!(prepared(&console, false, parse_remap(Some("0")).unwrap()).is_none(), "0: passed as it is");
    }
}
