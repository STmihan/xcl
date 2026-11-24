//! Which VGPR writes need their EXEC predication (compute programs).
//!
//! The emitter writes a VALU result as `v = select(lane active, value, v)`: lanes outside EXEC
//! keep their old value, as on the console. Most of those selects guard nothing – the mask is
//! full, or the lanes outside it never look at the register again before it is rewritten – and
//! on the largest retail programs (compiled at -O0) every one of them reaches the driver. This
//! module finds the writes whose select can go, without changing what any lane can observe.
//!
//! It works on what the emitter *wrote*, not on the operand tables: a first emission records,
//! per instruction, the VGPRs its HLSL names (`Trace::statement`), the writes that went through
//! `Em::write32` / `write64` outside any `{ }` (the only full overwrites of the active lanes,
//! `Trace::lane_write`), the SGPRs it assigns, and the masks made as `ballot(active && cond)`.
//! `plan` turns that into a `Plan`, and the program is emitted again with it.
//!
//! Three steps:
//! 1. **Masks, forward** (`Program::scalars`). Every SGPR word gets a name for the value it
//!    holds: what it had at the entry, the word an instruction defined, or a merge where paths
//!    disagree. `s_mov` and the save of `s_*_saveexec` copy names, so EXEC restored from its
//!    save has the name it had before. Alongside: `covering` (the word has the bit of every lane
//!    that runs – EXEC at the entry of a compute program, all ones, and what `and` / `or` keep of
//!    that) and `nonempty` (the mask has a lane: after `s_cbranch_execz` falls through, and its
//!    copies). All three are facts about the lane's own copy of the register, so they hold in
//!    the split wave64 form too, where a mask may stay in its half.
//! 2. **Bounds** (`Bounds`): `defined word ⊆ older word`, only from how the word was made:
//!    a VALU mask ⊆ EXEC, `a & b` ⊆ a and b, `a & ~b` ⊆ a, `a | b` and `a ^ b` ⊆ a common bound
//!    of both. A bound is always a name the instruction saw on entry (or an ancestor of one), so
//!    its definition dominates the bounded word's: while some register still holds the word,
//!    none of its bounds has been redefined, and the chain holds for the values now in place.
//! 3. **Lanes, backward** (`Liveness`). Per VGPR and mask word, the lanes that may still read
//!    its value: nowhere, within a named mask, or anywhere. A read by an instruction that only
//!    looks at its own active lane (plain / SDWA VALU, memory accesses – their effects are all
//!    under the lane test – and `v_readfirstlane` while EXEC has a lane) is "within EXEC";
//!    any other read (`v_readlane`, `v_writelane`, DPP, permutes, swizzles, anything naming
//!    another lane) is "anywhere". A full overwrite under EXEC ends the liveness that lies
//!    within EXEC. Going back over the definition of a mask word, "within it" becomes "within
//!    its bound" (or anywhere); over a merge, the name the predecessor had. A static name
//!    therefore never stands for the mask of another loop iteration.
//!
//! A write keeps its select unless the register is live nowhere outside EXEC after it. Then the
//! lanes outside EXEC get a value nobody reads: later instructions may compute with it in their
//! own inactive lanes, but nothing leaves a lane that is not active (stores, mask bits and
//! scalars are all gated by the lane test).
//!
//! Left alone: every mode but compute (kernel mode reads all lanes back, pixel waves have
//! helper lanes, mesh / geometry were not measured), programs with an indirect jump, and
//! programs where the recorded SGPR writes do not match the operand tables. A loop that runs
//! out of its `LOOP_GUARD` budget is not followed, as in `upper_lanes`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::upper_lanes::{is_valu, sgpr_outputs, successors};
use crate::gpu::rdna2::{self, Decoded, Variant};

const SGPRS: usize = 128;
const NULL_SGPR: usize = 125;
const VCC: usize = 106;
const TTMP0: usize = 108;
const M0: usize = 124;
const EXEC: usize = 126;
const ZERO_CODE: u64 = 128;
const ALL_ONES_CODE: u64 = 193;
const MAX_ANCESTORS: usize = 4096;
const MEAN_VISITS_PER_INSTRUCTION: usize = 256;
const MAX_BOUND_ROUNDS: usize = 8;
const MAX_LIVENESS_SWEEPS: usize = 256;
const CROSS_LANE_MARKERS: [&str; 9] = ["rla(", "_rla_", "lane_id", "lanebit_at(", "first_lane(", "WaveReadLane", ".Sample(", "ddx", "ddy"];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Mask {
    Entry(u8),
    Defined { at: u32, word: u8 },
    Merged { at: u32, word: u8 },
}

#[derive(Default, Debug)]
struct Access {
    reads: BTreeSet<u64>,
    overwrites: BTreeSet<u64>,
    scalar_writes: BTreeSet<usize>,
    exec_bounded: BTreeSet<(usize, usize)>,
    crosses_lanes: bool,
}

#[derive(Default, Debug)]
pub struct Trace {
    by_pc: BTreeMap<u64, Access>,
    open_scopes: usize,
}

fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn identifiers(text: &str) -> Vec<(usize, &str)> {
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !is_identifier_byte(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && is_identifier_byte(bytes[i]) {
            i += 1;
        }
        if !bytes[start].is_ascii_digit() {
            found.push((i, &text[start..i]));
        }
    }
    found
}

fn numbered(name: &str, prefix: &str) -> Option<u64> {
    let digits = name.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn vgpr_named(name: &str) -> Option<u64> {
    numbered(name, "v")
}

fn sgpr_named(name: &str) -> Option<usize> {
    match name {
        "vcc_lo" => Some(VCC),
        "vcc_hi" => Some(VCC + 1),
        "m0" => Some(M0),
        "exec_lo" => Some(EXEC),
        "exec_hi" => Some(EXEC + 1),
        _ => numbered(name, "ttmp").map(|n| TTMP0 + n as usize).or_else(|| numbered(name, "s").map(|n| n as usize)),
    }
}

fn is_assigned(text: &str, after_name: usize) -> bool {
    let rest = text[after_name..].trim_start().as_bytes();
    let assigns_at = |k: usize| rest.get(k) == Some(&b'=') && rest.get(k + 1) != Some(&b'=');
    match rest.first() {
        Some(b'=') => assigns_at(0),
        Some(b'+' | b'-' | b'*' | b'/' | b'%' | b'|' | b'&' | b'^') => assigns_at(1),
        Some(b'<' | b'>') => rest.get(1) == rest.first() && assigns_at(2),
        _ => false,
    }
}

fn vgprs_named_in(text: &str) -> impl Iterator<Item = u64> + '_ {
    identifiers(text).into_iter().filter_map(|(_, name)| vgpr_named(name))
}

fn names_other_lanes(text: &str, lane_test: &str) -> bool {
    let own_lane_only = text.replace(lane_test, "");
    CROSS_LANE_MARKERS.iter().any(|marker| own_lane_only.contains(marker))
}

fn is_comment(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("//") || text.starts_with("/*")
}

/// The statement assigns a word of EXEC.
pub fn assigns_exec(text: &str) -> bool {
    text.contains("exec_") && identifiers(text).into_iter().any(|(end, name)| matches!(sgpr_named(name), Some(r) if r >= EXEC) && is_assigned(text, end))
}

impl Trace {
    pub fn begin_instruction(&mut self) {
        self.open_scopes = 0;
    }

    pub fn statement(&mut self, pc: u64, text: &str, lane_test: &str) {
        if is_comment(text) {
            return;
        }
        let access = self.by_pc.entry(pc).or_default();
        access.reads.extend(vgprs_named_in(text));
        access.scalar_writes.extend(identifiers(text).into_iter().filter(|&(end, _)| is_assigned(text, end)).filter_map(|(_, name)| sgpr_named(name)));
        access.crosses_lanes |= names_other_lanes(text, lane_test);
        let opened = text.bytes().filter(|&b| b == b'{').count();
        let closed = text.bytes().filter(|&b| b == b'}').count();
        self.open_scopes = (self.open_scopes + opened).saturating_sub(closed);
    }

    pub fn lane_write(&mut self, pc: u64, vgpr: u64, value: &str, lane_test: &str) {
        let unconditional = self.open_scopes == 0;
        let access = self.by_pc.entry(pc).or_default();
        access.reads.extend(vgprs_named_in(value));
        access.crosses_lanes |= names_other_lanes(value, lane_test);
        if unconditional {
            access.overwrites.insert(vgpr);
        } else {
            access.reads.insert(vgpr);
        }
    }

    pub fn exec_bounded_mask(&mut self, pc: u64, sgpr: u64, exec_word: usize) {
        self.by_pc.entry(pc).or_default().exec_bounded.insert((sgpr as usize, exec_word));
    }

    fn vgpr_count(&self) -> usize {
        self.by_pc.values().flat_map(|a| a.reads.iter().chain(&a.overwrites)).max().map_or(0, |&top| top as usize + 1)
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct Plan {
    exec_covers_every_lane: BTreeSet<u64>,
    unpredicated: BTreeSet<(u64, u64)>,
}

impl Plan {
    pub fn exec_covers_every_lane(&self, pc: u64) -> bool {
        self.exec_covers_every_lane.contains(&pc)
    }

    pub fn writes_every_lane(&self, pc: u64, vgpr: u64) -> bool {
        self.exec_covers_every_lane(pc) || self.unpredicated.contains(&(pc, vgpr))
    }

    pub fn is_empty(&self) -> bool {
        self.exec_covers_every_lane.is_empty() && self.unpredicated.is_empty()
    }
}

#[derive(Clone, PartialEq)]
struct Scalars {
    masks: [Mask; SGPRS],
    covering: u128,
    nonempty: u128,
}

impl Scalars {
    fn entry() -> Scalars {
        let mut masks = [Mask::Entry(0); SGPRS];
        for (r, m) in masks.iter_mut().enumerate() {
            *m = Mask::Entry(r as u8);
        }
        Scalars { masks, covering: 3 << EXEC, nonempty: 1 << EXEC }
    }

    fn covers(&self, r: usize) -> bool {
        self.covering >> r & 1 == 1
    }

    fn set(&mut self, r: usize, mask: Mask, covering: bool) {
        self.masks[r] = mask;
        self.covering = self.covering & !(1 << r) | (covering as u128) << r;
        self.nonempty &= !(1 << r | 1 << r.saturating_sub(1));
    }

    fn has_a_lane(&self, first_word: usize) -> bool {
        self.nonempty >> first_word & 1 == 1
    }

    fn set_has_a_lane(&mut self, first_word: usize, nonempty: bool) {
        self.nonempty = self.nonempty & !(1 << first_word) | (nonempty as u128) << first_word;
    }

    fn merged_with(mut self, other: &Scalars, at: usize) -> Scalars {
        for (r, (mine, theirs)) in self.masks.iter_mut().zip(&other.masks).enumerate() {
            if mine != theirs {
                *mine = Mask::Merged { at: at as u32, word: r as u8 };
            }
        }
        self.covering &= other.covering;
        self.nonempty &= other.nonempty;
        self
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Source {
    Register(usize),
    AllOnes,
    Zero,
    Unknown,
}

fn is_tracked(r: usize) -> bool {
    r < SGPRS && r != NULL_SGPR
}

fn source_word(code: u64, word: usize, wide: bool) -> Source {
    match code {
        ZERO_CODE => Source::Zero,
        ALL_ONES_CODE => Source::AllOnes,
        c if c as usize == NULL_SGPR => Source::Zero,
        c if c < SGPRS as u64 && (!wide || c <= 123 || c == EXEC as u64) && is_tracked(c as usize + word) => Source::Register(c as usize + word),
        _ => Source::Unknown,
    }
}

fn source_covers(s: Source, st: &Scalars) -> bool {
    match s {
        Source::Register(r) => st.covers(r),
        Source::AllOnes => true,
        Source::Zero | Source::Unknown => false,
    }
}

fn source_mask(s: Source, st: &Scalars) -> Option<Mask> {
    match s {
        Source::Register(r) => Some(st.masks[r]),
        _ => None,
    }
}

fn result_covers(op: &str, first: bool, second: bool) -> bool {
    match op {
        "AND" => first && second,
        "OR" => first || second,
        "ORN2" => first,
        "ORN1" => second,
        _ => false,
    }
}

fn field(d: &Decoded, name: &str) -> u64 {
    rdna2::field_value(d.enc, &d.words, name).unwrap_or(0)
}

fn bit_operation(d: &Decoded) -> &'static str {
    d.inst.name.split('_').nth(1).unwrap_or("")
}

fn changes_exec_with_save(d: &Decoded) -> bool {
    d.inst.name.contains("SAVEEXEC") || d.inst.name.contains("WREXEC")
}

fn is_scalar_move(d: &Decoded) -> bool {
    d.enc.name == "ENC_SOP1" && matches!(d.inst.name, "S_MOV_B32" | "S_MOV_B64")
}

fn is_scalar_bit_operation(d: &Decoded) -> bool {
    d.enc.name == "ENC_SOP2" && matches!(bit_operation(d), "AND" | "OR" | "XOR" | "ANDN2" | "ORN2" | "NAND" | "NOR" | "XNOR") && (d.inst.name.ends_with("_B32") || d.inst.name.ends_with("_B64"))
}

fn accesses_memory_under_exec(d: &Decoded) -> bool {
    let memory = matches!(d.enc.name, "ENC_MUBUF" | "ENC_MTBUF" | "ENC_MIMG" | "ENC_DS" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH");
    let interlocked = ["ATOMIC", "CMPST", "CMPSWAP", "XCHG", "_RTN_"].iter().any(|part| d.inst.name.contains(part));
    memory && !interlocked
}

fn computes_per_lane(d: &Decoded) -> bool {
    is_valu(d) && matches!(d.variant, Variant::Plain | Variant::Sdwa)
}

fn reads_first_active_lane(d: &Decoded) -> bool {
    d.inst.name == "V_READFIRSTLANE_B32" && d.variant == Variant::Plain
}

fn exec_zero_branch(d: &Decoded) -> Option<bool> {
    match d.inst.name {
        "S_CBRANCH_EXECZ" => Some(false),
        "S_CBRANCH_EXECNZ" => Some(true),
        _ => None,
    }
}

fn writes_any_sgpr(d: &Decoded) -> bool {
    d.inst.name.starts_with("S_MOVREL")
}

fn table_outputs(d: &Decoded) -> impl Iterator<Item = usize> {
    sgpr_outputs(d).into_iter().flat_map(|(r, n)| r..r + n).map(|r| r as usize).filter(|&r| is_tracked(r))
}

struct Program<'a> {
    insts: &'a [Decoded],
    wave64: bool,
    succ: Vec<Vec<usize>>,
    pred: Vec<Vec<usize>>,
}

impl<'a> Program<'a> {
    fn of(insts: &'a [Decoded], wave64: bool) -> Option<Program<'a>> {
        let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
        let succ: Vec<Vec<usize>> = successors(insts, &pcs).into_iter().collect::<Option<_>>()?;
        let mut pred = vec![Vec::new(); insts.len()];
        for (i, targets) in succ.iter().enumerate() {
            for &j in targets {
                pred[j].push(i);
            }
        }
        Some(Program { insts, wave64, succ, pred })
    }

    fn mask_words(&self, d: &Decoded) -> usize {
        if self.wave64 && d.inst.name.ends_with("_B64") {
            2
        } else {
            1
        }
    }

    fn lane_words(&self) -> usize {
        if self.wave64 {
            2
        } else {
            1
        }
    }

    fn defined(i: usize, r: usize) -> Mask {
        Mask::Defined { at: i as u32, word: r as u8 }
    }

    fn exec_covers_every_lane(&self, st: &Scalars) -> bool {
        (0..self.lane_words()).all(|word| st.covers(EXEC + word))
    }

    fn step(&self, i: usize, st: &mut Scalars) {
        let before = st.clone();
        self.step_registers(i, st);
        self.step_nonempty(&self.insts[i], &before, st);
    }

    fn source_has_a_lane(&self, s: Source, st: &Scalars) -> bool {
        match s {
            Source::Register(r) => st.has_a_lane(r),
            Source::AllOnes => true,
            Source::Zero | Source::Unknown => false,
        }
    }

    fn step_nonempty(&self, d: &Decoded, before: &Scalars, st: &mut Scalars) {
        let whole_mask = self.mask_words(d) == self.lane_words() && !writes_any_sgpr(d);
        let wide = self.mask_words(d) == 2;
        let dst = field(d, "SDST") as usize;
        if whole_mask && changes_exec_with_save(d) {
            let source = self.source_has_a_lane(source_word(field(d, "SSRC0"), 0, wide), before);
            let exec = bit_operation(d) == "OR" && (source || before.has_a_lane(EXEC));
            st.set_has_a_lane(EXEC, exec);
            if is_tracked(dst) && is_tracked(dst + self.lane_words() - 1) {
                st.set_has_a_lane(dst, if d.inst.name.contains("WREXEC") { exec } else { before.has_a_lane(EXEC) });
            }
        } else if whole_mask && is_scalar_move(d) && is_tracked(dst) {
            st.set_has_a_lane(dst, self.source_has_a_lane(source_word(field(d, "SSRC0"), 0, wide), before));
        } else if whole_mask && is_scalar_bit_operation(d) && bit_operation(d) == "OR" && is_tracked(dst) {
            let first = self.source_has_a_lane(source_word(field(d, "SSRC0"), 0, wide), before);
            let second = self.source_has_a_lane(source_word(field(d, "SSRC1"), 0, wide), before);
            st.set_has_a_lane(dst, first || second);
        }
        if self.exec_covers_every_lane(st) {
            st.set_has_a_lane(EXEC, true);
        }
    }

    fn step_registers(&self, i: usize, st: &mut Scalars) {
        let d = &self.insts[i];
        if writes_any_sgpr(d) {
            for r in (0..SGPRS).filter(|&r| is_tracked(r)) {
                st.set(r, Self::defined(i, r), false);
            }
            return;
        }
        let before = st.clone();
        for r in table_outputs(d) {
            st.set(r, Self::defined(i, r), false);
        }
        let wide = self.mask_words(d) == 2;
        for word in 0..self.mask_words(d) {
            let dst = field(d, "SDST") as usize + word;
            if changes_exec_with_save(d) {
                let source = source_word(field(d, "SSRC0"), word, wide);
                let exec = EXEC + word;
                let covering = result_covers(bit_operation(d), source_covers(source, &before), before.covers(exec));
                st.set(exec, Self::defined(i, exec), covering);
                if !is_tracked(dst) {
                    continue;
                }
                if d.inst.name.contains("WREXEC") {
                    st.set(dst, Self::defined(i, dst), covering);
                } else {
                    st.set(dst, before.masks[exec], before.covers(exec));
                }
            } else if is_scalar_move(d) && is_tracked(dst) {
                match source_word(field(d, "SSRC0"), word, wide) {
                    Source::Register(r) => st.set(dst, before.masks[r], before.covers(r)),
                    Source::AllOnes => st.set(dst, Self::defined(i, dst), true),
                    Source::Zero | Source::Unknown => {}
                }
            } else if is_scalar_bit_operation(d) && is_tracked(dst) {
                let first = source_word(field(d, "SSRC0"), word, wide);
                let second = source_word(field(d, "SSRC1"), word, wide);
                let covering = result_covers(bit_operation(d), source_covers(first, &before), source_covers(second, &before));
                st.set(dst, Self::defined(i, dst), covering);
            }
        }
    }

    fn along_edge(&self, from: usize, to: usize, leaving: &Scalars) -> Scalars {
        let d = &self.insts[from];
        let target = rdna2::branch_target(d).filter(|_| d.far_target.is_none());
        let taken_only = target == Some(self.insts[to].pc) && to != from + 1;
        let fallthrough_only = to == from + 1 && target != Some(self.insts[to].pc);
        let mut arriving = leaving.clone();
        match exec_zero_branch(d) {
            Some(true) if taken_only => arriving.set_has_a_lane(EXEC, true),
            Some(false) if fallthrough_only => arriving.set_has_a_lane(EXEC, true),
            _ => {}
        }
        arriving
    }

    fn arriving(&self, i: usize, after: &[Option<Scalars>]) -> Option<Scalars> {
        let entry = (i == 0).then(Scalars::entry);
        let arrivals = self.pred[i].iter().filter_map(|&p| after[p].as_ref().map(|st| self.along_edge(p, i, st)));
        let mut merged = entry;
        for incoming in arrivals {
            merged = Some(match merged {
                None => incoming,
                Some(m) => m.merged_with(&incoming, i),
            });
        }
        merged
    }

    fn scalars(&self) -> Option<(Vec<Option<Scalars>>, Vec<Option<Scalars>>)> {
        let n = self.insts.len();
        let mut before: Vec<Option<Scalars>> = vec![None; n];
        let mut after: Vec<Option<Scalars>> = vec![None; n];
        let mut visits_left = n * MEAN_VISITS_PER_INSTRUCTION;
        let mut work = BTreeSet::from([0usize]);
        while let Some(i) = work.pop_first() {
            visits_left = visits_left.checked_sub(1)?;
            let Some(arriving) = self.arriving(i, &after) else { continue };
            let mut leaving = arriving.clone();
            self.step(i, &mut leaving);
            before[i] = Some(arriving);
            if after[i].as_ref() != Some(&leaving) {
                after[i] = Some(leaving);
                work.extend(self.succ[i].iter().copied());
            }
        }
        Some((before, after))
    }
}

#[derive(Default, PartialEq)]
struct Bounds {
    superset: HashMap<Mask, Mask>,
}

impl Bounds {
    fn ancestors(&self, m: Mask) -> impl Iterator<Item = Mask> + '_ {
        std::iter::successors(Some(m), |m| self.superset.get(m).copied()).take(MAX_ANCESTORS)
    }

    fn within(&self, m: Mask, bound: Mask) -> bool {
        self.ancestors(m).any(|a| a == bound)
    }

    fn common_bound(&self, a: Mask, b: Mask) -> Option<Mask> {
        let of_a: Vec<Mask> = self.ancestors(a).collect();
        self.ancestors(b).find(|m| of_a.contains(m))
    }

    fn tightest(&self, a: Option<Mask>, b: Option<Mask>) -> Option<Mask> {
        match (a, b) {
            (Some(a), Some(b)) if self.ancestors(a).count() > self.ancestors(b).count() => Some(a),
            (Some(_), Some(b)) => Some(b),
            (a, b) => a.or(b),
        }
    }

    fn union(&self, a: Source, b: Source, st: &Scalars) -> Option<Mask> {
        match (source_mask(a, st), source_mask(b, st)) {
            (Some(a), Some(b)) => self.common_bound(a, b),
            (Some(a), None) if b == Source::Zero => Some(a),
            (None, Some(b)) if a == Source::Zero => Some(b),
            _ => None,
        }
    }

    fn of_bit_operation(&self, op: &str, first: Source, second: Source, st: &Scalars) -> Option<Mask> {
        match op {
            "AND" => self.tightest(source_mask(first, st), source_mask(second, st)),
            "ANDN2" => source_mask(first, st),
            "ANDN1" => source_mask(second, st),
            "OR" | "XOR" => self.union(first, second, st),
            _ => None,
        }
    }

    fn refined(&self, p: &Program, before: &[Option<Scalars>], trace: &Trace) -> Bounds {
        let mut superset = HashMap::new();
        for (i, d) in p.insts.iter().enumerate() {
            let Some(st) = &before[i] else { continue };
            if writes_any_sgpr(d) {
                continue;
            }
            if let Some(access) = trace.by_pc.get(&d.pc) {
                for &(r, exec_word) in &access.exec_bounded {
                    superset.insert(Program::defined(i, r), st.masks[EXEC + exec_word]);
                }
            }
            let wide = p.mask_words(d) == 2;
            for word in 0..p.mask_words(d) {
                let dst = field(d, "SDST") as usize + word;
                let bound = if changes_exec_with_save(d) {
                    let source = source_word(field(d, "SSRC0"), word, wide);
                    self.of_bit_operation(bit_operation(d), source, Source::Register(EXEC + word), st).map(|b| (EXEC + word, b))
                } else if is_scalar_bit_operation(d) && is_tracked(dst) {
                    let first = source_word(field(d, "SSRC0"), word, wide);
                    let second = source_word(field(d, "SSRC1"), word, wide);
                    self.of_bit_operation(bit_operation(d), first, second, st).map(|b| (dst, b))
                } else {
                    None
                };
                if let Some((r, bound)) = bound {
                    superset.insert(Program::defined(i, r), bound);
                }
            }
        }
        Bounds { superset }
    }

    fn of(p: &Program, before: &[Option<Scalars>], trace: &Trace) -> Bounds {
        let mut bounds = Bounds::default();
        for _ in 0..MAX_BOUND_ROUNDS {
            let next = bounds.refined(p, before, trace);
            if next == bounds {
                break;
            }
            bounds = next;
        }
        bounds
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Lanes {
    Nowhere,
    Within(Mask),
    Anywhere,
}

type Live = Vec<[Lanes; 2]>;

struct Liveness<'a> {
    program: &'a Program<'a>,
    trace: &'a Trace,
    before: &'a [Option<Scalars>],
    after: &'a [Option<Scalars>],
    bounds: Bounds,
    vgprs: usize,
}

impl Liveness<'_> {
    fn join(&self, a: Lanes, b: Lanes) -> Lanes {
        match (a, b) {
            (Lanes::Nowhere, x) | (x, Lanes::Nowhere) => x,
            (Lanes::Within(a), Lanes::Within(b)) => self.bounds.common_bound(a, b).map_or(Lanes::Anywhere, Lanes::Within),
            _ => Lanes::Anywhere,
        }
    }

    fn reads_own_active_lane_only(&self, d: &Decoded, access: &Access, st: &Scalars) -> bool {
        let scalars_are_exec_bounded = access.scalar_writes.iter().all(|&r| access.exec_bounded.iter().any(|&(bounded, _)| bounded == r));
        let lane_local = computes_per_lane(d) || accesses_memory_under_exec(d);
        lane_local && !access.crosses_lanes && scalars_are_exec_bounded || reads_first_active_lane(d) && st.has_a_lane(EXEC)
    }

    fn before_definitions(&self, i: usize, live: &mut Live) {
        if table_outputs(&self.program.insts[i]).next().is_none() && !writes_any_sgpr(&self.program.insts[i]) {
            return;
        }
        for lanes in live.iter_mut().flatten() {
            if let Lanes::Within(m @ Mask::Defined { at, .. }) = *lanes {
                if at as usize == i {
                    *lanes = self.bounds.superset.get(&m).map_or(Lanes::Anywhere, |&bound| Lanes::Within(bound));
                }
            }
        }
    }

    fn before_instruction(&self, i: usize, live: &mut Live, unpredicated: &mut BTreeSet<(u64, u64)>) {
        let Some(st) = &self.before[i] else { return };
        let d = &self.program.insts[i];
        self.before_definitions(i, live);
        let Some(access) = self.trace.by_pc.get(&d.pc) else { return };
        let words = self.program.lane_words();
        for &v in &access.overwrites {
            let mut dead_elsewhere = true;
            for word in 0..words {
                let written = match live[v as usize][word] {
                    Lanes::Nowhere => true,
                    Lanes::Within(m) => st.covers(EXEC + word) || self.bounds.within(m, st.masks[EXEC + word]),
                    Lanes::Anywhere => st.covers(EXEC + word),
                };
                if written {
                    live[v as usize][word] = Lanes::Nowhere;
                } else {
                    dead_elsewhere = false;
                }
            }
            if dead_elsewhere {
                unpredicated.insert((d.pc, v));
            }
        }
        let own_lane = self.reads_own_active_lane_only(d, access, st);
        for &v in &access.reads {
            for word in 0..words {
                let read = if own_lane { Lanes::Within(st.masks[EXEC + word]) } else { Lanes::Anywhere };
                live[v as usize][word] = self.join(live[v as usize][word], read);
            }
        }
    }

    fn leaving(&self, from: usize, to: usize, entering: &Live) -> Live {
        let Some(st) = &self.after[from] else { return entering.clone() };
        let mut live = entering.clone();
        for lanes in live.iter_mut().flatten() {
            if let Lanes::Within(Mask::Merged { at, word }) = *lanes {
                if at as usize == to {
                    *lanes = Lanes::Within(st.masks[word as usize]);
                }
            }
        }
        live
    }

    fn dead(&self) -> Live {
        vec![[Lanes::Nowhere; 2]; self.vgprs]
    }

    fn at_end_of(&self, last: usize, entering: &[Option<Live>]) -> Live {
        let mut live = self.dead();
        for &to in &self.program.succ[last] {
            let Some(target) = &entering[to] else { continue };
            for (mine, theirs) in live.iter_mut().zip(self.leaving(last, to, target)) {
                for word in 0..2 {
                    mine[word] = self.join(mine[word], theirs[word]);
                }
            }
        }
        live
    }

    fn starts_block(&self, i: usize) -> bool {
        i == 0 || self.program.pred[i] != [i - 1] || self.program.succ[i - 1] != [i]
    }

    fn blocks(&self) -> Vec<(usize, usize)> {
        let n = self.program.insts.len();
        let starts: Vec<usize> = (0..n).filter(|&i| self.starts_block(i)).collect();
        starts.iter().enumerate().map(|(k, &s)| (s, starts.get(k + 1).copied().unwrap_or(n))).collect()
    }

    fn sweep(&self, blocks: &[(usize, usize)], entering: &mut [Option<Live>], unpredicated: &mut BTreeSet<(u64, u64)>) -> bool {
        let mut changed = false;
        for &(start, end) in blocks.iter().rev() {
            let mut live = self.at_end_of(end - 1, entering);
            for i in (start..end).rev() {
                self.before_instruction(i, &mut live, unpredicated);
            }
            if entering[start].as_ref() != Some(&live) {
                entering[start] = Some(live);
                changed = true;
            }
        }
        changed
    }

    fn unpredicated_writes(&self) -> BTreeSet<(u64, u64)> {
        let blocks = self.blocks();
        let mut entering: Vec<Option<Live>> = vec![None; self.program.insts.len()];
        for _ in 0..MAX_LIVENESS_SWEEPS {
            let mut unpredicated = BTreeSet::new();
            if !self.sweep(&blocks, &mut entering, &mut unpredicated) {
                return unpredicated;
            }
        }
        BTreeSet::new()
    }
}

fn trace_matches_tables(insts: &[Decoded], trace: &Trace) -> bool {
    insts.iter().all(|d| {
        let Some(access) = trace.by_pc.get(&d.pc) else { return true };
        let listed: BTreeSet<usize> = table_outputs(d).collect();
        writes_any_sgpr(d) || access.scalar_writes.iter().all(|r| listed.contains(r))
    })
}

pub fn plan(insts: &[Decoded], wave64: bool, trace: &Trace) -> Plan {
    let Some(program) = Program::of(insts, wave64) else { return Plan::default() };
    if insts.is_empty() || !trace_matches_tables(insts, trace) {
        return Plan::default();
    }
    let Some((before, after)) = program.scalars() else { return Plan::default() };
    let exec_covers_every_lane = insts
        .iter()
        .zip(&before)
        .filter(|(_, st)| st.as_ref().is_some_and(|st| program.exec_covers_every_lane(st)))
        .map(|(d, _)| d.pc)
        .collect();
    let bounds = Bounds::of(&program, &before, trace);
    let liveness = Liveness { program: &program, trace, before: &before, after: &after, bounds, vgprs: trace.vgpr_count() };
    Plan { exec_covers_every_lane, unpredicated: liveness.unpredicated_writes() }
}

#[cfg(test)]
mod tests {
    use super::super::{dxc, runner, translate, CsInterface, LanePredication, Mode, Options};
    use crate::gpu::rdna2::encode;

    const V: u64 = 256;
    const VCC: u64 = 106;
    const EXEC: u64 = 126;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)], Option<u32>);

    // lanes 0..19 of each wave: vcc = 20 > v0, EXEC saved in s[10:11]
    const NARROW: [I; 2] =
        [("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 20), ("VSRC1", 0)], None), ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", VCC)], None)];
    const RESTORE: I = ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC), ("SSRC0", 10)], None);
    // v10 = v0 << 1
    const TEMP: I = ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 10), ("SRC0", 129), ("VSRC1", 0)], None);
    // mem[s0 + 16 * v0] = v[24:27]
    const STORE: [I; 4] = [
        ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 132), ("VSRC1", 0)], None),
        ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 30), ("SRC0", 0), ("VSRC1", 30)], None),
        ("V_MOV_B32", "ENC_VOP1", &[("VDST", 31), ("SRC0", 128)], None),
        ("GLOBAL_STORE_DWORDX4", "ENC_FLAT_GLBL", &[("ADDR", 30), ("DATA", 24), ("SADDR", 0x7D)], None),
    ];

    fn program(lines: &[I]) -> Vec<u8> {
        let mut bytes: Vec<u8> = lines.iter().flat_map(|(n, e, f, l)| encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}"))).collect();
        bytes.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        bytes
    }

    fn compute(predication: LanePredication) -> Options {
        let cs = CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 3 };
        Options { mode: Mode::Compute(cs), wave64: true, lane_predication: predication, ..Default::default() }
    }

    fn hlsl_with(lines: &[I], options: &Options) -> String {
        let t = translate(&program(lines), options).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        t.hlsl
    }

    fn hlsl(lines: &[I]) -> String {
        hlsl_with(lines, &compute(LanePredication::WhereObservable))
    }

    fn predicated(hlsl: &str, vgpr: u64) -> usize {
        hlsl.matches(&format!("v{vgpr} = select(xcl_active, ")).count() + hlsl.matches(&format!("v{vgpr} = select(lanebit(exec_lo, exec_hi), ")).count()
    }

    fn unpredicated(hlsl: &str, vgpr: u64) -> usize {
        hlsl.lines().filter(|l| l.trim_start().starts_with(&format!("v{vgpr} = ")) && !l.contains("select(xcl_active") && !l.contains("select(lanebit(")).count()
    }

    #[test]
    fn nothing_is_predicated_while_exec_is_the_entry_mask() {
        let h = hlsl(&[TEMP, ("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", V + 10)], None), STORE[0], STORE[1], STORE[2], STORE[3]]);
        assert!(!h.contains("select(xcl_active"), "{h}");
        assert!(h.contains("if (true)"), "the store's lane test is known too");
    }

    #[test]
    fn exec_restored_from_its_save_is_full_again() {
        let h = hlsl(&[NARROW[0], NARROW[1], TEMP, RESTORE, ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 10)], None)]);
        assert_eq!((predicated(&h, 11), unpredicated(&h, 11)), (0, 1));
        assert_eq!(predicated(&h, 10), 1, "v10 is read in every lane after the restore");
    }

    #[test]
    fn a_value_only_read_under_the_same_mask_is_written_in_every_lane() {
        let h = hlsl(&[
            NARROW[0],
            NARROW[1],
            TEMP,
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 24), ("SRC0", V + 10), ("VSRC1", 24)], None),
            RESTORE,
            STORE[0],
            STORE[1],
            STORE[2],
            STORE[3],
        ]);
        assert_eq!((predicated(&h, 10), unpredicated(&h, 10)), (0, 1), "the temporary");
        assert_eq!(predicated(&h, 24), 1, "stored from every lane after the restore");
    }

    #[test]
    fn a_narrower_mask_reads_within_the_wider_write() {
        let h = hlsl(&[
            NARROW[0],
            NARROW[1],
            TEMP,
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 5), ("VSRC1", 0)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 10), ("VSRC1", 10)], None),
            RESTORE,
        ]);
        assert_eq!(predicated(&h, 10), 0);
        assert_eq!(predicated(&h, 11), 0, "never read");
    }

    #[test]
    fn a_wider_mask_reads_what_the_narrower_write_left_alone() {
        let h = hlsl(&[
            NARROW[0],
            NARROW[1],
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", EXEC)], None),
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 5), ("VSRC1", 0)], None),
            TEMP,
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC), ("SSRC0", 12)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 10), ("VSRC1", 10)], None),
            RESTORE,
        ]);
        assert_eq!(predicated(&h, 10), 1);
    }

    #[test]
    fn the_else_mask_is_within_the_saved_one() {
        let h = hlsl(&[
            NARROW[0],
            NARROW[1],
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", EXEC)], None),
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 5), ("VSRC1", 0)], None),
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", EXEC), ("SSRC0", 12), ("SSRC1", EXEC)], None),
            TEMP,
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 10), ("VSRC1", 10)], None),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC), ("SSRC0", 12)], None),
            // every lane of the saved mask is written here: nothing of the else branch is left
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 10), ("SRC0", 128)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 11), ("SRC0", V + 10)], None),
            RESTORE,
        ]);
        assert_eq!(predicated(&h, 10), 0, "{h}");
        assert_eq!(unpredicated(&h, 10), 2);
    }

    #[test]
    fn a_lane_read_by_index_sees_inactive_lanes() {
        let h = hlsl(&[NARROW[0], NARROW[1], TEMP, RESTORE, ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 12), ("SRC0", V + 10), ("SRC1", 128 + 40)], None)]);
        assert_eq!(predicated(&h, 10), 1);
    }

    #[test]
    fn a_dpp_source_sees_inactive_lanes() {
        let h = hlsl(&[
            NARROW[0],
            NARROW[1],
            TEMP,
            ("V_MOV_B32", "VOP1_VOP_DPP16", &[("VDST", 11), ("SRC0", 0xFA), ("VSRC0", 10), ("DPP_CTRL", 0x140), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)], None),
            RESTORE,
        ]);
        assert_eq!(predicated(&h, 10), 1);
    }

    #[test]
    fn the_first_active_lane_is_within_exec_only_when_exec_has_a_lane() {
        const FIRST: I = ("V_READFIRSTLANE_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 10)], None);
        let unguarded = hlsl(&[NARROW[0], NARROW[1], TEMP, FIRST, RESTORE]);
        assert_eq!(predicated(&unguarded, 10), 1, "EXEC may be empty: lane 0 is read");
        let guarded = hlsl(&[NARROW[0], NARROW[1], ("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", 2)], None), TEMP, FIRST, RESTORE]);
        assert_eq!(predicated(&guarded, 10), 0, "{guarded}");
    }

    #[test]
    fn a_buffer_address_is_read_in_active_lanes() {
        let h = hlsl(&[
            NARROW[0],
            NARROW[1],
            TEMP,
            ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 11), ("VADDR", 10), ("SRSRC", 1), ("SOFFSET", 128), ("OFFEN", 1)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 24), ("SRC0", V + 11), ("VSRC1", 24)], None),
            RESTORE,
            STORE[0],
            STORE[1],
            STORE[2],
            STORE[3],
        ]);
        assert_eq!(predicated(&h, 10), 0);
        assert_eq!(predicated(&h, 24), 1);
    }

    #[test]
    fn a_mask_of_another_iteration_is_another_mask() {
        // do { exec = saved & (s12 > v0); v11 += v10; v10 = v0 << 1; s12 += 7 } while (s12 != 28)
        let h = hlsl(&[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 10), ("SSRC0", EXEC)], None),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", EXEC), ("SSRC0", 10)], None),
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 12), ("VSRC1", 0)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 10), ("VSRC1", 11)], None),
            TEMP,
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 128 + 7)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 128 + 28)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF9)], None),
            RESTORE,
        ]);
        assert_eq!(predicated(&h, 10), 1, "the next iteration reads v10 in lanes this one did not write\n{h}");
    }

    #[test]
    fn every_write_stays_predicated_on_request_and_outside_compute() {
        let lines = [NARROW[0], NARROW[1], TEMP, ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 10), ("VSRC1", 10)], None), RESTORE];
        assert_eq!(predicated(&hlsl(&lines), 10), 0);
        assert_eq!(predicated(&hlsl_with(&lines, &compute(LanePredication::EveryWrite)), 10), 1);
        assert_eq!(predicated(&hlsl_with(&lines, &Options { mode: Mode::Kernel, ..Default::default() }), 10), 1, "kernel mode reads every lane back");
    }

    #[test]
    fn a_wave32_program_tests_its_one_exec_word() {
        let lines: [I; 5] = [
            ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 20), ("VSRC1", 0)], None),
            ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", VCC)], None),
            TEMP,
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 10), ("VSRC1", 11)], None),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", EXEC), ("SSRC0", 10)], None),
        ];
        let lane_test = "select(xcl_active, ";
        let h = hlsl_with(&lines, &Options { wave64: false, ..compute(LanePredication::WhereObservable) });
        assert!(!h.contains(lane_test), "{h}");
        let all = hlsl_with(&lines, &Options { wave64: false, ..compute(LanePredication::EveryWrite) });
        assert_eq!(all.matches(lane_test).count(), 2);
        assert_eq!(all.matches("xcl_active = (((exec_lo >> lane_id) & 1u) != 0u);").count(), 3, "at the entry and after each EXEC write\n{all}");
    }

    /// The same program with and without the dropped selects gives the same memory: lanes
    /// 0..19 run the conditional part, lane 40 – outside it – is read back by index.
    #[test]
    fn gpu_dropped_predication_changes_no_result() {
        match (runner::adapter(), runner::wave64_programs_supported(), dxc::location()) {
            (Ok(_), Ok(true), Ok(_)) => {}
            (a, w, d) => {
                println!("GPU/DXC unavailable – skipping ({:?} / {:?} / {:?})", a.as_ref().err(), w.as_ref().ok(), d.as_ref().err());
                runner::skip_test(if a.is_err() || w.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" });
                return;
            }
        }
        let lines: Vec<I> = vec![
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 24), ("SRC0", 255)], Some(0x1111)),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", 255)], Some(0x2222)),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", 255)], Some(0x3333)),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 27), ("SRC0", 255)], Some(0x4444)),
            NARROW[0],
            NARROW[1],
            TEMP,
            // read after the restore in every lane
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 24), ("SRC0", V + 10), ("VSRC1", 24)], None),
            // lane 40 is read back by index
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 25), ("SRC0", V + 10)], None),
            // rewritten in every lane before it is read
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 26), ("SRC0", V + 10), ("VSRC1", 10)], None),
            // a narrower mask, then the rest of the saved one
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 12), ("SSRC0", EXEC)], None),
            ("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", 128 + 5), ("VSRC1", 0)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 130), ("VSRC1", 10)], None),
            ("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 27), ("SRC0", V + 11), ("VSRC1", 27)], None),
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", EXEC), ("SSRC0", 12), ("SSRC1", EXEC)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 11), ("SRC0", 131), ("VSRC1", 10)], None),
            ("V_SUB_NC_U32", "ENC_VOP2", &[("VDST", 27), ("SRC0", V + 27), ("VSRC1", 11)], None),
            RESTORE,
            ("V_READLANE_B32", "ENC_VOP3", &[("VDST", 14), ("SRC0", V + 25), ("SRC1", 128 + 40)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 26), ("SRC0", 14)], None),
            STORE[0],
            STORE[1],
            STORE[2],
            STORE[3],
        ];
        let base = 0x100usize;
        let groups = 1usize;
        let run = |predication: LanePredication| {
            let mut options = compute(predication);
            options.split_wave64 = runner::split_for(&options.mode);
            let t = translate(&program(&lines), &options).unwrap();
            assert!(t.complete(), "{:?}", t.unsupported_why);
            let dxil = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}"));
            let mut io = vec![0u32; 128];
            io[0] = base as u32;
            (t.hlsl, runner::dispatch(&dxil, [groups as u32, 1, 1], &io, &vec![0u32; (base + groups * 64 * 16) / 4]).unwrap())
        };
        let (all_hlsl, all) = run(LanePredication::EveryWrite);
        let (hlsl, dropped) = run(LanePredication::WhereObservable);
        assert_eq!(predicated(&all_hlsl, 10), 1);
        assert_eq!((predicated(&hlsl, 10), predicated(&hlsl, 11), predicated(&hlsl, 26)), (0, 0, 0));
        assert_eq!((predicated(&hlsl, 24), predicated(&hlsl, 25), predicated(&hlsl, 27)), (1, 1, 2));
        assert_eq!(all, dropped);
        for thread in 0..groups * 64 {
            let lane = (thread % 64) as u32;
            let at = (base + thread * 16) / 4;
            let temp = 2 * lane;
            let expected = match lane {
                0..=4 => [0x1111 + temp, temp, 0x2222, 0x4444 + 4 * temp],
                5..=19 => [0x1111 + temp, temp, 0x2222, 0x4444 - 8 * temp],
                _ => [0x1111, 0x2222, 0x2222, 0x4444],
            };
            assert_eq!(dropped[at..at + 4], expected, "thread {thread}");
        }
    }
}
