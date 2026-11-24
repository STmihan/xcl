//! Dead SCC writes: which `scc = …` the emitter can leave out.
//!
//! Most scalar ALU instructions set SCC, and most of those values are overwritten before
//! anything reads them. This is a backward liveness of SCC over the decoded instructions, on
//! the control-flow rules of `x64_masks::flow_live_out`: an indirect jump or an unresolved
//! branch target makes SCC live, and every loop may be left at its `LOOP_GUARD`, so what is
//! live after a loop or at the end is live before each of its instructions. The end of a
//! kernel-mode program reads SCC (the runner stores it); no other mode does. `translate`
//! applies it to kernel, compute and mesh programs.
//!
//! A missed read drops a value the program needs, a missed overwrite only keeps a line, so:
//! - reads are over-approximated: the ISA's SCC inputs (`s_cbranch_scc0/1`, `s_cselect_*`,
//!   `s_cmov*`, `s_addc_u32`, `s_subb_u32`) and any instruction with the source code 253
//!   (`src_scc`) in a non-immediate field of its encoding or its SDWA / DPP extension dword;
//! - a write is an overwrite only where the emitter really assigns `scc`: the ISA's SCC
//!   outputs without the instructions it does not translate.
//! `tests::scc_liveness_rules_match_what_the_emitter_reads_and_writes` holds both against the
//! emitter for every scalar instruction.
//!
//! VCC was measured and left alone: of 338 142 VCC writes in the compute programs of one
//! retail pack (15.6 million instructions) 2 617 are never read, none of them a `v_cmp`.

use std::collections::BTreeSet;

use super::emit::ext_encoding;
use super::x64_masks::{flow_live_out, Regs};
use crate::gpu::rdna2::{self, Decoded, Encoding, Variant};

/// SCC as the only member of an `x64_masks` register set.
const SCC: Regs = 1;
const NOTHING: Regs = 0;
const SCC_OPERAND: &str = "OPR_SSRC_SPECIAL_SCC";
/// Scalar source code `src_scc` (the emitter's `code32` / `code64`).
const SCC_SOURCE_CODE: u64 = 253;
/// Encoding fields that never hold a source code.
const IMMEDIATE_FIELDS: &[&str] = &["ENCODING", "OP", "SIMM16", "LITERAL", "OFFSET", "OFFSET0", "OFFSET1"];
/// SCC outputs of the ISA the emitter does not translate (they leave `scc` as it was).
const SCC_OUTPUTS_THE_EMITTER_LEAVES: &[&str] = &["S_QUADMASK_B32", "S_QUADMASK_B64"];

fn has_scc_operand(d: &Decoded, input: bool) -> bool {
    d.ienc.operands.iter().any(|o| o.ty == SCC_OPERAND && if input { o.input } else { o.output })
}

/// Any field that may hold a source code holds `src_scc`.
fn names_scc_as_source(enc: &Encoding, words: &[u32]) -> bool {
    enc.fields
        .iter()
        .filter(|f| !IMMEDIATE_FIELDS.contains(&f.name))
        .any(|f| rdna2::field_value(enc, words, f.name) == Some(SCC_SOURCE_CODE))
}

fn extension_names_scc_as_source(d: &Decoded) -> bool {
    match d.variant {
        Variant::Plain => false,
        // an extension layout we do not know: taken as a read
        _ => ext_encoding(d).map_or(true, |e| names_scc_as_source(e, &d.words)),
    }
}

/// The emitted statements of `d` may read `scc`.
pub fn reads_scc(d: &Decoded) -> bool {
    has_scc_operand(d, true) || names_scc_as_source(d.enc, &d.words) || extension_names_scc_as_source(d)
}

/// The emitter assigns `scc` for `d`, whatever its old value.
pub fn overwrites_scc(d: &Decoded) -> bool {
    has_scc_operand(d, false) && !SCC_OUTPUTS_THE_EMITTER_LEAVES.contains(&d.inst.name)
}

fn gen_kill(d: &Decoded) -> (Regs, Regs) {
    // the `s_setpc_b64` of a resolved far jump is not emitted
    if d.far_target.is_some() {
        return (NOTHING, NOTHING);
    }
    (if reads_scc(d) { SCC } else { NOTHING }, if overwrites_scc(d) { SCC } else { NOTHING })
}

/// pcs of the instructions whose SCC write no path reads, for the emitter.
/// `end_reads_scc`: the program's end reads SCC (kernel mode).
pub fn dead_writes(insts: &[Decoded], end_reads_scc: bool) -> BTreeSet<u64> {
    let gen_kill: Vec<(Regs, Regs)> = insts.iter().map(gen_kill).collect();
    let live_out = flow_live_out(insts, &gen_kill, if end_reads_scc { SCC } else { NOTHING }, NOTHING);
    insts
        .iter()
        .zip(live_out)
        .filter(|(d, live)| live & SCC == 0 && d.far_target.is_none() && overwrites_scc(d))
        .map(|(d, _)| d.pc)
        .collect()
}
