# Recompiler: size of the generated HLSL

Notes for `crates/xcl/src/gpu/recomp/emit.rs`, `mod.rs` and `dxc.rs`: what the emitter leaves out of the
HLSL it writes, and how DXC's optimisation level is chosen from the size of that HLSL.

## What is measured

HLSL bytes are a proxy. What costs time and memory is DXC and, after it, the driver's compiler, which
works on the DXIL. Measured on a 47 000-instruction compute program of a retail pack, DXC 1.9.2609,
level pinned to `-O0`:

- comments and indentation never reach the DXIL: stripping both gave a byte-identical container
  (checked with DXC 1.8.2502);
- everything described here together: HLSL 9.7 MB to 4.3 MB, DXIL 4 435 504 to 4 381 772 bytes
  (-1.2 %). Single-use temporaries do reach the DXIL at `-O0`, a few bytes each;
- DXC time: 151 s and 72 s before, 108 s and 101 s after, run alternately on a machine with other
  load. Not distinguishable from noise.

So the text changes make dumps and caches smaller and leave the driver's input almost as it was. The
level does not: the same program at `-O1` gives 2 258 696 bytes of DXIL (51 %) and takes 542 s.

## Indentation

One space per nesting level (`emit::INDENT`), used by `Em::line` and by the structured control flow in
`mod.rs`. Large programs nest 15 and more levels deep; with four spaces per level the indentation was
59 % of the bytes of the largest program.

## Source comments

The `// <pc>: <disassembly>` line per instruction is written only when `Options::source_comments` is
set or the environment variable `XCL_RECOMP_COMMENTS` is present (the variable also bypasses the
translated-shader cache, like the other debug overrides in `pso::cache_allowed`). Instructions without
an effect in this model (`NOOPS`: `s_nop`, `s_waitcnt`, `s_inst_prefetch`, ...) then leave no text at
all. The `/* UNSUPPORTED ... */` markers are always written.

## Scalar results without a temporary

A scalar ALU instruction used to be `uint t = <result>; scc = <condition on t>; dst = t;`. It is now
`dst = <result>; scc = <condition on dst>;` whenever nothing still needs the old value of the
destination:

- 32-bit: the SCC condition must not read the destination register. `s_add_u32 s0, s0, s1` keeps its
  temporary, because its carry is `result < s0` with the old `s0`.
- 64-bit pair: additionally the expression of the high half must not read the low destination
  register. `s_mov_b64 s[8:9], s[7:8]` keeps its temporaries.
- A move of a register (or pair) onto itself writes nothing.
- A destination that is not a register (`Dst::Null`) with no SCC writes nothing.
- An SCC that nothing reads (`scc_liveness`, kernel / compute / mesh programs) counts as no SCC: it is
  not written, so its condition cannot keep the temporary alive. `s_add_u32 s0, s0, s1` followed by an
  instruction that overwrites SCC is `s0 = s0 + s1;`, and an `s_and_b32` into the null register whose
  SCC is dead leaves no text (`Em::read_scc_write`,
  `tests::a_dead_scc_write_does_not_keep_a_temporary`).

`s_*_saveexec` / `s_*_wrexec`:

- the destination does not feed the new mask: `dst = exec; exec = <new>; scc = ...` (no temporary);
- the destination is also the source (`s_and_saveexec_b64 s[30:31], s[30:31]`): the old exec is saved
  in one temporary per half, exec is narrowed, then the destination gets the saved value;
- `wrexec`: `exec = <new>; scc = ...; dst = exec`;
- the destination overlaps exec itself, or the new high half reads the low half of exec: the previous
  fully buffered form.

The check is by identifier (`mentions_identifier`), never by substring: `s1` is not found in `s10`.

## Skip guards that outweigh the program

Structured control flow puts every item after a forward jump out of its level under
`if (!(skip_a || skip_b || ...))`, one flag per pending target (`emit_structured` in `mod.rs`). With
thousands of crossing jumps the guards grow with the square of their number. One compute program of a
retail pack (76 696 instructions, 3 861 skip flags) came to 111 318 363 bytes of HLSL, 92 % of it
guard lines (152 MB in the form with comments and four-space indentation); as the guarded block chain
the same program is 4 819 218 bytes. DXC 1.9.2609 does not get through the structured form at any
level: its memory went past 40 GB within 55 s at `-O1` and past 32 GB within 28 s at `-O0` (both runs
were ended there).

The block chain has structured stretches of its own (`structured_span`), with the same guards; in
that pack none of them is large.

`translate` therefore emits a program as the block chain when the guard lines of its structured form
are more than half of its code (`GUARDS_OUTWEIGH_THE_CODE`). Over the 26 698 programs of that pack the
largest shares are 92 % (that program) and then 5.6 %; no other program is more than 12 % larger
structured than as the block chain. Test: `tests::a_program_with_more_guard_than_code_is_emitted_as_the_block_chain`.

## DXC optimisation level

`dxc::optimisation_level_for` picks the level from the HLSL size when the caller passes none (why a
level is chosen at all: the comment in `dxc::compile_here`):

| HLSL bytes | level |
| --- | --- |
| above `UNOPTIMISED_ABOVE_HLSL_BYTES` (3 940 000) | `-O0` |
| above `LIGHTLY_OPTIMISED_ABOVE_HLSL_BYTES` (520 000) | `-O1` |
| otherwise | DXC default (`-O3`) |

The limits were 8 MiB and 1 MiB while the HLSL carried comments and four-space indentation. They were
rescaled when those went away so that no program gets a *higher* level than before and none that had
`-O1` drops to `-O0`. Calibration over all 26 698 programs of one retail shader pack, old size against
new size of the same program, in the native form (wave64 not split):

| | one memory window | four windows (`XCL_RECOMP_WINDOWS=4`) |
| --- | --- | --- |
| programs above 8 MiB before | 6, now 4 029 730 bytes and more | 6, now 4 582 613 and more |
| programs between 1 and 8 MiB before | 2 143, now 528 261 to 3 450 296 | 2 660, now 583 933 to 3 849 100 |
| programs below 1 MiB before, now `-O1` | 954 | 1 035 |

3 940 000 lies between 3 849 100 and 4 029 730, a margin of about 90 KB on either side. The split
wave64 form NVIDIA GPUs get (`Options::split_wave64`) was not calibrated: `xcl shaders --recomp` does
not produce it.

Any change that makes the HLSL of every program smaller or larger moves programs across these limits
(the smallest program that must stay at `-O0` is only 2 % above the limit). Repeat the comparison:
dump every program before and after with `XCL_DUMP_HLSL` and `XCL_DUMP_EVERY=1`, join the sizes by
file name, and move the limits with it.
