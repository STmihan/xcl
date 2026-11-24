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
the same program is 4 819 218 bytes. DXC 1.9 does not get through the structured form at any
level: its memory went past 40 GB within 55 s at `-O1` and past 32 GB within 28 s at `-O0` (both runs
were ended there; the 152 MB form went past 12 GB within 21 s at `-O0`).

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
| above `UNOPTIMISED_ABOVE_HLSL_BYTES` (64 MiB) | `-O0` |
| above `LIGHTLY_OPTIMISED_ABOVE_HLSL_BYTES` (360 000) | `-O1` |
| otherwise | DXC default (`-O3`) |

### The `-O0` limit is not rescaled

The limit was 8 MiB at first. The six largest compute programs of one retail pack were above it, and
their `-O0` DXIL was more than NVIDIA's driver compiler could take: for a 29.6 MB source 21.3 MB of
DXIL at `-O0` against 9.2 MB at `-O1`, and the driver went past 50 GB of commit on the `-O0` DXIL
without finishing. The owner's decision since: these programs compile at `-O1`, and the limit became
64 MiB.

When the HLSL lost its comments, indentation, temporaries, dead SCC writes and unobservable selects,
the limit could have been rescaled with the text (3 940 000 bytes would have kept "the same programs
at `-O0`"). It deliberately was not: that would have put exactly those six programs back at `-O0`.
`-O0` is a safety net for sources far beyond anything measured, so a change that makes the text
smaller must leave it alone – every program only moves further below it, in every form (split wave64,
four memory windows, `XCL_RECOMP_COMMENTS`). In the pack the largest HLSL is now 9 302 789 bytes
with one memory window and 10 330 079 with four: nothing is compiled at `-O0`.

Measured with DXC 1.9 (`dxc.exe -E main -T cs_6_6 -HV 2021 -O1`), the HLSL of
`xcl shaders --recomp` (one memory window, wave64 not split), before = the form with comments,
four-space indentation, every SCC write and every select:

| program (name hash) | HLSL before | HLSL now | DXIL before | DXIL now | DXC time before | now | peak memory now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `3a47dae75a517a13` | 29 598 855 | 9 302 789 | 9 236 536 | 8 994 604 | 232 s | 193 s | 2.8 GB |
| `a983c0d77d33ea04` | 29 524 131 | 9 267 775 | 9 195 688 | 8 958 352 | 166 s | 155 s | 2.8 GB |
| `a3b015f9c6a34f41` | 12 473 761 | 3 992 519 | 1 975 824 | 1 965 296 | 824 s | 677 s | 0.9 GB |
| `cb694c31e0525691` | 152 022 357 | 4 819 218 (block chain) | none (see above) | none after 60 min | – | > 3 600 s | 2.6 GB |

The runs were not alone on the machine (a second DXC process beside most of them): differences in
time of a fifth say nothing. The last program is the one of "Skip guards that outweigh the program":
as the block chain DXC gets through it at `-O0` (48.7 s, 1.6 GB, 11 356 484 bytes of DXIL) and had
not finished at `-O1` after an hour. It is the one program of the pack for which the `-O1` rule is
not yet shown to work.

The text is a third of what it was; the DXIL, which is what the driver compiles, is 2.6 % smaller for the two largest and 0.5 % for the third.

### The `-O1` limit is recalibrated

`-O1` instead of `-O3` costs nothing in DXIL size on large programs and saves DXC time, so here the
limit follows the text: every program that had `-O1` keeps it. Calibration over all 26 698 programs
of the pack, size before (limit 1 MiB) against size now of the same program:

| | one memory window | four windows (`XCL_RECOMP_WINDOWS=4`) |
| --- | --- | --- |
| programs between 1 MiB and 64 MiB before | 2 199, now 386 513 to 9 302 789 bytes | 2 727, now 435 616 to 10 330 079 |
| programs below 1 MiB before, now `-O1` | 2 006 | 2 039 |

360 000 is 7 % below the smallest program that had `-O1`. The price of keeping all of them is that
the number of programs at `-O1` about doubles (one window: 2 199 before, 4 205 now): the two smallest
shrank to 36 % and 33 % of their size, the third smallest is 458 366 bytes. With the limit at 450 000
those two would go to `-O3` and 1 434 programs (one window) would be new at `-O1`.

The split wave64 form NVIDIA GPUs get (`Options::split_wave64`) was not calibrated:
`xcl shaders --recomp` does not produce it.

Any change that makes the HLSL of every program smaller or larger moves programs across this limit.
Repeat the comparison: dump every program before and after with `XCL_DUMP_HLSL` and
`XCL_DUMP_EVERY=1`, join the sizes by file name, and move the limit with it. The `-O0` limit stays.
