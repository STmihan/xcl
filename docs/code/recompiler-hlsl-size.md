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

`s_*_saveexec` / `s_*_wrexec`:

- the destination does not feed the new mask: `dst = exec; exec = <new>; scc = ...` (no temporary);
- the destination is also the source (`s_and_saveexec_b64 s[30:31], s[30:31]`): the old exec is saved
  in one temporary per half, exec is narrowed, then the destination gets the saved value;
- `wrexec`: `exec = <new>; scc = ...; dst = exec`;
- the destination overlaps exec itself, or the new high half reads the low half of exec: the previous
  fully buffered form.

The check is by identifier (`mentions_identifier`), never by substring: `s1` is not found in `s10`.

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
