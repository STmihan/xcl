# Recompiler: size of the generated HLSL

Notes for `crates/xcl/src/gpu/recomp/emit.rs`, `mod.rs` and `dxc.rs`: what the emitter leaves out of the
HLSL it writes, and how DXC's optimisation level is chosen from the size of that HLSL.

## What is measured

HLSL bytes are a proxy. What costs time and memory is DXC and, after it, the driver's compiler, which
works on the DXIL. Comments and indentation never reach the DXIL: with the optimisation level pinned,
stripping both from a 47 000-instruction compute program gave a byte-identical DXIL container.
Single-use temporaries do reach it at `-O0` (about 5 bytes each, 0.6 % of that program's DXIL) and are
gone at `-O1` and above.

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

`dxc::optimisation_level_for` picks the level from the HLSL size when the caller passes none:

| HLSL bytes | level | why |
| --- | --- | --- |
| above `UNOPTIMISED_ABOVE_HLSL_BYTES` (3 700 000) | `-O0` | safety net for enormous sources |
| above `LIGHTLY_OPTIMISED_ABOVE_HLSL_BYTES` (520 000) | `-O1` | same DXIL size as `-O3`, about a quarter less time |
| otherwise | DXC default (`-O3`) | |

History of the limits: DXC's optimiser was superlinear on translated programs full of short-circuit
`?:` / `&&` / `if` per instruction (a 1 589-instruction console compute shader: about 6 minutes and
more than 30 GB at `-O3`). With branch-free emission (`select` / `and` / `or`) it is fast: that shader
0.3 s, a 15 720-instruction one 6 s (585 KB of DXIL against 1.6 MB at `-O0`, which the driver could not
compile). `-O1` for large sources: a retail title's 19 600-instruction compute shader took 13.5 s
instead of 17.7 s at `-O3`; `-O0` took 10.6 s but gave twice the DXIL.

The limits were 8 MiB and 1 MiB while the HLSL carried comments and four-space indentation. They were
rescaled when those went away so that no program gets a *higher* level than before. Calibration over
all 26 698 programs of one retail shader pack (native form, one memory window), old size against new
size of the same program:

- the 6 programs above 8 MiB are now 4.03 MB and more; the 2 143 between 1 and 8 MiB are now
  0.53 to 3.45 MB, so 3 700 000 separates the two groups exactly;
- 520 000 keeps all of those 2 143 at `-O1`; 954 programs that were below 1 MiB (default level) now
  also get `-O1`.

Any change that makes the HLSL of every program smaller or larger moves programs across these limits.
Repeat the comparison (dump every program before and after with `XCL_DUMP_HLSL` and `XCL_DUMP_EVERY=1`,
join the sizes by file name) and move the limits with it. For the largest programs the level matters
far more than the text: the 47 000-instruction program above gives twice the DXIL at `-O0` than at
`-O1`, and `-O1` takes several times longer.
