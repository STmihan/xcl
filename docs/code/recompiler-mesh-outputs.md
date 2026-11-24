# Recompiler: which register an MS `exp param N` writes

## The bug

`exp param N` names a signature register picked by the console's compiler, and the FSSM records
that choice in an output table. Before this fix, the VS path read that table
(`VsInterface::params`, `vs_output_table` in `gpu/recomp/mod.rs`), but the MS path did not. The
Mesh `32..=63` arm of `export_graphics` in `gpu/recomp/emit.rs` took the N-th entry of
`param_registers(outputs, true)` instead: user registers in register order, then
clip/cull. The table parser also rejected most GTA VI tables, so the VS path fell back to the
same order for 482 VS.

The linking between stages could not hide this. The pass-through mesh shader
(`mesh_passthrough_for`) declares the pixel shader's input elements and fills each from the record
slot of the MS output with the same semantic. The pixel shader reads each attribute at the
register from its own input table (`ps_attribute_slot`). D3D12 links MS to PS by semantic, so the
linking itself is right. What was wrong is what sat in the slot: if param N went to the wrong
register, the data was stored under the wrong semantic, and the PS read, for example, clip
distances as its `TEXCOORD0`.

## The console's table

The table sits at +0x1D4 (+8 for FSSM sub-version 6). It has records of 6 bytes: kind, register,
param number, and three more bytes. The record count is at +0xC0. Kinds are 6 (user), 7 (clip
distance), 8 (cull distance) and 0 (position). The MS variant (NGG, hardware stage 7) has the same
table as the VS.

Counts over the pack (`shaders.rpf`, 9 282 MS and 2 236 VS programs, 2026-11-24):

- Every MS and VS has a table whose param numbers run 0, 1, 2 … in order. In all 9 282 MS the
  record count equals the number of distinct `exp param` targets in the code, and the largest
  target is count − 1.
- The last three bytes of a record take three forms: 0, 0, 0 (46 811 records); the record's
  register, 0, 3 (12 656); or 0, 0, 1 (67). What the 3 and the 1 mean is unknown. The old parser
  accepted zeros only, so it rejected the table of 4 630 MS and 482 VS. The parser now accepts
  these three forms and still rejects every other tail (`param_record_tail_known`).
- 1 906 MS export `pos1`, which holds the clip distances. In 1 905 of them the `exp param` whose
  sources equal `pos1`'s is exactly the param the table's kind 7/8 record names. That param is
  the clip register's place in register order (often param 0), not the last param, as XCL
  assumed. In 1 605 of these MS, XCL put the clip register in a different place.
- A register the program never writes gets no param. For example, with registers 1, 2, 3, 4 and
  the table 1, 2, 4, the console's param 2 is register 4, but XCL took it as register 3.
- The table's registers ascend strictly in every MS. In 8 VS they do not; those 8 were not
  examined.

## Effect on the pack

| | MS | VS |
|---|---|---|
| programs | 9 282 | 2 236 |
| with a parsed table, before | 4 652 (not used) | 1 754 |
| with a parsed table, after | 9 282 | 2 236 |
| HLSL changed | 4 730 | 450 |
| of those, with clip/cull outputs | 1 606 | 8 |

In every changed program, at least one output semantic now gets its data from a different param.
`xcl shaders --recomp` still translates 26 698 / 26 698 programs. The MS HLSL total goes from
1 298 522 138 to 1 298 523 059 bytes and the VS total from 186 984 414 to 186 993 305; no other
mode changes. Only store offsets change, so DXC was run on a sample of changed programs only:
the first 32 met (30 MS as `cs_6_6`, 2 VS as `vs_6_6`), all valid.

## Not covered

- MS-to-PS pairs are only formed at run time (`pso::console_mesh(ms, Some(ps))`). The pack does
  not pair them statically, so the fix was checked per MS and not on real pairs. The visual effect
  (world geometry drawn as flat, unlit shapes) is a hypothesis that needs a run.
- The GS emulation is unchanged. It reads outputs from LDS by register, and its record has no table
  (`GsInterface::record` sets `params: None`).
- Earlier counts (261 / 211 / 780) were not reproduced, and how they were made is not recorded.
  This document's counts use the parser above.

Tests: `mesh_params_follow_the_console_output_table` (`gpu/recomp/tests.rs`) and
`output_table_tails_of_the_gta_vi_pack` (`gpu/recomp/mod.rs`).
