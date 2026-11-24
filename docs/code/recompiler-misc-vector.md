# Recompiler: the misc position vector (render-target array and viewport index)

On RDNA2 `exp pos1` is the misc vector when the program writes a render-target array index or a
viewport index: `.z` bits 0..15 hold the array index, bits 16..31 the viewport index, and the
clip/cull distances move to `pos2`..`pos3`. Without those outputs, `pos1`..`pos2` are the clip/cull
distances. The layout is taken from the public reference (VIctory PR #1417, facts only); it is not
measured on console output here.

## Gate on the signature, not on `exp pos1`

The translator treats `pos1` as the misc vector only when the output signature declares
`SV_RenderTargetArrayIndex` (system value 4) or `SV_ViewportArrayIndex` (5); `has_misc_vector`
in `gpu/recomp/mod.rs`. Count over the GTA VI pack (26 698 programs, 2026-11-24):

- 1 937 programs export `pos1`: 31 VS and 1 906 MS. Every one of them declares
  `SV_ClipDistance` or `SV_CullDistance`, and none exports `pos2` or `pos3`.
- No signature in the pack (any stage, input or output) declares system value 4 or 5.
- In `VS_Passthrough_36b96a91` `pos1.xy` are the clip distances (the same registers as its
  clip-distance param) and `.zw` the constants 0 and 1.0.

So in this pack `pos1` is always the clip vector. Reading it as the misc vector without the gate
would turn clip-distance float bits into slice and viewport indices. Clip/cull distances still come
from their params, as before, in both layouts.

## Per stage

- VS: `nointerpolation uint xcl_layer : SV_RenderTargetArrayIndex` and/or
  `xcl_viewport : SV_ViewportArrayIndex` after the params in the output struct (shared with the
  rect-list GS). DXC 1.9 sets the SFI0 flag 0x2000
  (`ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer`) from these semantics in `vs_6_6`
  (`vs_misc_vector_exports_layer_and_viewport`).
- MS: the translated program stores the raw `.z` dword in one extra dword per vertex of the
  record (`MsInterface::misc_dword`). D3D12 mesh shaders take both indices as per-primitive
  outputs, so the pass-through declares `out primitives XclPrim` and takes the value of each
  triangle's first vertex, which is the provoking vertex under D3D's setting. The primitive
  attributes count toward the 32 KB mesh output limit in `passthrough_split`.
- GS (compute emulation): the indices are read from the GS's LDS output vertex at their OSG1
  register and component and stored in the same misc dword of the record.

Without either output, the record layout, the structs and the HLSL are byte-identical to the
previous output (`--recomp` over the pack gives the same per-mode HLSL byte totals).

## Not covered

- A pixel shader that reads `SV_RenderTargetArrayIndex` or `SV_ViewportArrayIndex` as an input.
- A GPU readback of a slice above 0 or a viewport above 0: the test harness renders into a single
  2D target with one viewport. The GPU tests draw slice and viewport 0 and so check only that
  pipelines carrying these outputs are created and link.
