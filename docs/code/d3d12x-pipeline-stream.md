# D3D12.X: pipeline state stream subobject numbering

Code: `crates/xcl/src/hle/d3d12x/stream_types.rs`, called first in `dev_create_pipeline_state`
(`crates/xcl/src/hle/d3d12x/real.rs`). The readers below it are unchanged.

## How a title's stream is read

`CreatePipelineState` gets a `D3D12_PIPELINE_STATE_STREAM_DESC` (size, pointer). Every reader walks it
with the PC numbering and the PC payload table in `null::subobject_payload` (type → payload size and
alignment). A subobject is a u32 type, its payload at +4 or +8, and the next one at the next 8-byte
boundary:

| reader | file | what it does with the types |
|---|---|---|
| `capture_stream_desc` | `null.rs` | shader capture; stops at the first unknown type |
| `stream_subobjects` | `real.rs` | `None` on any unknown type or a payload past the end |
| `known_subobjects` | `real.rs` | stops at the first unknown type (warns, 4 times) |
| `translated_mesh_stream_once` | `real.rs` | AS = 24, MS = 25, PS = 2; refuses a console MS container in 24 with no MS |
| `stream_pipeline` | `real.rs` | VS / PS / CS stream as a plain desc |
| `own_stream`, `fallback_stream` | `real.rs` | copies and stand-ins |

Before this change nothing in xcl knew another numbering: no translation table, and no
per-call or per-version marker. The only console layout switch xcl has is `QUEUE_SHIFTED` (the newer
queue / command list vtable). It says nothing about subobject types. The retail title in question
has the newer layout ("device method #133 called").

## PC numbering (d3d12.h)

| type | subobject | payload (size, align) |
|---|---|---|
| 0 | ROOT_SIGNATURE | 8, 8 |
| 1–6 | VS PS DS HS GS CS | 16, 8 |
| 7 | STREAM_OUTPUT | 32, 8 |
| 8 | BLEND | 328, 4 |
| 9 | SAMPLE_MASK | 4, 4 |
| 10 | RASTERIZER | 44, 4 |
| 11 | DEPTH_STENCIL | 52, 4 |
| 12 | INPUT_LAYOUT | 16, 8 |
| 13 | IB_STRIP_CUT_VALUE | 4, 4 |
| 14 | PRIMITIVE_TOPOLOGY | 4, 4 |
| 15 | RENDER_TARGET_FORMATS | 36, 4 |
| 16 | DEPTH_STENCIL_FORMAT | 4, 4 |
| 17 | SAMPLE_DESC | 8, 4 |
| 18 | NODE_MASK | 4, 4 |
| 19 | CACHED_PSO | 16, 8 |
| 20 | FLAGS | 4, 4 |
| 21 | DEPTH_STENCIL1 | 56, 4 |
| 22 | VIEW_INSTANCING | 24, 8 |
| 23 | *(not used)* | – |
| 24 | AS | 16, 8 |
| 25 | MS | 16, 8 |
| 26 | DEPTH_STENCIL2 | 60, 4 |
| 27 | RASTERIZER1 | 44, 4 |
| 28 | RASTERIZER2 | 40, 4 |

## What is known about the console numbering

- **Types 0–22 are the PC's (strong evidence).** In run32 and run36, the 40 refused pipelines
  ("mesh pipeline with its MS … in the AS slot – refused") all passed `stream_subobjects`
  first. That walk fails on any type outside the table above and on any payload past the end.
  So every subobject in those streams has a type the PC knows, and the payload sizes add up. A
  shifted numbering in 0–22 would break the walk: BLEND (328 bytes) read as SAMPLE_MASK (4 bytes)
  would put every later type at a wrong offset. No run logged "unknown subobject type"
  (`known_subobjects`). The title's VS/PS pipelines also come as streams, are read by
  `stream_pipeline` with this numbering, and draw its menus.
- **No stream contained type 23.** `stream_subobjects` would have failed on it, and the refusal
  comes after that call.
- **Console MS = 24 (inferred).** The 40 refused pipelines carry a single-stage console mesh
  container (DXIL kind MS, NGG, no amplification program; decoded by an upstream contributor in
  xcl-dev/xcl#1269, raw stream bytes not captured) in type 24, and have no type 25. A PC stream never
  puts a mesh shader in its AS slot.
- **Console AS = 23 (inferred).** It is the one value the PC leaves out, and it is what the public
  references read for the console. No stream with it has been seen.
- **Console types from 25 on: unknown.** Whether DEPTH_STENCIL2 / RASTERIZER1 / RASTERIZER2 moved
  down by one is not known. None of the 40 streams has a type ≥ 25, so their walk says nothing about
  it. A console stream with one of them is left as it is.

| console type | read as PC type | how sure |
|---|---|---|
| 0–22 | the same | size walk of 40 retail streams |
| 23 | 24 (AS) | inferred (PC gap, public references) |
| 24 | 25 (MS) | inferred (MS container in it, no 25) |
| ≥ 25 | – | unknown; `XCL_TRACE_PSO_STREAM` shows any such stream |

The hypothesis that the whole stream is shifted is not supported so far. With 0–22 unchanged, the
rewritten stream is the pipeline an earlier build made when it translated these containers as
mesh shaders (the black screen in 38b05b7). If the title renders nothing again with
`XCL_CONSOLE_STREAM_TYPES=1`, the cause is likely somewhere other than the numbering, for example
the record layout or the mesh-group split of the two-pass path, or what the title expects of these
minimap pipelines.

## Environment variables

- `XCL_TRACE_PSO_STREAM=1`: one line per distinct stream (deduplicated by its text):

  ```
  D3D12.X: PSO stream <size> B [<kind>], PC walk <ok to 0x… | stops at 0x… on type N>, console walk <…>; read as <pc|console>: <off>:<type>(<payload size>) …, …
  ```

  `kind` is the PC reading: `MS in the AS slot`, `console mesh`, `PC mesh`, `console graphics`,
  `PC graphics`, `console compute`, `PC compute` or `other`. A shader subobject shows its console
  stage (or `pc`), its size and fnv64 (`null` when empty). Topology, RT formats (count and the 8
  formats), DS format and sample desc show their values. Pointer payloads show `ptr` or `zero`. Any
  other payload shows its fnv64. No shader bytes are logged.
- `XCL_CONSOLE_STREAM_TYPES=1` (off by default): when the stream reads to the end with the console
  numbering and its type 24 holds a console MS container, its types are rewritten to the PC's (only
  the type words change: AS and MS payloads are a `D3D12_SHADER_BYTECODE` in both). That copy is
  what every reader and the driver see. The mesh-only pipeline then takes the existing two-pass
  path (compute pass, mesh pass, the title's PS). One line per pipeline:

  ```
  D3D12.X: console-numbered pipeline stream, MS <fnv64>: types [<console types>] read as the PC's [<PC types>] (shaders <type>:<fnv64> …)
  ```

  Unset, nothing changes: the refusal in `translated_mesh_stream_once` stays.

## What the next run should check

- With `XCL_TRACE_PSO_STREAM=1`, do the `[MS in the AS slot]` lines say `PC walk ok to <size>`
  (the walk ends exactly at `SizeInBytes`)? Their RT / DS formats and topology should be plausible
  for a minimap pass.
- Is there any line whose PC walk `stops at … on type 23` or on a type ≥ 26?
- With `XCL_CONSOLE_STREAM_TYPES=1`: are there 40 "console-numbered pipeline stream" lines, two-pass
  "pipeline from the console MS" lines, and does the title still reach its menus?
