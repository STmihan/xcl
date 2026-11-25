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

- **Types 0–22: consistent with the PC's, not proven.** In run32 and run36, the 40 refused
  pipelines ("mesh pipeline with its MS … in the AS slot – refused") all passed
  `stream_subobjects` first. That walk fails on any type outside the table above and on any
  payload past the end. So every subobject in those streams has a type the PC table knows, and the
  payload footprints add up. The walk only checks each subobject's padded footprint, though, and
  many types share one: 8 bytes (9, 13, 14, 16, 18, 20), 16 (0, 17), 24 (1–6, 12, 19, 24, 25),
  40 (7, 15), 48 (10, 27, 28), 64 (21, 26). A shift is caught only when a type with its own
  footprint (BLEND 336, DEPTH_STENCIL 56, VIEW_INSTANCING 32) is in the stream, and which types
  these streams hold has not been logged. No run logged "unknown subobject type" (`known_subobjects`).
  Whether the title's VS/PS pipelines come as streams too is not established: `graphics_pso`
  serves both the plain and the stream path, and its log line does not tell them apart.
- **No refused stream contained type 23**, and none a type 25 with readable bytecode.
  `stream_subobjects` would have failed on 23, and the refusal needs no MS. Types 26–28 may be there:
  the PC table accepts them.
- **Console MS = 24 (inferred).** The 40 refused pipelines carry a single-stage console mesh
  container (DXIL kind MS, NGG, no amplification program; decoded by an upstream contributor in
  xcl-dev/xcl#1269, raw stream bytes not captured) in type 24, and have no type 25. A PC stream never
  puts a mesh shader in its AS slot.
- **Console AS = 23 (inferred).** It is the one value the PC leaves out, and it is what the public
  references read for the console. No stream with it has been seen.
- **Console types from 25 on: unknown.** Whether DEPTH_STENCIL2 / RASTERIZER1 / RASTERIZER2 moved
  down by one is not known.

| console type | read as PC type | how sure |
|---|---|---|
| 0–22 | the same | consistent with the footprint walk of 40 retail streams; to be confirmed by the trace |
| 23 | 24 (AS) | inferred (PC gap, public references) |
| 24 | 25 (MS) | inferred (MS container in it, no 25) |
| ≥ 25 | – | unknown; read as the PC's in an "MS only" remap (below) |

The hypothesis that the whole stream is shifted is neither supported nor ruled out: the footprint
walk allows permutations within a footprint class. The trace prints the value of every 4-byte
payload, so a permutation among those (sample mask, strip cut, topology, DS format, node mask,
flags) shows up on the next run. With 0–22 unchanged, the
rewritten stream is the pipeline an earlier build made when it translated these containers as
mesh shaders (the black screen in 38b05b7). If the title renders nothing again with the
console numbering on, the cause is likely somewhere other than the numbering, for example
the record layout or the mesh-group split of the two-pass path, or what the title expects of these
minimap pipelines.

## Environment variables

- `XCL_TRACE_PSO_STREAM=1`: one line per distinct stream (deduplicated by its text):

  ```
  D3D12.X: PSO stream <size> B [<kind>], PC walk <ok to 0x… | stops at 0x… on type N>, console walk <…>; read as <pc|console|MS only>: <off>:<type>(<payload size>) …, …
  ```

  `kind` is the PC reading: `MS in the AS slot`, `console mesh`, `PC mesh`, `console graphics`,
  `PC graphics`, `console compute`, `PC compute` or `other`. A shader subobject shows its console
  stage (or `pc`), its size and fnv64 (`null` when empty). Topology, RT formats (count and the 8
  formats) and sample desc show their values, as does every 4-byte payload (`=0x…`). Pointer payloads
  show `ptr` or `zero`. Any other payload shows its fnv64. No shader bytes are logged.
- `XCL_CONSOLE_STREAM_TYPES` (on by default; `0` turns it off, `1` is on too, any other value
  ends the process when the D3D12 backend starts, naming the variable): a stream whose type 24 holds a console MS
  container is rewritten to the PC numbering before any reader sees it. Only the type words change:
  AS and MS payloads are a `D3D12_SHADER_BYTECODE` either way. There are two cases:
  - **console**: the stream reads to the end with the console table (types 0–24). Every type goes
    through it: 23 → 24, 24 → 25.
  - **MS only**: the console table stops on a type ≥ 25, the PC table reads the stream to the end,
    and there is no type 25 at all. Only 24 → 25 is changed; the other types stay as the PC read them.

  A stream with an MS container in 24 that fits neither case (type 25 present, even with null
  bytecode, or no numbering reads it to the end) is left as it is and logged once:

  ```
  D3D12.X: pipeline stream MS <fnv64> in type 24 left as it is: <why>; types [<PC reading>]
  ```

  The rewritten copy is what every reader and the driver see. The mesh-only pipeline then takes the
  existing two-pass path (compute pass, mesh pass, the title's PS). One line per pipeline:

  ```
  D3D12.X: console-numbered pipeline stream, MS <fnv64> (<console|MS only>): types [<title's types>] read as the PC's [<PC types>] (shaders <PC type>:<fnv64> …)
  ```

  With `XCL_CONSOLE_STREAM_TYPES=0` nothing changes: the refusal in `translated_mesh_stream_once`
  stays, and the backend logs once:

  ```
  D3D12.X: XCL_CONSOLE_STREAM_TYPES=0: pipeline streams in the console's subobject numbering are passed as they are
  ```

  Why it is on by default: VIctory, a fork of this emulator, has always read console types 23 and
  24 as AS and MS, and all of the title's world mesh shaders translate there. The numbering is
  still inferred, not measured (the table above).

## What the next run should check

- Set both variables. Do the `[MS in the AS slot]` lines say `PC walk ok to <size>` (the walk ends
  exactly at `SizeInBytes`)? Their RT / DS formats and topology should be plausible for a minimap
  pass. Which types do they hold, and is any of them 26–28?
- Is there any line whose PC walk `stops at … on type 23` or on a type ≥ 26?
- With the console numbering on (the default): are there 40 "console-numbered pipeline stream" lines, two-pass
  "pipeline from the console MS" lines, and does the title still reach its menus?
