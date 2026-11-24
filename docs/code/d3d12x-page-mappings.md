# D3D12.X: CopyPageMappingsBatchX (command queue #29 older / #31 newer layout)

Code: `crates/xcl/src/hle/d3d12x/page_mappings.rs`, dispatched from `queue_dispatch` in
`crates/xcl/src/hle/d3d12x/real.rs` (older-layout slot 29; on the newer layout every queue slot from
14 on sits two later, so the title calls #31).

## Evidence

- Every run of a retail game listed `ID3D12CommandQueue::#31 (newer layout)`
  as a stub, about 340 calls per run. "First called from 0x0" only means `queue_dispatch` records the
  call with caller 0 (`stubs::record_call(info, 0)`); it says nothing about the title.
- VIctory's measured queue layout (facts only, no code): older #29 = CopyPageMappingsBatchX, PresentX #30;
  the newer layout shifts every slot from 14 on by +2. Range kinds: 0 = consecutive pool pages,
  1 = one repeated page, 2 = unmapped, 3 = left as is.
- The title's executable has exactly one call through vtable offset 0xF8 on the queue it keeps for
  page mappings (read from its code; nothing of it is in the repository). At start the title:
  1. allocates N 64 KiB physical pages (`XMemAllocatePhysicalPages`), reserves a pool range
     (`XMemVirtualAlloc`) and maps them there in order (`XMemMapPhysicalPages`): pool page `i` is at
     `pool base + i * 64 KiB`;
  2. passes the pool base and N to a console device method, then creates the queue it uses for #31;
  3. reserves two ranges of the same size: a CPU view (PAGE_READWRITE) and a GPU view
     (PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS, `0x801`). `kernelx::mirrors` already pairs them
     (log: "reservation … (0x600000000 bytes, no CPU access) is the GPU view of …").
- When it streams, the same function first maps physical pages into the CPU view with
  `XMemMapPhysicalPagesScatter` (tile `t` at `CPU base + t * 64 KiB`, pool page `p`'s physical page)
  and then calls #31 for the GPU view with the same tile and pool page.

## Argument layout

`CopyPageMappingsBatchX(UINT batchCount, const BATCH* batches, void* poolBase, UINT poolPageCount, UINT flags)`

| BATCH (0x18 bytes) | |
|---|---|
| +0x00 u64 | destination address (64 KiB aligned; GPU view base + tile * 64 KiB) |
| +0x08 u32 | range count |
| +0x0C | padding (the title leaves it uninitialised) |
| +0x10 ptr | ranges |

| RANGE (0x0C bytes) | |
|---|---|
| +0x00 u32 | kind (0 consecutive, 1 repeated, 2 unmapped, 3 kept) |
| +0x04 u32 | pool page index |
| +0x08 u32 | page count |

Ranges follow each other: each advances the destination by its page count.

- `poolPageCount` is a dword in the 5th argument's 8-byte stack slot: only its low 32 bits count.
- The 6th argument (`flags`) is always 0 at the title's call site. `queue_dispatch`'s thunks take four
  arguments, so it is not read.

## What XCL does

- Decodes and checks the batches (known kind, 64 KiB aligned destination, pool page index below the
  pool page count, at most 2^20 pages per batch). A malformed call is logged once, counted ("calls
  refused" in the summary line) and changes nothing. Every call returns 0, as the stub did: the
  title ignores the result, so its type is not known.
- Keeps a table of destination page → (pool base, pool page) for the mapped pages.
- Does not move or copy any page. On the console the GPU view and the CPU view hold the same physical
  pages at the same tile, and `kernelx::mirrors` already makes textures in the GPU view read from the
  CPU view (`title_textures::mirror_source`). Mapping the pool page at the GPU view address with
  `kernelx::map_pages` would be wrong: XCL keeps a physical page in one place at a time, so it would
  take the page from the CPU view.
- Counts, for each mapped page, whether the CPU view behind it holds a physical page
  (`kernelx::physical_page_at`), and whether the destination is in a GPU view at all; the log line
  "CopyPageMappingsBatchX #n" (first call and every power of two) gives these counts.

## Unknown / not exercised

- The title only issues kind 1 with a count of 1 (map one pool page) and kind 2 with a count of n
  (unmap). Kinds 0 and 3, and kind 1 with a count above 1, are implemented from VIctory's description
  and the unit tests only.
- The meaning of the 6th argument.
- The title also has paths that change only the CPU view or only the GPU view; one maps pool pages
  into the GPU view (kind 1) without a scatter entry for the CPU view. Such a page reads the CPU view's
  contents here (the mirror), not the pool page; the "CPU view had no page" count shows how often.
  Texture reads are not redirected to pool pages through the table, and such a redirect is not simply
  pool base + index: kernelx keeps a physical page in one place, so a page the title also mapped into
  the CPU view is no longer at its pool address.
- "Outside a GPU view" above 0 would mean the destinations are not in the reservation `mirrors`
  pairs with the CPU view; that pairing is inferred from the reservations, not proven by a run.
- Whether the call is ordered with the title's other queues (it has a queue of its own); XCL applies it
  at the call.
