#include "cuda_passes.h"
#include "gather.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_PASS_CHECK(call)                                                 \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[cuda_passes] %s:%d %s -> %s\n", __FILE__,  \
                __LINE__, #call, cudaGetErrorString(_e));                     \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// The dense part of the grid is one int per cell, and the cell count grows as
// r^-3 while the PPM schedule shrinks r — so on a card with a few GB the grid
// runs out of memory long before the photon map does.  The budget below is the
// point where the build stops subdividing and starts growing the cell instead.
//
// Growing the cell is the safe direction: the gather derives its cell index
// from grid.cell_size and walks 3x3x3 cells, which covers the search sphere for
// any cell_size >= r.  Clamping the dimensions while leaving the cell at r
// would not — the two ends would disagree about which cell a point is in, and
// the gather would silently lose photons.
static constexpr long long kMaxGridCells = 32LL << 20; // 32M cells = 128 MB

// Every hash table is sized by one rule: the smallest power of two that is at
// least table_load (--table-load, 4 by default) times the number of entries
// it will hold, and never fewer than kMinHashBuckets.  Sizing from the
// entries actually filed is what keeps the load factor the same with and
// without culling: a culled table holds fewer entries and gets
// proportionally fewer buckets, never a fuller table.  The tables filled
// during the photon trace (stochastic, rectified, mlrect with
// --mlrect-in-trace) cannot know their entries in advance, and are sized from
// the most there can be, the photon capacity.
//
// The caps are memory limits, not part of the rule: 2^27 buckets are 512 MB of
// offsets for the full grids (and as much again of build scratch), 1.5 GB of
// 12-byte buckets for the rectified ones.  The original stochastic grid keeps
// a whole 52-byte photon in every bucket, so its cap is lower.  A table that
// reaches its cap is fuller than table_load asks for, and the build says so.
static constexpr long long kMinHashBuckets = 1LL << 10;
static constexpr long long kMaxHashBuckets = 1LL << 27;
static constexpr long long kMaxStochasticPhotonBuckets = 1LL << 25;

static constexpr int kBlock = 256;

static inline int blocksFor(int n) { return (n + kBlock - 1) / kBlock; }

static long long tableBuckets(double entries, float tableLoad, long long cap, const char* table)
{
    const double load = tableLoad > 0.f ? (double)tableLoad : 4.0;
    const double want = std::ceil(std::max(entries, 0.0) * load);
    long long buckets = kMinHashBuckets;
    while ((double)buckets < want && buckets < cap)
        buckets <<= 1;
    if ((double)buckets < want) {
        static std::vector<std::string> warned;
        if (std::find(warned.begin(), warned.end(), table) == warned.end()) {
            warned.push_back(table);
            std::fprintf(stderr, "\n[cuda_passes] %s table capped at %lld buckets for %.0f entries"
                                 " (%.2f buckets per entry, --table-load asks for %.2f)\n",
                table, buckets, entries, (double)buckets / std::max(entries, 1.0), load);
        }
    }
    return buckets;
}

// Uniform grid key: the cell's position in the dense array.  Must agree exactly
// with gatherPhotonsGrid in device.cu; photonCellCoord is shared for that
// reason.
__global__ void photonCellKeysKernel(const Photon* __restrict__ photons, int n,
    float3 bmin, float cellSize, int3 dims,
    int* __restrict__ keys, int* __restrict__ ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 p = photons[i].pos;
    int ix = photonCellCoord(p.x, bmin.x, cellSize);
    int iy = photonCellCoord(p.y, bmin.y, cellSize);
    int iz = photonCellCoord(p.z, bmin.z, cellSize);
    ix = max(0, min(ix, dims.x - 1));
    iy = max(0, min(iy, dims.y - 1));
    iz = max(0, min(iz, dims.z - 1));

    keys[i] = (iz * dims.y + iy) * dims.x + ix;
    ids[i] = i;
}

// Hash grid key: the bucket the photon's cell hashes to.  Must agree exactly
// with gatherPhotonsHash in device.cu; photonCellCoord and hashCell are shared
// for that reason.  No clamping — every integer cell has a bucket.
__global__ void photonHashKeysKernel(const Photon* __restrict__ photons, int n,
    float3 origin, float cellSize, unsigned int hashMask,
    int* __restrict__ keys, int* __restrict__ ids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 p = photons[i].pos;
    const int ix = photonCellCoord(p.x, origin.x, cellSize);
    const int iy = photonCellCoord(p.y, origin.y, cellSize);
    const int iz = photonCellCoord(p.z, origin.z, cellSize);

    keys[i] = (int)(hashCell(ix, iy, iz) & hashMask);
    ids[i] = i;
}

// The levels a multi-level grid is built over, passed to its kernels by value.
struct LevelSet {
    int num;
    int levels[MAX_LEVELS];
    float s0;
    float3 origin;
    const unsigned int* cullMask; // nullptr = keep every entry
    unsigned int cullMaskBits;
};

// One entry per (photon, level in use), at e = photon * num + j.  Must agree
// exactly with gatherPhotonsMultiLevel in gather.h — the same cell
// (levelCellSize, photonCellCoord, the level origin) and the same hash
// (hashCell4) — and, with culling, with markLookupCellsKernel: an entry is
// kept when the cell it is filed under is one some lookup on that level
// searches.
//
// Without culling every entry is kept and its key is already the bucket
// (hash & mask).  With culling the full hash and a keep flag are written, the
// table size is decided after the compaction, and the keys masked then.
__global__ void multiLevelEntriesKernel(const Photon* __restrict__ photons, int n,
    LevelSet ls, unsigned int mask, unsigned int* __restrict__ keys,
    int* __restrict__ values, unsigned char* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 p = photons[i].pos;
    for (int j = 0; j < ls.num; ++j) {
        const int l = ls.levels[j];
        const float cs = levelCellSize(ls.s0, l);
        const int cx = photonCellCoord(p.x, ls.origin.x, cs);
        const int cy = photonCellCoord(p.y, ls.origin.y, cs);
        const int cz = photonCellCoord(p.z, ls.origin.z, cs);
        const size_t e = (size_t)i * ls.num + j;
        keys[e] = hashCell4(cx, cy, cz, l) & mask;
        values[e] = (int)((unsigned int)i | ((unsigned int)l << LEVEL_SHIFT));
        if (flags)
            flags[e] = cullMaskTest(ls.cullMask, ls.cullMaskBits, cx, cy, cz, l) ? 1 : 0;
    }
}

// The multi-level rectified grid's build: every stored photon inserted once
// per level in use, into the bucket of its own cell on that level — the same
// cells, hash and culling test as multiLevelEntriesKernel.
__global__ void multiLevelInsertKernel(const Photon* __restrict__ photons, int n,
    LevelSet ls, unsigned int mask, int frame, int seed,
    int* __restrict__ count, unsigned long long* __restrict__ best)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 p = photons[i].pos;
    for (int j = 0; j < ls.num; ++j) {
        const int l = ls.levels[j];
        const float cs = levelCellSize(ls.s0, l);
        const int cx = photonCellCoord(p.x, ls.origin.x, cs);
        const int cy = photonCellCoord(p.y, ls.origin.y, cs);
        const int cz = photonCellCoord(p.z, ls.origin.z, cs);
        if (ls.cullMask && !cullMaskTest(ls.cullMask, ls.cullMaskBits, cx, cy, cz, l))
            continue;
        insertRepresentative(count, best, hashCell4(cx, cy, cz, l) & mask,
            (unsigned int)i, l, frame, seed);
    }
}

// With culling, how many insertions multiLevelInsertKernel is going to make —
// the same cells and mask test, counted only — so that the table can be sized
// from them like every other table.  One atomic per warp.
__global__ void multiLevelCountKernel(const Photon* __restrict__ photons, int n,
    LevelSet ls, unsigned long long* __restrict__ total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c = 0;
    if (i < n) {
        const float3 p = photons[i].pos;
        for (int j = 0; j < ls.num; ++j) {
            const int l = ls.levels[j];
            const float cs = levelCellSize(ls.s0, l);
            if (cullMaskTest(ls.cullMask, ls.cullMaskBits, photonCellCoord(p.x, ls.origin.x, cs),
                    photonCellCoord(p.y, ls.origin.y, cs), photonCellCoord(p.z, ls.origin.z, cs), l))
                ++c;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        c += __shfl_down_sync(0xFFFFFFFFu, c, offset);
    if ((threadIdx.x & 31) == 0 && c > 0)
        atomicAdd(total, (unsigned long long)c);
}

// The reservoir self-test: `reps` buckets, `n` insertions into each, every
// repetition with a seed of its own.  `scattered` spreads the insertions of a
// bucket across the whole launch, so they come from different warps and
// blocks at different times instead of from neighbouring threads.
__global__ void reservoirSelfTestKernel(int n, int reps, bool scattered,
    int* __restrict__ count, unsigned long long* __restrict__ best)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * reps)
        return;
    const int rep = scattered ? t % reps : t / n;
    const int j = scattered ? t / reps : t % n;
    insertRepresentative(count, best, (unsigned int)rep, (unsigned int)j, 0, 0, rep);
}

// --inline-positions: the position of every entry's photon, in entry order.
__global__ void entryPositionsKernel(const int* __restrict__ entries, int n,
    const Photon* __restrict__ photons, float4* __restrict__ positions)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float3 p = photons[(unsigned int)entries[i] & LEVEL_INDEX_MASK].pos;
    positions[i] = make_float4(p.x, p.y, p.z, 0.f);
}

__global__ void maskKeysKernel(unsigned int* __restrict__ keys, int n, unsigned int mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        keys[i] &= mask;
}

__global__ void countCellsKernel(const int* __restrict__ keys, int n,
    int* __restrict__ counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    atomicAdd(&counts[keys[i]], 1);
}

// The query pass, one kernel per structure (and per --stats on or off): the
// structure is a template parameter, so no kernel carries another one's search
// and each gets the registers its own code needs.  One thread per pixel; the
// launch parameters come by value, in the constant bank, as they do for an
// OptiX launch.
template <int ACCEL, bool STATS, bool INLINE>
__global__ void queryKernel(const __grid_constant__ Params P)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= (int)(P.width * P.height))
        return;
    queryPixel<ACCEL, STATS, INLINE>(P, pixel);
}

// Photon culling, the lookup side: mark the cells every lookup of the frame
// is going to search.
//
// A lookup of radius r at level l = levelOf(r) searches the cells its sphere
// overlaps at level l (cellRange), and those contain every point within r of
// it.  So a photon that some lookup of this frame can use lies in a cell that
// lookup marked here, and the photon pass, which keeps every photon that falls
// in a marked cell on some level in use, keeps it.  Hash collisions only ever
// set bits nobody asked for: they can keep a photon no lookup needs, never
// drop one that a lookup does.  The argument is about which photons lie
// within r, not about how a structure goes looking for them, so it holds for
// every search structure; the multi-level ones visit exactly these cells.
//
// Neighbouring pixels mark mostly the same cells, so a bit is read first and
// the atomic only issued while it is still clear.  A stale read can only show
// fewer bits than are set — the mask is cleared before this launch and bits
// are only ever set — so at worst it costs a redundant atomicOr.
__global__ void markLookupCellsKernel(const __grid_constant__ Params P)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= (int)(P.width * P.height))
        return;
    const GatherPoint& gp = P.gather_points[pixel];
    if (!gp.valid)
        return;

    const int level = gp.level;
    int3 lo, hi;
    cellRange3(gp.pos, gp.r, P.level_origin, levelCellSize(P.level_s0, level), lo, hi);
    for (int z = lo.z; z <= hi.z; ++z) {
        for (int y = lo.y; y <= hi.y; ++y) {
            for (int x = lo.x; x <= hi.x; ++x) {
                const unsigned int b = cullMaskBit(x, y, z, level, P.cull_mask_bits);
                unsigned int* word = P.cull_mask + (b >> 5);
                const unsigned int bit = 1u << (b & 31u);
                if (!(*word & bit))
                    atomicOr(word, bit);
            }
        }
    }
    // --hier-mask: the same cells' parents at every coarser level in use.  A
    // level-l cell lies in exactly one cell of every coarser level (they share
    // the origin and double in size), cell c of level l in cell c >> d of level
    // l + d — an arithmetic shift, so negative coordinates round down too.
    if (P.hier_mask) {
        for (int i = 0; i < P.num_levels; ++i) {
            const int lp = P.levels[i];
            const int d = lp - level;
            if (d <= 0)
                continue;
            for (int z = lo.z >> d; z <= (hi.z >> d); ++z) {
                for (int y = lo.y >> d; y <= (hi.y >> d); ++y) {
                    for (int x = lo.x >> d; x <= (hi.x >> d); ++x) {
                        const unsigned int b = cullMaskBit(x, y, z, lp, P.cull_mask_bits);
                        unsigned int* word = P.cull_parent + (b >> 5);
                        const unsigned int bit = 1u << (b & 31u);
                        if (!(*word & bit))
                            atomicOr(word, bit);
                    }
                }
            }
        }
    }
}

// --- Splatting: the visible-point grid ------------------------------------
//
// The visible points (the frame's gather points) are filed like the photons
// of the multi-level hash grid, but under every cell their sphere overlaps
// rather than the one cell they lie in, so that a photon only needs to look in
// its own cell (see splatDeposit in device.cu).  Five steps:
//   1. count: how many cells each point's sphere overlaps on its level;
//      an exclusive scan of the counts gives each point its first entry
//   2. the table size, table_load buckets per entry
//   3. emit: a 64-bit key per (point, cell), the bucket above the entry
//      (point | level << LEVEL_SHIFT) — the order a stable sort by bucket
//      keeps, so a point's entries in one bucket end up next to each other
//   4. sort by the bucket bits only, then drop repeated keys: two cells of one
//      point that share a bucket would otherwise make the point count twice
//   5. a histogram of the buckets and its exclusive scan -> cell_start, the
//      entries' low halves -> values
// Count and emit run the same cellRange on the same level, so they agree on
// every point's cells.

// The level a visible point is filed on: 0 in the single-level grid, else
// the smallest level whose cells are at least F x r wide.
__device__ __forceinline__ int vpLevelOf(const Params& P, float r)
{
    return P.vpgrid.cell_size > 0.f ? 0 : levelOf(P.splat_cell_factor * r, P.level_s0);
}

__global__ void vpCountKernel(const __grid_constant__ Params P, int* __restrict__ counts,
    int* __restrict__ levelFlags)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (int)(P.width * P.height))
        return;
    const GatherPoint& gp = P.gather_points[j];
    if (!gp.valid)
        return; // invalid points are not filed (counts were cleared)
    const int lv = vpLevelOf(P, gp.r);
    int3 lo, hi;
    cellRange3(gp.pos, gp.r, P.level_origin, vpCellSize(P.vpgrid, P.level_s0, lv), lo, hi);
    counts[j] = (hi.x - lo.x + 1) * (hi.y - lo.y + 1) * (hi.z - lo.z + 1);
    if (!levelFlags[lv])
        levelFlags[lv] = 1;
}

__global__ void vpEmitKernel(const __grid_constant__ Params P, const int* __restrict__ offsets,
    unsigned int mask, unsigned long long* __restrict__ keys)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (int)(P.width * P.height))
        return;
    const GatherPoint& gp = P.gather_points[j];
    if (!gp.valid)
        return;
    const int lv = vpLevelOf(P, gp.r);
    int3 lo, hi;
    cellRange3(gp.pos, gp.r, P.level_origin, vpCellSize(P.vpgrid, P.level_s0, lv), lo, hi);
    const unsigned long long entry = (unsigned int)j | ((unsigned int)lv << LEVEL_SHIFT);
    unsigned long long* out = keys + offsets[j];
    for (int z = lo.z; z <= hi.z; ++z)
        for (int y = lo.y; y <= hi.y; ++y)
            for (int x = lo.x; x <= hi.x; ++x)
                *out++ = ((unsigned long long)(hashCell4(x, y, z, lv) & mask) << 32) | entry;
}

// Over the de-duplicated keys (as many as *count): the histogram of their
// buckets, and their entries.
__global__ void vpBucketKernel(const unsigned long long* __restrict__ keys,
    const int* __restrict__ count, int* __restrict__ buckets, unsigned int* __restrict__ values,
    const GatherPoint* __restrict__ points, float4* __restrict__ inlinePoints)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *count)
        return;
    const unsigned long long k = keys[i];
    atomicAdd(&buckets[(unsigned int)(k >> 32)], 1);
    values[i] = (unsigned int)k;
    // --inline-positions: the point's position and radius next to its entry
    if (inlinePoints) {
        const GatherPoint& gp = points[(unsigned int)k & LEVEL_INDEX_MASK];
        inlinePoints[i] = make_float4(gp.pos.x, gp.pos.y, gp.pos.z, gp.r);
    }
}

// The end of a splatting frame: the same formula a gather ends with (see
// photonIrradiance and queryPixel in gather.h), with the flux summed over the
// accumulator copies.  Invalid points keep just their direct light.
__global__ void splatFinalizeKernel(const __grid_constant__ Params P)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = (int)(P.width * P.height);
    if (j >= n)
        return;
    const GatherPoint gp = P.gather_points[j];
    float3 radiance = gp.direct;
    if (gp.valid) {
        float3 flux = make_float3(0.f, 0.f, 0.f);
        for (int s = 0; s < P.splat_stripes; ++s) {
            const float* a = P.splat_acc + ((size_t)s * n + j) * 3;
            flux = make_float3(flux.x + a[0], flux.y + a[1], flux.z + a[2]);
        }
        const float r2 = gp.r * gp.r;
        radiance += gp.weight * (flux * (3.f / (M_PIf * r2)));
    }
    P.query_radiance[j] = radiance;
}

__global__ void accumulateKernel(float3* __restrict__ accum,
    uchar4* __restrict__ frame,
    const float3* __restrict__ radiance,
    int n, int frameIndex)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    const float3 r = radiance[i];
    const float3 prev = (frameIndex == 0) ? make_float3(0.f, 0.f, 0.f) : accum[i];
    const float3 acc = make_float3(prev.x + r.x, prev.y + r.y, prev.z + r.z);
    accum[i] = acc;

    if (!frame)
        return;

    // Running mean, clamped, gamma 2.0 — the same tone map the path tracer
    // applies at the end of __raygen__rg.
    const float nf = float(frameIndex + 1);
    const float mr = sqrtf(fminf(fmaxf(acc.x / nf, 0.f), 1.f));
    const float mg = sqrtf(fminf(fmaxf(acc.y / nf, 0.f), 1.f));
    const float mb = sqrtf(fminf(fmaxf(acc.z / nf, 0.f), 1.f));
    frame[i] = make_uchar4((unsigned char)(mr * 255.f),
        (unsigned char)(mg * 255.f),
        (unsigned char)(mb * 255.f), 255u);
}

static void ensureCapacity(void** ptr, size_t& capacity, size_t needed)
{
    if (*ptr && capacity >= needed)
        return;
    if (*ptr)
        CUDA_PASS_CHECK(cudaFree(*ptr));
    capacity = needed + needed / 4; // a little headroom, as on the host side
    CUDA_PASS_CHECK(cudaMalloc(ptr, capacity));
}

// --inline-positions: the positions next to the `entries` sorted entries
// (part of the build, so timed with it); otherwise none.  Returns the bytes.
static long long attachPositions(Params& params, GridScratch& s, int entries, cudaStream_t stream)
{
    params.grid.positions = nullptr;
    if (!params.inline_positions || entries <= 0)
        return 0;
    size_t bytes = s.positions_cap * sizeof(float4);
    ensureCapacity((void**)&s.positions, bytes, (size_t)entries * sizeof(float4));
    s.positions_cap = bytes / sizeof(float4);
    entryPositionsKernel<<<blocksFor(entries), kBlock, 0, stream>>>(s.ids_out, entries,
        params.photon_map, s.positions);
    params.grid.positions = s.positions;
    return (long long)entries * (long long)sizeof(float4);
}

static long long dimsForCellSize(const float ext[3], float cellSize, int3& dims)
{
    dims.x = std::max(1, (int)std::ceil(ext[0] / cellSize));
    dims.y = std::max(1, (int)std::ceil(ext[1] / cellSize));
    dims.z = std::max(1, (int)std::ceil(ext[2] / cellSize));
    return (long long)dims.x * (long long)dims.y * (long long)dims.z;
}

// Makes every build buffer large enough for `storedCount` photons and
// `numKeys` distinct key values (cells for the uniform grid, buckets for the
// hash grid).  Only allocates when something has outgrown its buffer.
static void reserveBuildBuffers(GridScratch& s, int storedCount, long long numKeys)
{
    const size_t numOffsets = (size_t)numKeys + 1;
    ensureCapacity((void**)&s.cell_start, s.cell_start_cap, numOffsets * sizeof(int));
    ensureCapacity((void**)&s.cell_counts, s.cell_counts_cap, numOffsets * sizeof(int));
    if (!s.keys_in || s.photon_cap < (size_t)storedCount) {
        if (s.keys_in) {
            CUDA_PASS_CHECK(cudaFree(s.keys_in));
            CUDA_PASS_CHECK(cudaFree(s.keys_out));
            CUDA_PASS_CHECK(cudaFree(s.ids_in));
            CUDA_PASS_CHECK(cudaFree(s.ids_out));
        }
        s.photon_cap = (size_t)storedCount + (size_t)storedCount / 4;
        const size_t bytes = s.photon_cap * sizeof(int);
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.keys_in, bytes));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.keys_out, bytes));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.ids_in, bytes));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.ids_out, bytes));
    }
}

// The part of the build both grids share, run once keys_in and ids_in have
// been written: group the photon ids by key.
//
// Keys are integers in [0, numKeys).  A histogram of them followed by an
// exclusive prefix sum gives each key's starting offset; a stable radix sort of
// the (key, id) pairs then lays the ids down in exactly that order.
static void groupPhotonsByKey(GridScratch& s, int storedCount, long long numKeys,
    cudaStream_t stream)
{
    const int numOffsets = (int)(numKeys + 1);

    // Only the bits the key actually uses need sorting.
    int endBit = 1;
    while ((1LL << endBit) < numKeys)
        ++endBit;

    size_t scanBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes,
        s.cell_counts, s.cell_start, numOffsets, stream));
    size_t sortBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sortBytes,
        s.keys_in, s.keys_out, s.ids_in, s.ids_out, storedCount, 0, endBit, stream));
    ensureCapacity(&s.temp, s.temp_cap, std::max(scanBytes, sortBytes));

    // Histogram of keys -> exclusive prefix sum = offsets.  numOffsets entries,
    // the last count staying zero, so cell_start[numKeys] ends up holding the
    // total and every entry can be read as a [start, end) pair.
    CUDA_PASS_CHECK(cudaMemsetAsync(s.cell_counts, 0,
        (size_t)numOffsets * sizeof(int), stream));
    countCellsKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        s.keys_in, storedCount, s.cell_counts);
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(s.temp, scanBytes,
        s.cell_counts, s.cell_start, numOffsets, stream));

    // Photon ids grouped by key.  The radix sort is stable and the ids go in
    // ascending, so each key lists its photons in photon-map order — the same
    // order every run, which is what makes two renders bit-for-bit comparable.
    CUDA_PASS_CHECK(cub::DeviceRadixSort::SortPairs(s.temp, sortBytes,
        s.keys_in, s.keys_out, s.ids_in, s.ids_out, storedCount, 0, endBit, stream));
}

void buildPhotonGridGPU(Params& params, int storedCount, float radius,
    GridScratch& s, cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    storedCount = std::min(storedCount, params.photon_capacity);
    if (storedCount <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }

    // Bounds: the scene box padded by one radius, not a reduction over the
    // photons.  Photons only land on geometry, so the scene box contains them,
    // and taking it from the host means the build never has to read anything
    // back from the device.  The padding keeps a photon on the boundary inside
    // the grid the way the old per-photon box did.
    const float r = radius;
    const float3 bmin = make_float3(params.scene_aabb_min.x - r,
        params.scene_aabb_min.y - r,
        params.scene_aabb_min.z - r);
    const float3 bmax = make_float3(params.scene_aabb_max.x + r,
        params.scene_aabb_max.y + r,
        params.scene_aabb_max.z + r);
    const float ext[3] = { bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z };

    float cellSize = r;
    int3 dims;
    long long cells = dimsForCellSize(ext, cellSize, dims);
    bool capped = false;
    for (int guard = 0; cells > kMaxGridCells && guard < 32; ++guard) {
        capped = true;
        const double grow = std::cbrt((double)cells / (double)kMaxGridCells);
        cellSize *= (float)std::max(grow, 1.01);
        cells = dimsForCellSize(ext, cellSize, dims);
    }

    reserveBuildBuffers(s, storedCount, cells);

    // 1. one cell index per photon
    photonCellKeysKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        params.photon_map, storedCount, bmin, cellSize, dims, s.keys_in, s.ids_in);

    // 2 and 3. histogram -> prefix sum -> sort
    groupPhotonsByKey(s, storedCount, cells, stream);

    params.grid.aabb_min = bmin;
    params.grid.aabb_max = bmax;
    params.grid.dims = dims;
    params.grid.cell_size = cellSize;
    params.grid.hash_mask = 0;
    params.grid.cell_start = s.cell_start;
    params.grid.grid_photon_ids = s.ids_out;
    params.grid.positions = nullptr;
    params.accel = ACCEL_UNIFORM_GRID;

    if (stats) {
        stats->cells = cells;
        stats->dims = dims;
        stats->cell_size = cellSize;
        stats->radius = r;
        stats->photons = storedCount;
        stats->cell_size_capped = capped;
        stats->bytes = (cells + 1) * (long long)sizeof(int)
            + (long long)storedCount * (long long)sizeof(int);
    }
}

void buildPhotonHashGridGPU(Params& params, int storedCount, float radius,
    GridScratch& s, cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    storedCount = std::min(storedCount, params.photon_capacity);
    if (storedCount <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }

    // The cell size is always exactly r.  The hash grid has no dense array, so
    // there is no memory budget that could force the cell to grow.  The scene
    // box is used only as the origin of the cell coordinates (padded by r, as
    // for the uniform grid, so coordinates of photons are never negative);
    // there is no upper bound.
    const float r = radius;
    const float3 origin = make_float3(params.scene_aabb_min.x - r,
        params.scene_aabb_min.y - r,
        params.scene_aabb_min.z - r);

    // Number of buckets: table_load per photon, a power of two (see
    // tableBuckets), which turns "hash mod buckets" into "hash & mask".
    const long long buckets = tableBuckets(storedCount, params.table_load, kMaxHashBuckets, "hash");
    const unsigned int mask = (unsigned int)(buckets - 1);

    reserveBuildBuffers(s, storedCount, buckets);

    // 1. one bucket index per photon
    photonHashKeysKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        params.photon_map, storedCount, origin, r, mask, s.keys_in, s.ids_in);

    // 2 and 3. histogram -> prefix sum -> sort, exactly as for the uniform grid
    groupPhotonsByKey(s, storedCount, buckets, stream);

    params.grid.aabb_min = origin;
    params.grid.aabb_max = make_float3(params.scene_aabb_max.x + r,
        params.scene_aabb_max.y + r,
        params.scene_aabb_max.z + r);
    params.grid.dims = make_int3(0, 0, 0);
    params.grid.cell_size = r;
    params.grid.hash_mask = mask;
    params.grid.cell_start = s.cell_start;
    params.grid.grid_photon_ids = s.ids_out;
    const long long positionBytes = attachPositions(params, s, storedCount, stream);
    params.accel = ACCEL_HASH_GRID;

    if (stats) {
        stats->buckets = buckets;
        stats->cell_size = r;
        stats->radius = r;
        stats->photons = storedCount;
        stats->bytes = (buckets + 1) * (long long)sizeof(int)
            + (long long)storedCount * (long long)sizeof(int) + positionBytes;
    }
}

void buildMultiLevelHashGridGPU(Params& params, int storedCount, GridScratch& s,
    cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    storedCount = std::min(storedCount, params.photon_capacity);
    if (storedCount <= 0 || params.num_levels <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }
    // An entry packs the photon index into the low LEVEL_SHIFT bits.
    if ((unsigned int)storedCount > LEVEL_INDEX_MASK + 1u) {
        std::fprintf(stderr, "[cuda_passes] %d photons do not fit the %d-bit entry index\n",
            storedCount, LEVEL_SHIFT);
        std::abort();
    }
    const long long candidates = (long long)storedCount * params.num_levels;
    if (candidates > (long long)INT_MAX / 2) {
        std::fprintf(stderr, "[cuda_passes] %lld multi-level entries are too many\n", candidates);
        std::abort();
    }
    const int n = (int)candidates;

    LevelSet ls;
    ls.num = params.num_levels;
    for (int j = 0; j < MAX_LEVELS; ++j)
        ls.levels[j] = params.levels[j];
    ls.s0 = params.level_s0;
    ls.origin = params.level_origin;
    ls.cullMask = params.culling ? params.cull_mask : nullptr;
    ls.cullMaskBits = params.cull_mask_bits;

    auto bucketsFor = [&](long long entries) {
        return tableBuckets((double)entries, params.table_load, kMaxHashBuckets, "mlhash");
    };

    // keys_in / ids_in hold the entries the table is grouped from.
    reserveBuildBuffers(s, n, kMinHashBuckets);
    unsigned int* keys = reinterpret_cast<unsigned int*>(s.keys_in);
    int entries = n;
    long long buckets;
    if (!params.culling) {
        // Every entry is kept, so the table size is known up front and the
        // kernel can write buckets straight away.
        buckets = bucketsFor(n);
        multiLevelEntriesKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
            params.photon_map, storedCount, ls, (unsigned int)(buckets - 1),
            keys, s.ids_in, nullptr);
    } else {
        if (!s.ml_keys || s.ml_cap < (size_t)n) {
            if (s.ml_keys) {
                CUDA_PASS_CHECK(cudaFree(s.ml_keys));
                CUDA_PASS_CHECK(cudaFree(s.ml_values));
                CUDA_PASS_CHECK(cudaFree(s.ml_flags));
            }
            s.ml_cap = (size_t)n + (size_t)n / 4;
            CUDA_PASS_CHECK(cudaMalloc((void**)&s.ml_keys, s.ml_cap * sizeof(unsigned int)));
            CUDA_PASS_CHECK(cudaMalloc((void**)&s.ml_values, s.ml_cap * sizeof(int)));
            CUDA_PASS_CHECK(cudaMalloc((void**)&s.ml_flags, s.ml_cap));
        }
        if (!s.ml_selected)
            CUDA_PASS_CHECK(cudaMalloc((void**)&s.ml_selected, sizeof(int)));

        multiLevelEntriesKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
            params.photon_map, storedCount, ls, 0xFFFFFFFFu,
            s.ml_keys, s.ml_values, s.ml_flags);

        // Compact keys and values with the same flags, then read back how
        // many survived: the table is sized from that.
        size_t bytesK = 0, bytesV = 0;
        CUDA_PASS_CHECK(cub::DeviceSelect::Flagged(nullptr, bytesK, s.ml_keys, s.ml_flags,
            keys, s.ml_selected, n, stream));
        CUDA_PASS_CHECK(cub::DeviceSelect::Flagged(nullptr, bytesV, s.ml_values, s.ml_flags,
            s.ids_in, s.ml_selected, n, stream));
        ensureCapacity(&s.temp, s.temp_cap, std::max(bytesK, bytesV));
        CUDA_PASS_CHECK(cub::DeviceSelect::Flagged(s.temp, bytesK, s.ml_keys, s.ml_flags,
            keys, s.ml_selected, n, stream));
        CUDA_PASS_CHECK(cub::DeviceSelect::Flagged(s.temp, bytesV, s.ml_values, s.ml_flags,
            s.ids_in, s.ml_selected, n, stream));
        CUDA_PASS_CHECK(cudaMemcpyAsync(&entries, s.ml_selected, sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        CUDA_PASS_CHECK(cudaStreamSynchronize(stream));

        buckets = bucketsFor(entries);
        if (entries > 0)
            maskKeysKernel<<<blocksFor(entries), kBlock, 0, stream>>>(
                keys, entries, (unsigned int)(buckets - 1));
    }

    if (entries <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }

    // Histogram -> prefix sum -> sort by the low log2(buckets) bits, exactly
    // as for the full hash grid.
    reserveBuildBuffers(s, n, buckets);
    groupPhotonsByKey(s, entries, buckets, stream);

    params.grid.dims = make_int3(0, 0, 0);
    params.grid.cell_size = 0.f; // per level: levelCellSize(level_s0, l)
    params.grid.hash_mask = (unsigned int)(buckets - 1);
    params.grid.cell_start = s.cell_start;
    params.grid.grid_photon_ids = s.ids_out;
    const long long positionBytes = attachPositions(params, s, entries, stream);
    params.accel = ACCEL_MLHASH;

    if (stats) {
        stats->buckets = buckets;
        stats->entries = entries;
        stats->photons = storedCount;
        stats->bytes = (buckets + 1) * (long long)sizeof(int)
            + (long long)entries * (long long)sizeof(int) + positionBytes;
    }
}

// Sizes the multi-level rectified table for `entries` insertions — one per
// photon and level in use, fewer with culling — clears it and points
// params.grid at it.  `bucketsOverride` > 0 (--rect-buckets) replaces the
// rule.  Returns the number of buckets.
static long long prepareMultiLevelRectTable(Params& params, double entries,
    long long bucketsOverride, GridScratch& s, cudaStream_t stream)
{
    const long long buckets = bucketsOverride > 0 ? bucketsOverride
        : tableBuckets(entries, params.table_load, kMaxHashBuckets, "mlrect");

    // count = 0; best = all ones, larger than any key, so the first insertion
    // into a bucket always takes it.
    ensureCapacity((void**)&s.bucket_count, s.bucket_count_cap, (size_t)buckets * sizeof(int));
    ensureCapacity((void**)&s.bucket_pick, s.bucket_pick_cap,
        (size_t)buckets * sizeof(unsigned long long));
    CUDA_PASS_CHECK(cudaMemsetAsync(s.bucket_count, 0, (size_t)buckets * sizeof(int), stream));
    CUDA_PASS_CHECK(cudaMemsetAsync(s.bucket_pick, 0xFF,
        (size_t)buckets * sizeof(unsigned long long), stream));

    params.grid.dims = make_int3(0, 0, 0);
    params.grid.cell_size = 0.f; // per level: levelCellSize(level_s0, l)
    params.grid.hash_mask = (unsigned int)(buckets - 1);
    params.grid.bucket_count = s.bucket_count;
    params.grid.bucket_pick = s.bucket_pick;
    params.accel = ACCEL_MLRECT;
    return buckets;
}

void prepareMultiLevelRectGridForTraceGPU(Params& params, long long bucketsOverride,
    GridScratch& s, cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();
    // Filled while tracing, before anyone knows how many photons will be
    // stored, so sized from the most there can be — as the stochastic grids.
    const long long buckets = prepareMultiLevelRectTable(params,
        (double)params.photon_capacity * std::max(params.num_levels, 1), bucketsOverride, s, stream);
    if (stats) {
        stats->buckets = buckets;
        stats->bytes = buckets * (long long)(sizeof(int) + sizeof(unsigned long long));
    }
}

void buildMultiLevelRectGridGPU(Params& params, int storedCount, long long bucketsOverride,
    GridScratch& s, cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    storedCount = std::min(storedCount, params.photon_capacity);
    if (storedCount <= 0 || params.num_levels <= 0) {
        params.accel = ACCEL_BRUTE_FORCE;
        return;
    }
    if ((unsigned int)storedCount > LEVEL_INDEX_MASK + 1u) {
        std::fprintf(stderr, "[cuda_passes] %d photons do not fit the %d-bit entry index\n",
            storedCount, LEVEL_SHIFT);
        std::abort();
    }

    LevelSet ls;
    ls.num = params.num_levels;
    for (int j = 0; j < MAX_LEVELS; ++j)
        ls.levels[j] = params.levels[j];
    ls.s0 = params.level_s0;
    ls.origin = params.level_origin;
    ls.cullMask = params.culling ? params.cull_mask : nullptr;
    ls.cullMaskBits = params.cull_mask_bits;

    // Without culling every (photon, level) pair is inserted; with it only
    // counting tells, and the count is read back to size the table — as the
    // multi-level hash grid reads back its compacted entries.
    long long entries = (long long)storedCount * params.num_levels;
    if (params.culling && bucketsOverride <= 0) {
        ensureCapacity(&s.radius_stats, s.radius_stats_cap, sizeof(unsigned long long));
        unsigned long long* total = static_cast<unsigned long long*>(s.radius_stats);
        CUDA_PASS_CHECK(cudaMemsetAsync(total, 0, sizeof(unsigned long long), stream));
        multiLevelCountKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
            params.photon_map, storedCount, ls, total);
        unsigned long long h = 0;
        CUDA_PASS_CHECK(cudaMemcpyAsync(&h, total, sizeof(h), cudaMemcpyDeviceToHost, stream));
        CUDA_PASS_CHECK(cudaStreamSynchronize(stream));
        entries = (long long)h;
    }

    const long long buckets = prepareMultiLevelRectTable(params, (double)entries,
        bucketsOverride, s, stream);
    const unsigned int mask = (unsigned int)(buckets - 1);

    multiLevelInsertKernel<<<blocksFor(storedCount), kBlock, 0, stream>>>(
        params.photon_map, storedCount, ls, mask, params.frame_index, params.seed,
        s.bucket_count, s.bucket_pick);
    CUDA_PASS_CHECK(cudaGetLastError());

    if (stats) {
        stats->buckets = buckets;
        // Counted above, unless --rect-buckets fixed the table without it;
        // then only countMultiLevelRectEntries can tell.
        stats->entries = (params.culling && bucketsOverride > 0) ? 0 : entries;
        stats->photons = storedCount;
        stats->bytes = buckets * (long long)(sizeof(int) + sizeof(unsigned long long));
    }
}

long long countMultiLevelRectEntries(const Params& params, GridScratch& s, cudaStream_t stream)
{
    const int buckets = (int)params.grid.hash_mask + 1;
    ensureCapacity(&s.radius_stats, s.radius_stats_cap, sizeof(long long));
    long long* total = static_cast<long long*>(s.radius_stats);
    size_t bytes = 0;
    CUDA_PASS_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, params.grid.bucket_count, total,
        buckets, stream));
    ensureCapacity(&s.temp, s.temp_cap, bytes);
    CUDA_PASS_CHECK(cub::DeviceReduce::Sum(s.temp, bytes, params.grid.bucket_count, total,
        buckets, stream));
    long long h = 0;
    CUDA_PASS_CHECK(cudaMemcpyAsync(&h, total, sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_PASS_CHECK(cudaStreamSynchronize(stream));
    return h;
}

bool runReservoirSelfTest(int n, int reps)
{
    // Chi-square with n-1 degrees of freedom; the Wilson-Hilferty cube-root
    // transform turns it into an (approximately) standard normal z, which
    // gives the p-value and the 1% critical value without tables.
    const double k = n - 1;
    const double mu = 1.0 - 2.0 / (9.0 * k), sigma = std::sqrt(2.0 / (9.0 * k));
    const double crit = k * std::pow(mu + 2.3263478740 * sigma, 3.0); // p = 0.01

    int* count = nullptr;
    unsigned long long* best = nullptr;
    CUDA_PASS_CHECK(cudaMalloc((void**)&count, (size_t)reps * sizeof(int)));
    CUDA_PASS_CHECK(cudaMalloc((void**)&best, (size_t)reps * sizeof(unsigned long long)));
    std::vector<unsigned long long> hostBest(reps);
    std::vector<int> hostCount(reps);

    bool allOk = true;
    std::printf("[Reservoir] %d insertions into one bucket, %d repetitions, a seed each;"
                " chi-square with %d degrees of freedom, 1%% critical value %.2f\n",
        n, reps, n - 1, crit);
    for (int layout = 0; layout < 2; ++layout) {
        const bool scattered = layout == 1;
        CUDA_PASS_CHECK(cudaMemset(count, 0, (size_t)reps * sizeof(int)));
        CUDA_PASS_CHECK(cudaMemset(best, 0xFF, (size_t)reps * sizeof(unsigned long long)));
        reservoirSelfTestKernel<<<blocksFor(n * reps), kBlock>>>(n, reps, scattered, count, best);
        CUDA_PASS_CHECK(cudaGetLastError());
        CUDA_PASS_CHECK(cudaMemcpy(hostBest.data(), best, (size_t)reps * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));
        CUDA_PASS_CHECK(cudaMemcpy(hostCount.data(), count, (size_t)reps * sizeof(int),
            cudaMemcpyDeviceToHost));

        std::vector<long long> wins(n, 0);
        int badCounts = 0;
        for (int r = 0; r < reps; ++r) {
            if (hostCount[r] != n)
                ++badCounts;
            const unsigned int winner = (unsigned int)hostBest[r] & LEVEL_INDEX_MASK;
            if (winner < (unsigned int)n)
                ++wins[winner];
            else
                ++badCounts;
        }
        const double expect = double(reps) / n;
        double chi2 = 0.0;
        for (int j = 0; j < n; ++j)
            chi2 += (wins[j] - expect) * (wins[j] - expect) / expect;
        const double z = (std::cbrt(chi2 / k) - mu) / sigma;
        const double pValue = 0.5 * std::erfc(z / std::sqrt(2.0));
        const bool ok = badCounts == 0 && chi2 < crit;
        allOk &= ok;

        std::printf("[Reservoir] %-10s chi-square = %.3f  p = %.3f  %s   wins:",
            scattered ? "scattered" : "adjacent", chi2, pValue, ok ? "PASS" : "FAIL");
        for (int j = 0; j < n; ++j)
            std::printf(" %lld", wins[j]);
        if (badCounts)
            std::printf("   (%d buckets with a wrong count or winner)", badCounts);
        std::printf("\n");
    }
    cudaFree(count);
    cudaFree(best);
    return allOk;
}

void prepareStochasticHashGridGPU(Params& params, AccelType type, float radius,
    GridScratch& s, cudaStream_t stream, GridBuildStats* stats)
{
    if (stats)
        *stats = GridBuildStats();

    // Cell size and origin exactly as in buildPhotonHashGridGPU, so all three
    // hash grids file a photon under the same cell.
    const float r = radius;
    const float3 origin = make_float3(params.scene_aabb_min.x - r,
        params.scene_aabb_min.y - r,
        params.scene_aabb_min.z - r);

    // The full hash grid takes table_load buckets per stored photon, but this
    // table is filled during the trace, before anyone knows how many photons
    // will be stored.  So it is sized from the most there can be — the photon
    // map's capacity — which keeps the load factor at or below the full grid's.
    const long long buckets = tableBuckets(params.photon_capacity, params.table_load,
        type == ACCEL_STOCHASTIC_HASH_GRID ? kMaxStochasticPhotonBuckets : kMaxHashBuckets,
        type == ACCEL_STOCHASTIC_HASH_GRID ? "stochastic" : "rect");
    const unsigned int mask = (unsigned int)(buckets - 1);

    ensureCapacity((void**)&s.bucket_count, s.bucket_count_cap,
        (size_t)buckets * sizeof(int));
    CUDA_PASS_CHECK(cudaMemsetAsync(s.bucket_count, 0,
        (size_t)buckets * sizeof(int), stream));

    if (type == ACCEL_STOCHASTIC_HASH_GRID) {
        // Not cleared: a bucket is only read when its count says a photon was
        // written there this pass.
        ensureCapacity((void**)&s.bucket_photon, s.bucket_photon_cap,
            (size_t)buckets * sizeof(Photon));
    } else {
        // Zero is the smallest 64-bit word, so the first photon filed into a
        // bucket always takes it.
        ensureCapacity((void**)&s.bucket_pick, s.bucket_pick_cap,
            (size_t)buckets * sizeof(unsigned long long));
        CUDA_PASS_CHECK(cudaMemsetAsync(s.bucket_pick, 0,
            (size_t)buckets * sizeof(unsigned long long), stream));
    }

    params.grid.aabb_min = origin;
    params.grid.aabb_max = make_float3(params.scene_aabb_max.x + r,
        params.scene_aabb_max.y + r,
        params.scene_aabb_max.z + r);
    params.grid.dims = make_int3(0, 0, 0);
    params.grid.cell_size = r;
    params.grid.hash_mask = mask;
    params.grid.bucket_count = s.bucket_count;
    params.grid.bucket_photon = s.bucket_photon;
    params.grid.bucket_pick = s.bucket_pick;
    params.accel = type;

    if (stats) {
        stats->buckets = buckets;
        stats->cell_size = r;
        stats->radius = r;
        stats->bytes = buckets * (long long)(sizeof(int)
            + (type == ACCEL_STOCHASTIC_HASH_GRID ? sizeof(Photon) : sizeof(unsigned long long)));
    }
}

// One gather point's share of GatherRadiusStats, and how two shares combine.
// Invalid gather points (r = 0) contribute nothing.
struct RadiusSample {
    float max_r;
    double sum_r;
    double sum_px;
    long long valid;
};

struct ToRadiusSample {
    float pixelAngle;
    __host__ __device__ RadiusSample operator()(const GatherPoint& gp) const
    {
        if (!gp.valid)
            return RadiusSample { 0.f, 0.0, 0.0, 0 };
        const float w = gp.t * pixelAngle;
        return RadiusSample { gp.r, (double)gp.r, w > 0.f ? (double)(gp.r / w) : 0.0, 1 };
    }
};

struct CombineRadiusSamples {
    __host__ __device__ RadiusSample operator()(const RadiusSample& a,
        const RadiusSample& b) const
    {
        return RadiusSample { fmaxf(a.max_r, b.max_r), a.sum_r + b.sum_r,
            a.sum_px + b.sum_px, a.valid + b.valid };
    }
};

void reduceGatherRadii(const Params& params, GridScratch& s, cudaStream_t stream,
    GatherRadiusStats& out)
{
    const int n = (int)(params.width * params.height);
    cub::TransformInputIterator<RadiusSample, ToRadiusSample, const GatherPoint*>
        in(params.gather_points, ToRadiusSample { params.pixel_angle });
    const RadiusSample zero { 0.f, 0.0, 0.0, 0 };

    ensureCapacity(&s.radius_stats, s.radius_stats_cap, sizeof(RadiusSample));
    RadiusSample* result = static_cast<RadiusSample*>(s.radius_stats);
    size_t bytes = 0;
    CUDA_PASS_CHECK(cub::DeviceReduce::Reduce(nullptr, bytes, in, result, n,
        CombineRadiusSamples(), zero, stream));
    ensureCapacity(&s.temp, s.temp_cap, bytes);
    CUDA_PASS_CHECK(cub::DeviceReduce::Reduce(s.temp, bytes, in, result, n,
        CombineRadiusSamples(), zero, stream));

    RadiusSample h;
    CUDA_PASS_CHECK(cudaMemcpyAsync(&h, result, sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_PASS_CHECK(cudaStreamSynchronize(stream));
    out.max_r = h.max_r;
    out.sum_r = h.sum_r;
    out.sum_px = h.sum_px;
    out.valid = h.valid;
}

struct IsLookup {
    __host__ __device__ long long operator()(int v) const { return v >= 0 ? 1 : 0; }
};
struct LookupValue {
    __host__ __device__ long long operator()(int v) const { return v > 0 ? v : 0; }
};

void summarizeQueryStats(const int* perPixel, int n, GridScratch& s, cudaStream_t stream,
    QueryStatSummary& out)
{
    out = QueryStatSummary();
    ensureCapacity((void**)&s.stat_sorted, s.stat_sorted_cap, (size_t)n * sizeof(int));
    if (!s.stat_scalars)
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.stat_scalars, 2 * sizeof(long long)));

    cub::TransformInputIterator<long long, IsLookup, const int*> isLookup(perPixel, IsLookup());
    cub::TransformInputIterator<long long, LookupValue, const int*> value(perPixel, LookupValue());
    size_t sortBytes = 0, countBytes = 0, sumBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sortBytes, perPixel, s.stat_sorted,
        n, 0, 32, stream));
    CUDA_PASS_CHECK(cub::DeviceReduce::Sum(nullptr, countBytes, isLookup, s.stat_scalars, n, stream));
    CUDA_PASS_CHECK(cub::DeviceReduce::Sum(nullptr, sumBytes, value, s.stat_scalars + 1, n, stream));
    ensureCapacity(&s.temp, s.temp_cap, std::max(sortBytes, std::max(countBytes, sumBytes)));

    // Signed keys: the -1s of the pixels without a lookup sort first.
    CUDA_PASS_CHECK(cub::DeviceRadixSort::SortKeys(s.temp, sortBytes, perPixel, s.stat_sorted,
        n, 0, 32, stream));
    CUDA_PASS_CHECK(cub::DeviceReduce::Sum(s.temp, countBytes, isLookup, s.stat_scalars, n, stream));
    CUDA_PASS_CHECK(cub::DeviceReduce::Sum(s.temp, sumBytes, value, s.stat_scalars + 1, n, stream));

    long long scalars[2] = { 0, 0 };
    CUDA_PASS_CHECK(cudaMemcpyAsync(scalars, s.stat_scalars, sizeof(scalars),
        cudaMemcpyDeviceToHost, stream));
    CUDA_PASS_CHECK(cudaStreamSynchronize(stream));
    out.lookups = scalars[0];
    if (out.lookups == 0)
        return;
    out.mean = double(scalars[1]) / double(out.lookups);

    const long long rank = (long long)std::ceil(0.95 * double(out.lookups)); // 1-based
    const long long p95At = (n - out.lookups) + rank - 1;
    CUDA_PASS_CHECK(cudaMemcpy(&out.p95, s.stat_sorted + p95At, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_PASS_CHECK(cudaMemcpy(&out.max, s.stat_sorted + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
}

void freeGridScratch(GridScratch& s)
{
    auto release = [](void* p) {
        if (p)
            cudaFree(p);
    };
    release(s.cell_start);
    release(s.cell_counts);
    release(s.keys_in);
    release(s.keys_out);
    release(s.ids_in);
    release(s.ids_out);
    release(s.temp);
    release(s.bucket_count);
    release(s.bucket_photon);
    release(s.bucket_pick);
    release(s.radius_stats);
    release(s.ml_keys);
    release(s.ml_values);
    release(s.ml_flags);
    release(s.ml_selected);
    release(s.stat_sorted);
    release(s.stat_scalars);
    release(s.vp_counts);
    release(s.vp_offsets);
    release(s.vp_keys_in);
    release(s.vp_keys_out);
    release(s.vp_values);
    release(s.vp_points);
    release(s.vp_scalars);
    release(s.positions);
    s = GridScratch();
}

void launchAccumulate(const Params& params, cudaStream_t stream)
{
    const int n = (int)(params.width * params.height);
    accumulateKernel<<<blocksFor(n), kBlock, 0, stream>>>(
        params.accum_buffer, params.frame_buffer, params.query_radiance,
        n, params.frame_index);
    CUDA_PASS_CHECK(cudaGetLastError());
}

// Every query kernel, by structure, with and without --stats; the hash grids
// also with their positions inline (--inline-positions).
template <bool STATS>
static const void* queryKernelFor(int accel, bool inl = false)
{
    switch (accel) {
    case ACCEL_UNIFORM_GRID:
        return (const void*)queryKernel<ACCEL_UNIFORM_GRID, STATS, false>;
    case ACCEL_HASH_GRID:
        return inl ? (const void*)queryKernel<ACCEL_HASH_GRID, STATS, true>
                   : (const void*)queryKernel<ACCEL_HASH_GRID, STATS, false>;
    case ACCEL_STOCHASTIC_HASH_GRID:
        return (const void*)queryKernel<ACCEL_STOCHASTIC_HASH_GRID, STATS, false>;
    case ACCEL_RECTIFIED_HASH_GRID:
        return (const void*)queryKernel<ACCEL_RECTIFIED_HASH_GRID, STATS, false>;
    case ACCEL_MLHASH:
        return inl ? (const void*)queryKernel<ACCEL_MLHASH, STATS, true>
                   : (const void*)queryKernel<ACCEL_MLHASH, STATS, false>;
    case ACCEL_MLRECT:
        return (const void*)queryKernel<ACCEL_MLRECT, STATS, false>;
    default:
        return (const void*)queryKernel<ACCEL_BRUTE_FORCE, STATS, false>;
    }
}

void launchQueryKernel(const Params& params, cudaStream_t stream)
{
    const int n = (int)(params.width * params.height);
    const bool inl = params.grid.positions != nullptr;
    const void* kernel = params.stats ? queryKernelFor<true>(params.accel, inl)
                                      : queryKernelFor<false>(params.accel, inl);
    void* args[] = { const_cast<Params*>(&params) };
    CUDA_PASS_CHECK(cudaLaunchKernel(kernel, dim3(blocksFor(n)), dim3(kBlock), args, 0, stream));
}

void launchMarkKernel(const Params& params, cudaStream_t stream)
{
    const int n = (int)(params.width * params.height);
    markLookupCellsKernel<<<blocksFor(n), kBlock, 0, stream>>>(params);
    CUDA_PASS_CHECK(cudaGetLastError());
}

void buildVisiblePointGridGPU(Params& params, GridScratch& s, cudaStream_t stream,
    VpGridStats* stats)
{
    if (stats)
        *stats = VpGridStats();
    const int n = (int)(params.width * params.height);
    if ((unsigned int)n > LEVEL_INDEX_MASK + 1u) {
        std::fprintf(stderr, "[cuda_passes] %d visible points do not fit the %d-bit entry index\n",
            n, LEVEL_SHIFT);
        std::abort();
    }
    if (!s.vp_counts || s.vp_points_cap < (size_t)n + 1) {
        if (s.vp_counts) {
            CUDA_PASS_CHECK(cudaFree(s.vp_counts));
            CUDA_PASS_CHECK(cudaFree(s.vp_offsets));
        }
        s.vp_points_cap = (size_t)n + 1;
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_counts, s.vp_points_cap * sizeof(int)));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_offsets, s.vp_points_cap * sizeof(int)));
    }
    if (!s.vp_scalars)
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_scalars, (2 + MAX_LEVELS) * sizeof(int)));
    int* levelFlags = s.vp_scalars + 2;

    // 1. cells per point, then where each point's entries start
    CUDA_PASS_CHECK(cudaMemsetAsync(s.vp_counts, 0, ((size_t)n + 1) * sizeof(int), stream));
    CUDA_PASS_CHECK(cudaMemsetAsync(levelFlags, 0, MAX_LEVELS * sizeof(int), stream));
    vpCountKernel<<<blocksFor(n), kBlock, 0, stream>>>(params, s.vp_counts, levelFlags);
    size_t scanBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, s.vp_counts, s.vp_offsets,
        n + 1, stream));
    ensureCapacity(&s.temp, s.temp_cap, scanBytes);
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(s.temp, scanBytes, s.vp_counts, s.vp_offsets,
        n + 1, stream));
    int entries = 0;
    int flags[MAX_LEVELS] = {};
    CUDA_PASS_CHECK(cudaMemcpyAsync(&entries, s.vp_offsets + n, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_PASS_CHECK(cudaMemcpyAsync(flags, levelFlags, sizeof(flags), cudaMemcpyDeviceToHost, stream));
    CUDA_PASS_CHECK(cudaStreamSynchronize(stream));

    VisiblePointGrid& g = params.vpgrid;
    g.num_levels = 0;
    for (int l = 0; l < MAX_LEVELS; ++l)
        if (flags[l])
            g.levels[g.num_levels++] = l;

    // 2. table_load buckets per entry
    const long long buckets = tableBuckets((double)entries, params.table_load, kMaxHashBuckets,
        params.vpgrid.cell_size > 0.f ? "splat" : "mlsplat");
    const unsigned int mask = (unsigned int)(buckets - 1);
    ensureCapacity((void**)&s.cell_start, s.cell_start_cap, ((size_t)buckets + 1) * sizeof(int));
    ensureCapacity((void**)&s.cell_counts, s.cell_counts_cap, ((size_t)buckets + 1) * sizeof(int));
    if (entries > 0 && (!s.vp_keys_in || s.vp_entries_cap < (size_t)entries)) {
        if (s.vp_keys_in) {
            CUDA_PASS_CHECK(cudaFree(s.vp_keys_in));
            CUDA_PASS_CHECK(cudaFree(s.vp_keys_out));
            CUDA_PASS_CHECK(cudaFree(s.vp_values));
            if (s.vp_points)
                CUDA_PASS_CHECK(cudaFree(s.vp_points));
            s.vp_points = nullptr;
        }
        s.vp_entries_cap = (size_t)entries + (size_t)entries / 4;
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_keys_in, s.vp_entries_cap * sizeof(unsigned long long)));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_keys_out, s.vp_entries_cap * sizeof(unsigned long long)));
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_values, s.vp_entries_cap * sizeof(unsigned int)));
    }
    if (entries > 0 && params.inline_positions && !s.vp_points)
        CUDA_PASS_CHECK(cudaMalloc((void**)&s.vp_points, s.vp_entries_cap * sizeof(float4)));
    float4* inlinePoints = params.inline_positions ? s.vp_points : nullptr;

    int unique = 0;
    CUDA_PASS_CHECK(cudaMemsetAsync(s.cell_counts, 0, ((size_t)buckets + 1) * sizeof(int), stream));
    if (entries > 0) {
        // 3. one key per (point, cell)
        vpEmitKernel<<<blocksFor(n), kBlock, 0, stream>>>(params, s.vp_offsets, mask, s.vp_keys_in);

        // 4. stable sort by the bucket bits alone, then drop repeated keys
        int bucketBits = 0;
        while ((1LL << bucketBits) < buckets)
            ++bucketBits;
        size_t sortBytes = 0, uniqueBytes = 0;
        CUDA_PASS_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sortBytes, s.vp_keys_in,
            s.vp_keys_out, entries, 32, 32 + bucketBits, stream));
        CUDA_PASS_CHECK(cub::DeviceSelect::Unique(nullptr, uniqueBytes, s.vp_keys_out, s.vp_keys_in,
            s.vp_scalars + 1, entries, stream));
        ensureCapacity(&s.temp, s.temp_cap, std::max(sortBytes, uniqueBytes));
        CUDA_PASS_CHECK(cub::DeviceRadixSort::SortKeys(s.temp, sortBytes, s.vp_keys_in,
            s.vp_keys_out, entries, 32, 32 + bucketBits, stream));
        CUDA_PASS_CHECK(cub::DeviceSelect::Unique(s.temp, uniqueBytes, s.vp_keys_out, s.vp_keys_in,
            s.vp_scalars + 1, entries, stream));

        // 5. bucket histogram -> offsets; entries -> values
        vpBucketKernel<<<blocksFor(entries), kBlock, 0, stream>>>(s.vp_keys_in, s.vp_scalars + 1,
            s.cell_counts, s.vp_values, params.gather_points, inlinePoints);
    }
    size_t offBytes = 0;
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, offBytes, s.cell_counts, s.cell_start,
        (int)(buckets + 1), stream));
    ensureCapacity(&s.temp, s.temp_cap, offBytes);
    CUDA_PASS_CHECK(cub::DeviceScan::ExclusiveSum(s.temp, offBytes, s.cell_counts, s.cell_start,
        (int)(buckets + 1), stream));
    if (entries > 0) {
        CUDA_PASS_CHECK(cudaMemcpyAsync(&unique, s.vp_scalars + 1, sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        CUDA_PASS_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_PASS_CHECK(cudaGetLastError());

    g.cell_start = s.cell_start;
    g.values = s.vp_values;
    g.points = entries > 0 ? inlinePoints : nullptr;
    g.hash_mask = mask;

    if (stats) {
        stats->entries = entries;
        stats->unique = unique;
        stats->buckets = buckets;
        stats->bytes = (buckets + 1) * (long long)sizeof(int)
            + (long long)unique * (long long)(sizeof(int) + (inlinePoints ? sizeof(float4) : 0));
        stats->num_levels = g.num_levels;
        stats->level_min = g.num_levels ? g.levels[0] : -1;
        stats->level_max = g.num_levels ? g.levels[g.num_levels - 1] : -1;
    }
}

void launchSplatFinalize(const Params& params, cudaStream_t stream)
{
    const int n = (int)(params.width * params.height);
    splatFinalizeKernel<<<blocksFor(n), kBlock, 0, stream>>>(params);
    CUDA_PASS_CHECK(cudaGetLastError());
}

std::vector<KernelResources> kernelResources()
{
    std::vector<KernelResources> out;
    auto add = [&](const char* name, const void* f) {
        cudaFuncAttributes a;
        CUDA_PASS_CHECK(cudaFuncGetAttributes(&a, f));
        out.push_back({ name, a.numRegs, a.localSizeBytes, a.maxThreadsPerBlock });
    };
    const int accels[] = { ACCEL_BRUTE_FORCE, ACCEL_UNIFORM_GRID, ACCEL_HASH_GRID,
        ACCEL_STOCHASTIC_HASH_GRID, ACCEL_RECTIFIED_HASH_GRID, ACCEL_MLHASH, ACCEL_MLRECT };
    const char* names[] = { "query_brute", "query_uniform", "query_hash", "query_stochastic",
        "query_rect", "query_mlhash", "query_mlrect" };
    const char* statNames[] = { "query_brute_stats", "query_uniform_stats", "query_hash_stats",
        "query_stochastic_stats", "query_rect_stats", "query_mlhash_stats", "query_mlrect_stats" };
    for (int i = 0; i < 7; ++i)
        add(names[i], queryKernelFor<false>(accels[i]));
    for (int i = 0; i < 7; ++i)
        add(statNames[i], queryKernelFor<true>(accels[i]));
    add("query_hash_inline", queryKernelFor<false>(ACCEL_HASH_GRID, true));
    add("query_mlhash_inline", queryKernelFor<false>(ACCEL_MLHASH, true));
    add("mark", (const void*)markLookupCellsKernel);
    add("accumulate", (const void*)accumulateKernel);
    add("ml_entries", (const void*)multiLevelEntriesKernel);
    add("ml_insert", (const void*)multiLevelInsertKernel);
    add("ml_count", (const void*)multiLevelCountKernel);
    add("vp_count", (const void*)vpCountKernel);
    add("vp_emit", (const void*)vpEmitKernel);
    add("vp_bucket", (const void*)vpBucketKernel);
    add("splat_finalize", (const void*)splatFinalizeKernel);
    return out;
}
