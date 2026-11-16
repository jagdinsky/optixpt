#pragma once
#include "params.h"

// The photon lookup: the tests a photon has to pass to count for a gather
// point, and the search each acceleration structure runs to find it.
//
// Shared by two compilations.  The query passes are plain CUDA kernels
// (cuda_passes.cu), one per structure, so that each gets its own register
// allocation instead of the one OptiX gives every program of a pipeline; the
// OptiX programs (device.cu) use the same tests, so that every path through
// the renderer agrees on which photon belongs to which gather point.  The
// launch parameters are therefore always passed in, never named.

// helpers
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float t, float3 v) { return make_float3(t * v.x, t * v.y, t * v.z); }
__device__ __forceinline__ float3 operator*(float3 v, float t) { return make_float3(t * v.x, t * v.y, t * v.z); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3& operator+=(float3& a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __forceinline__ float3 normalize(float3 v)
{
    float len2 = dot(v, v);
    if (len2 < 1e-20f) return make_float3(0.f, 0.f, 0.f);
    float inv = rsqrtf(len2);
    return make_float3(v.x * inv, v.y * inv, v.z * inv);
}
__device__ __forceinline__ float length(float3 v) { return sqrtf(dot(v, v)); }

// The pair test and the cone weight, in explicitly rounded arithmetic.
//
// A plain a*b + c that NVVM does not fuse itself reaches the code generator as
// a separate mul and add, and the code generator may still fuse them into one
// FMA — OptiX's and ptxas, which compile the OptiX programs and the CUDA
// kernels, do not always agree.  Then the same photon gets a distance or a
// weight one bit apart in two passes, and a photon on the edge of a test
// counts in one and not in the other.  fmaf and the _rn intrinsics are never
// fused or reordered, so every compilation computes the same values: the
// gathers and the splatting photon pass find exactly the same pairs.  The
// dot product keeps the association NVVM gave it before, y first.
__device__ __forceinline__ float3 subRn(float3 a, float3 b)
{
    return make_float3(__fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y), __fsub_rn(a.z, b.z));
}
__device__ __forceinline__ float dotRn(float3 a, float3 b)
{
    return fmaf(a.z, b.z, fmaf(a.x, b.x, __fmul_rn(a.y, b.y)));
}
// 1 - d/r, the cone filter (k = 1), for a photon at squared distance d2.
__device__ __forceinline__ float coneWeight(float d2, float invR)
{
    return __fsub_rn(1.f, __fmul_rn(sqrtf(d2), invR));
}

// flux += w * power, the product rounded on its own — which is all the
// splatting pass can do, one atomic add of a finished product per channel —
// so a pair adds the same three numbers in both, and the exact gathers and
// splatting differ only in the order of their sums.
__device__ __forceinline__ void addWeighted(float3& flux, float w, float3 power)
{
    flux.x = __fadd_rn(flux.x, __fmul_rn(w, power.x));
    flux.y = __fadd_rn(flux.y, __fmul_rn(w, power.y));
    flux.z = __fadd_rn(flux.z, __fmul_rn(w, power.z));
}

// Decides whether a stored photon may contribute to the estimate at `x`.
// The plain 3D distance test is not enough: a sphere of radius r around a point
// near a corner also encloses photons that landed on the *neighbouring* wall,
// and those are what leak across edges and show up as blotches.  Requiring the
// photon to sit near the tangent plane turns the sphere into a thin disc, and
// requiring a matching normal keeps the two sides of a thin wall apart.
__device__ __forceinline__ bool photonUsable(const Photon& ph, float3 x, float3 n,
    float r2, float planeTol, float normalTol, int maxPhotonDepth, float& d2)
{
    // Path length first, because it is the cheapest test and because getting it
    // wrong is not a matter of a few stray photons: without it the camera walk
    // and the photon walk each get their own max_depth, so photon mapping
    // integrates paths up to twice as long as the path tracer does and the two
    // modes stop solving the same problem.  See __raygen__camera for the budget.
    if (ph.depth > maxPhotonDepth)
        return false;
    const float3 diff = subRn(ph.pos, x);
    d2 = dotRn(diff, diff);
    if (d2 > r2)
        return false;
    if (dotRn(n, ph.normal) < normalTol)
        return false;
    if (fabsf(dotRn(diff, n)) > planeTol)
        return false;
    // The photon must have arrived on the side we are shading.
    if (dotRn(n, ph.dir) <= 1e-4f)
        return false;
    return true;
}

// What one lookup looked at, for --stats: photon records it read and tested,
// how many of them passed photonUsable, and how many cells it visited.
struct GatherTally {
    int tested;
    int accepted;
    int cells;
};

// Tests the photons grid_photon_ids[start .. end) and accumulates the ones that
// pass.  [start, end) is one cell of the uniform grid or one bucket of the hash
// grid.  Both structures call this same loop, so the only thing that differs
// between them is which ranges get visited — which is the thing being compared.
//
// INLINE (--inline-positions): the distance is tested first, on the position
// stored next to the entry, and the photon itself is loaded only when it is
// within r.  The test is photonUsable's own arithmetic on the same position,
// so it turns away exactly the photons photonUsable would.
template <bool INLINE>
__device__ __forceinline__ void gatherRange(const Params& P, int start, int end,
    float3 hitpos, float3 n, float r2, float invR, float planeTol, float normalTol,
    int maxPhotonDepth, float3& flux, int& count, GatherTally& tally)
{
    const int* ids = P.grid.grid_photon_ids;
    tally.tested += end - start;
    for (int k = start; k < end; ++k) {
        if (INLINE) {
            const float4 q = P.grid.positions[k];
            const float3 diff = subRn(make_float3(q.x, q.y, q.z), hitpos);
            if (dotRn(diff, diff) > r2)
                continue;
        }
        const Photon& ph = P.photon_map[ids[k]];

        float d2;
        if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                maxPhotonDepth, d2))
            continue;

        const float weight = coneWeight(d2, invR);
        addWeighted(flux, weight, ph.power);
        ++count;
        ++tally.accepted;
    }
}

// Cone-filtered flux within `r`, and how many photons contributed.  The count
// is what lets the caller adapt the radius to the local photon density.
__device__ inline void gatherPhotonsGrid(const Params& P, float3 hitpos, float3 n, float r,
    float r2, int maxPhotonDepth, float3& flux, int& count, GatherTally& tally)
{
    const PhotonGrid& g = P.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    // The cells the search sphere overlaps, inside the dense array (photons
    // never land outside the box it covers).
    int3 lo, hi;
    cellRange3(hitpos, r, g.aabb_min, g.cell_size, lo, hi);
    lo = make_int3(max(lo.x, 0), max(lo.y, 0), max(lo.z, 0));
    hi = make_int3(min(hi.x, g.dims.x - 1), min(hi.y, g.dims.y - 1), min(hi.z, g.dims.z - 1));

    for (int nz = lo.z; nz <= hi.z; ++nz) {
        for (int ny = lo.y; ny <= hi.y; ++ny) {
            for (int nx = lo.x; nx <= hi.x; ++nx) {
                const int cell = (nz * g.dims.y + ny) * g.dims.x + nx;
                ++tally.cells;
                gatherRange<false>(P, g.cell_start[cell], g.cell_start[cell + 1],
                    hitpos, n, r2, invR, planeTol, normalTol, maxPhotonDepth,
                    flux, count, tally);
            }
        }
    }
}

// Bound on the cells a single-level hash grid lookup visits.  Its cells are at
// least as wide as the largest radius of the frame, so the sphere overlaps at
// most 3 per axis, and rounding at the two ends of the range adds one at most.
#define MAX_HASH_VISITS 64

// The hash grids visit the same cells as the uniform grid — the ones the
// search sphere overlaps — but find each cell's photons through the hash
// table instead of a dense array.  Two consequences of sharing buckets
// between cells:
//
// 1. A bucket can also hold photons from cells outside the visited range
//    that hashed to the same bucket (a collision).  They need no special
//    handling.  The range contains the whole search sphere, so every photon
//    outside it is further than r from the query and the distance test in
//    photonUsable rejects it.  A collision costs time, never correctness.
//
// 2. Two of the visited cells can hash to the same bucket.  Visiting that
//    bucket a second time would count its photons twice, which *is* a
//    correctness error.  So the buckets already visited for this query are
//    remembered, and a repeated one is skipped.  The list is at most
//    MAX_HASH_VISITS entries and lives in per-thread local memory, which is
//    cached in L1.
//
// Unlike the uniform grid there is no bounds check: every integer cell has a
// bucket, including cells outside the scene box, which simply hold no photons
// of their own.
//
// The full and the original stochastic hash grid walk their buckets through
// this one function, so they visit exactly the same buckets and differ only
// in what they find in each.
template <typename Visit>
__device__ __forceinline__ void forEachHashBucket(const Params& P, float3 hitpos, float r,
    GatherTally& tally, Visit&& visit)
{
    const PhotonGrid& g = P.grid;

    int3 lo, hi;
    cellRange3(hitpos, r, g.aabb_min, g.cell_size, lo, hi);

    unsigned int visited[MAX_HASH_VISITS];
    int numVisited = 0;

    for (int nz = lo.z; nz <= hi.z; ++nz) {
        for (int ny = lo.y; ny <= hi.y; ++ny) {
            for (int nx = lo.x; nx <= hi.x; ++nx) {
                ++tally.cells;
                const unsigned int bucket = hashCell(nx, ny, nz) & g.hash_mask;

                bool alreadyVisited = false;
                for (int j = 0; j < numVisited; ++j) {
                    if (visited[j] == bucket) {
                        alreadyVisited = true;
                        break;
                    }
                }
                if (alreadyVisited)
                    continue;
                if (numVisited < MAX_HASH_VISITS)
                    visited[numVisited++] = bucket;

                visit(bucket);
            }
        }
    }
}

// ANYSIZE: the cells may be narrower than r (a grid built before the frame's
// radii were known, --fused), so the range can hold more cells than the
// bucket list of forEachHashBucket.  Then every visited cell's bucket is read
// and only the photons of that cell are taken, as the multi-level grid does:
// exact for any number of cells, no bucket remembered.
template <bool INLINE, bool ANYSIZE = false>
__device__ inline void gatherPhotonsHash(const Params& P, float3 hitpos, float3 n, float r,
    float r2, int maxPhotonDepth, float3& flux, int& count, GatherTally& tally)
{
    const PhotonGrid& g = P.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    if (ANYSIZE) {
        int3 lo, hi;
        cellRange3(hitpos, r, g.aabb_min, g.cell_size, lo, hi);
        for (int nz = lo.z; nz <= hi.z; ++nz) {
            for (int ny = lo.y; ny <= hi.y; ++ny) {
                for (int nx = lo.x; nx <= hi.x; ++nx) {
                    ++tally.cells;
                    const unsigned int bucket = hashCell(nx, ny, nz) & g.hash_mask;
                    const int end = g.cell_start[bucket + 1];
                    tally.tested += end - g.cell_start[bucket];
                    for (int k = g.cell_start[bucket]; k < end; ++k) {
                        const Photon& ph = P.photon_map[g.grid_photon_ids[k]];
                        if (photonCellCoord(ph.pos.x, g.aabb_min.x, g.cell_size) != nx
                            || photonCellCoord(ph.pos.y, g.aabb_min.y, g.cell_size) != ny
                            || photonCellCoord(ph.pos.z, g.aabb_min.z, g.cell_size) != nz)
                            continue; // another cell sharing the bucket
                        float d2;
                        if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                                maxPhotonDepth, d2))
                            continue;
                        addWeighted(flux, coneWeight(d2, invR), ph.power);
                        ++count;
                        ++tally.accepted;
                    }
                }
            }
        }
        return;
    }

    forEachHashBucket(P, hitpos, r, tally, [&](unsigned int bucket) {
        gatherRange<INLINE>(P, g.cell_start[bucket], g.cell_start[bucket + 1],
            hitpos, n, r2, invR, planeTol, normalTol, maxPhotonDepth,
            flux, count, tally);
    });
}

// The stochastic hash grids keep one photon per bucket, plus c, the number of
// photons that hashed there.  The one photon stands in for all c, so its
// contribution is multiplied by c.  If it was picked uniformly, each of the c
// is the one kept with probability 1/c, and the expected contribution is
//
//     sum_i (1/c) * c * f(photon_i) = sum_i f(photon_i),
//
// exactly what the full hash grid adds up for the same bucket.  A photon that
// fails photonUsable — too far, wrong side, too deep, or a collision from
// another cell — is just an f = 0 term of that sum: it costs variance, never
// bias.  What *does* bias the estimate is a pick that is not uniform, and that
// is the whole difference between the two stochastic grids; see
// insertStochastic in device.cu.
//
// `count` gets c for every photon used, i.e. the number of photons the
// estimate stands for, so the adaptive radius sees the same kind of number the
// full grid gives it.
//
// This is the original stochastic grid, whose bucket holds a copy of the
// photon; the rectified one is gatherPhotonsRect below.
__device__ inline void gatherPhotonsStochastic(const Params& P, float3 hitpos, float3 n,
    float r, float r2, int maxPhotonDepth, float3& flux, int& count, GatherTally& tally)
{
    const PhotonGrid& g = P.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    forEachHashBucket(P, hitpos, r, tally, [&](unsigned int bucket) {
        const int c = g.bucket_count[bucket];
        if (c == 0)
            return;
        ++tally.tested;

        const Photon ph = g.bucket_photon[bucket];

        float d2;
        if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                maxPhotonDepth, d2))
            return;

        const float weight = coneWeight(d2, invR);
        flux += (weight * (float)c) * ph.power;
        count += c;
        ++tally.accepted;
    });
}

// The rectified stochastic hash grid: the representative of a bucket is
// counted for the visited cell only if it lies in that cell.  Two visited
// cells that share a bucket then each look at it, and the representative —
// which lies in at most one of them — counts at most once, for its own cell,
// so the buckets need no de-duplication; a representative from a cell outside
// the range is further than r away and counts for nothing either way.  The
// value is the same as with de-duplicated buckets and the distance test alone,
// sum over the visited cells of c * f(the representative, if it is in the
// cell), whose expectation is the sum over the cells' own photons (see above).
// Its cell is recomputed exactly as insertStochastic filed it.
__device__ inline void gatherPhotonsRect(const Params& P, float3 hitpos, float3 n, float r,
    float r2, int maxPhotonDepth, float3& flux, int& count, GatherTally& tally)
{
    const PhotonGrid& g = P.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    int3 lo, hi;
    cellRange3(hitpos, r, g.aabb_min, g.cell_size, lo, hi);

    for (int nz = lo.z; nz <= hi.z; ++nz) {
        for (int ny = lo.y; ny <= hi.y; ++ny) {
            for (int nx = lo.x; nx <= hi.x; ++nx) {
                ++tally.cells;
                const unsigned int bucket = hashCell(nx, ny, nz) & g.hash_mask;
                const int c = g.bucket_count[bucket];
                if (c == 0)
                    continue;
                ++tally.tested;

                const Photon& ph = P.photon_map[(unsigned int)(g.bucket_pick[bucket] & 0xFFFFFFFFull)];
                if (photonCellCoord(ph.pos.x, g.aabb_min.x, g.cell_size) != nx
                    || photonCellCoord(ph.pos.y, g.aabb_min.y, g.cell_size) != ny
                    || photonCellCoord(ph.pos.z, g.aabb_min.z, g.cell_size) != nz)
                    continue; // the representative of another cell sharing the bucket

                float d2;
                if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                        maxPhotonDepth, d2))
                    continue;

                const float weight = coneWeight(d2, invR);
                flux += (weight * (float)c) * ph.power;
                count += c;
                ++tally.accepted;
            }
        }
    }
}

// The multi-level hash grid.  A lookup searches its own level — the smallest
// whose cells are at least r wide — so it walks 8 to 27 cells between r and 2r
// across, whatever r is, rather than cells as wide as the frame's largest
// radius.
//
// Every photon is filed once per level in use, and cells of any level can
// share a bucket, so a bucket also holds entries that do not belong to the
// cell being visited.  Two checks skip those, and they are what makes the
// result exact without de-duplicating buckets the way the single-level hash
// grids do:
//   - an entry of another level: the photon is filed under this level too,
//     in the cell it actually lies in, and counted there;
//   - a photon of another cell: if that cell is one of the visited ones, the
//     photon is counted when that cell is visited; if not, it is further than
//     r away.
// So each photon within r is counted exactly once, from its own cell on the
// lookup's level.  The cell is recomputed exactly as the build computed it
// (photonCellCoord with the level's cell size), so the two always agree.
template <bool INLINE>
__device__ inline void gatherPhotonsMultiLevel(const Params& P, float3 hitpos, float3 n,
    float r, float r2, int level, int maxPhotonDepth, float3& flux, int& count,
    GatherTally& tally)
{
    const PhotonGrid& g = P.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    const float cs = levelCellSize(P.level_s0, level);
    const float3 o = P.level_origin;
    int3 lo, hi;
    cellRange3(hitpos, r, o, cs, lo, hi);

    for (int nz = lo.z; nz <= hi.z; ++nz) {
        for (int ny = lo.y; ny <= hi.y; ++ny) {
            for (int nx = lo.x; nx <= hi.x; ++nx) {
                ++tally.cells;
                const unsigned int bucket = hashCell4(nx, ny, nz, level) & g.hash_mask;
                const int end = g.cell_start[bucket + 1];
                tally.tested += end - g.cell_start[bucket];
                for (int k = g.cell_start[bucket]; k < end; ++k) {
                    const unsigned int v = (unsigned int)g.grid_photon_ids[k];
                    if ((int)(v >> LEVEL_SHIFT) != level)
                        continue; // filed under another level
                    // The cell and, inlined, the distance from the position
                    // next to the entry; the photon only once it passes both.
                    float3 pos;
                    if (INLINE) {
                        const float4 q = g.positions[k];
                        pos = make_float3(q.x, q.y, q.z);
                    } else {
                        pos = P.photon_map[v & LEVEL_INDEX_MASK].pos;
                    }
                    if (photonCellCoord(pos.x, o.x, cs) != nx
                        || photonCellCoord(pos.y, o.y, cs) != ny
                        || photonCellCoord(pos.z, o.z, cs) != nz)
                        continue; // another cell sharing the bucket
                    if (INLINE) {
                        const float3 diff = subRn(pos, hitpos);
                        if (dotRn(diff, diff) > r2)
                            continue;
                    }
                    const Photon& ph = P.photon_map[v & LEVEL_INDEX_MASK];

                    float d2;
                    if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                            maxPhotonDepth, d2))
                        continue;

                    const float weight = coneWeight(d2, invR);
                    addWeighted(flux, weight, ph.power);
                    ++count;
                    ++tally.accepted;
                }
            }
        }
    }
}

// The multi-level rectified grid: the same cells on the lookup's level as
// gatherPhotonsMultiLevel, but each bucket holds one representative standing
// in for all n insertions counted there.  The representative is uniform over
// those n (insertRepresentative in params.h), so weighting it by n gives
// each bucket an expected contribution equal to the sum over everything
// inserted into it — and the same two checks as the full multi-level grid
// zero out insertions of another level or another cell, which leaves, in
// expectation, exactly the sum over the cell's own photons.  No bucket
// de-duplication is needed: a bucket visited for two cells counts its
// representative for at most the one cell it lies in.
//
// `count` gets n for every representative used, the number of photons the
// estimate stands for, as in gatherPhotonsStochastic.
__device__ inline void gatherPhotonsMultiLevelRect(const Params& P, float3 hitpos, float3 n,
    float r, float r2, int level, int maxPhotonDepth, float3& flux, int& count,
    GatherTally& tally)
{
    const PhotonGrid& g = P.grid;
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    const float cs = levelCellSize(P.level_s0, level);
    const float3 o = P.level_origin;
    int3 lo, hi;
    cellRange3(hitpos, r, o, cs, lo, hi);

    for (int nz = lo.z; nz <= hi.z; ++nz) {
        for (int ny = lo.y; ny <= hi.y; ++ny) {
            for (int nx = lo.x; nx <= hi.x; ++nx) {
                ++tally.cells;
                const unsigned int bucket = hashCell4(nx, ny, nz, level) & g.hash_mask;
                const int c = g.bucket_count[bucket];
                if (c == 0)
                    continue;
                ++tally.tested;
                const unsigned int v = (unsigned int)(g.bucket_pick[bucket] & 0xFFFFFFFFull);
                if ((int)(v >> LEVEL_SHIFT) != level)
                    continue; // the representative belongs to another level
                const Photon& ph = P.photon_map[v & LEVEL_INDEX_MASK];
                if (photonCellCoord(ph.pos.x, o.x, cs) != nx
                    || photonCellCoord(ph.pos.y, o.y, cs) != ny
                    || photonCellCoord(ph.pos.z, o.z, cs) != nz)
                    continue; // ... or to another cell sharing the bucket

                float d2;
                if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                        maxPhotonDepth, d2))
                    continue;

                const float weight = coneWeight(d2, invR);
                flux += (weight * (float)c) * ph.power;
                count += c;
                ++tally.accepted;
            }
        }
    }
}

__device__ inline void gatherBruteForce(const Params& P, float3 hitpos, float3 n, float r,
    float r2, int maxPhotonDepth, float3& flux, int& count, GatherTally& tally)
{
    flux = make_float3(0.f, 0.f, 0.f);
    count = 0;

    const float planeTol = P.photon_plane_tol * r;
    const float normalTol = P.photon_normal_tol;
    const float invR = 1.f / r;

    int stored = min(*P.photon_count, P.photon_capacity);
    tally.tested += stored;
    for (int i = 0; i < stored; i++) {
        const Photon& ph = P.photon_map[i];

        float d2;
        if (!photonUsable(ph, hitpos, n, r2, planeTol, normalTol,
                maxPhotonDepth, d2))
            continue;

        const float weight = coneWeight(d2, invR);
        addWeighted(flux, weight, ph.power);
        ++count;
        ++tally.accepted;
    }
}

// The structure is a template parameter, not a run-time switch: each query
// kernel contains its own structure's search and nothing else.
template <int ACCEL, bool INLINE, bool ANYSIZE = false>
__device__ __forceinline__ void gatherPhotons(const Params& P, float3 hitpos, float3 n,
    float r, float r2, int level, int maxPhotonDepth, float3& flux, int& count,
    GatherTally& tally)
{
    if constexpr (ACCEL == ACCEL_UNIFORM_GRID)
        gatherPhotonsGrid(P, hitpos, n, r, r2, maxPhotonDepth, flux, count, tally);
    else if constexpr (ACCEL == ACCEL_HASH_GRID)
        gatherPhotonsHash<INLINE, ANYSIZE>(P, hitpos, n, r, r2, maxPhotonDepth, flux, count, tally);
    else if constexpr (ACCEL == ACCEL_STOCHASTIC_HASH_GRID)
        gatherPhotonsStochastic(P, hitpos, n, r, r2, maxPhotonDepth, flux, count, tally);
    else if constexpr (ACCEL == ACCEL_RECTIFIED_HASH_GRID)
        gatherPhotonsRect(P, hitpos, n, r, r2, maxPhotonDepth, flux, count, tally);
    else if constexpr (ACCEL == ACCEL_MLHASH)
        gatherPhotonsMultiLevel<INLINE>(P, hitpos, n, r, r2, level, maxPhotonDepth, flux, count, tally);
    else if constexpr (ACCEL == ACCEL_MLRECT)
        gatherPhotonsMultiLevelRect(P, hitpos, n, r, r2, level, maxPhotonDepth, flux, count, tally);
    else
        gatherBruteForce(P, hitpos, n, r, r2, maxPhotonDepth, flux, count, tally);
}

// Irradiance from the photon map at a surface point.
//
// The cone filter w(d) = 1 - d/r integrates to  ∫(1 - d/r) dA = π r² / 3  over
// the disc, so the density estimate is  E = 3 Σ w Φ / (π r²).
//
// With a single fixed radius the estimate is either too noisy (sparse regions)
// or too blurry (dense ones) — the "big stains".  Probing the density at the
// maximum radius first and then shrinking towards `target_photons` (photon
// count grows with r², hence the sqrt) keeps well-lit regions sharp without
// starving the dark ones.
//
// `r` is the lookup's own radius (GatherPoint::r), and it is the one used
// everywhere below — the distance and plane tests, the cone weight and the
// normalisation — so a per-pixel radius changes nothing but r itself.
template <int ACCEL, bool INLINE, bool ANYSIZE = false>
__device__ __forceinline__ float3 photonIrradiance(const Params& P, float3 hitpos, float3 n,
    int maxPhotonDepth, float r, int level, float& usedRadius, GatherTally& tally)
{
    float r2 = r * r;
    float3 flux;
    int count;

    gatherPhotons<ACCEL, INLINE, ANYSIZE>(P, hitpos, n, r, r2, level, maxPhotonDepth, flux, count, tally);

    if (P.adaptive_radius && P.target_photons > 0 && count > P.target_photons) {
        float scale = sqrtf((float)P.target_photons / (float)count);
        r *= fmaxf(scale, 0.05f);
        r2 = r * r;
        gatherPhotons<ACCEL, INLINE, ANYSIZE>(P, hitpos, n, r, r2, level, maxPhotonDepth, flux, count, tally);
    }

    usedRadius = r;
    if (count == 0)
        return make_float3(0.f, 0.f, 0.f);
    return flux * (3.f / (M_PIf * r2));
}

// The query pass for one pixel: the photon lookup, and nothing else.
//
// This is the pass the acceleration structures are compared on, so it is kept
// deliberately bare — read the gather point, ask the photon map for irradiance,
// write the pixel's radiance.  Accumulation and tone mapping happen afterwards
// in a separate kernel so that they stay out of the measurement, and so that
// this pass can be run twice over one photon map (grid against brute force)
// without the two runs accumulating on top of each other.  With STATS false
// the tally is dead code and compiles away.
template <int ACCEL, bool STATS, bool INLINE = false>
__device__ __forceinline__ void queryPixel(const Params& P, int pixel)
{
    const GatherPoint gp = P.gather_points[pixel];
    float3 radiance = gp.direct;
    GatherTally tally = { 0, 0, 0 };

    if (gp.valid) {
        float usedRadius;
        float3 irradiance = photonIrradiance<ACCEL, INLINE>(P, gp.pos, gp.normal,
            gp.max_photon_depth, gp.r, gp.level, usedRadius, tally);
        radiance += gp.weight * irradiance;
    }

    P.query_radiance[pixel] = radiance;

    // --stats: one slot per pixel, so no atomics; -1 = no lookup here.
    if (STATS) {
        P.stat_tested[pixel] = gp.valid ? tally.tested : -1;
        P.stat_accepted[pixel] = gp.valid ? tally.accepted : -1;
        P.stat_cells[pixel] = gp.valid ? tally.cells : -1;
    }
}
