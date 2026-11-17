#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <optix.h>
#include <vector_types.h>

// Helpers shared by the host and the device.  main.cpp is compiled by the host
// compiler, which does not understand the CUDA function attributes, so they are
// only emitted when nvcc is the one reading this header.
#ifdef __CUDACC__
#define RT_INLINE __host__ __device__ __forceinline__
#else
#define RT_INLINE inline
#endif

#define RT_PI 3.14159265358979323846f

// Triangles for scene geometry
struct Triangle {
    float3 v0, v1, v2; // vertices
    float3 n0, n1, n2; // per-vertex normals
    float2 uv0, uv1, uv2; // per-vertex UVs
    int mat_id;
};

// Material description (one per material ID)
enum MatType : int {
    MAT_DIFFUSE = 0,
    MAT_MIRROR = 1,
    MAT_GLASS = 2
};

// Material data structure
struct Material {
    int matType = MAT_DIFFUSE; // 0 = diffuse, 1 = mirror, 2 = glass
    float3 albedo;
    float3 emission;
    float ior = 1.5f; // for glass: index of refraction
    cudaTextureObject_t base_color_tex; // 0 = no texture
    cudaTextureObject_t emissive_tex; // 0 = no texture
};

// Emissive light descriptor (one per emissive triangle)
// Built on the host, uploaded to params.lights / params.num_lights.
// area = |cross(e1,e2)| / 2  (precomputed for sampling efficiency)
struct EmissiveTriangle {
    float3 v0, v1, v2; // world-space vertices (duplicated for fast access)
    float3 emission; // emitted radiance (already resolved from material)
    float3 normal; // precomputed face normal (for sampling)
    float area; // triangle area
    int tri_idx; // index into params.triangles (for normal lookup)
};

// Photon data structure.
// `normal` is what lets the gather reject photons that landed on a *different*
// surface than the one being shaded, and it is what the light visualisation
// needs to turn a photon's flux back into radiance.
struct Photon {
    float3 pos; // world-space position
    float3 power; // Φ = flux carried by this photon [W]
    float3 dir; // unit vector pointing back along the incoming path
    float3 normal; // face-forward shading normal of the surface it landed on
    int depth; // bounces this photon made before landing (first hit = 0)
    // 4 x 12 + 4 = 52 bytes.  `depth` is what lets the gather spend one shared
    // path-length budget with the camera walk instead of two independent ones;
    // without it photon mapping reaches paths roughly twice as long as the path
    // tracer and the two modes are not solving the same problem.
};

// Which structure the gather searches the photon map with.
enum AccelType : int {
    ACCEL_BRUTE_FORCE = 0, // no structure: every photon is tested
    ACCEL_UNIFORM_GRID = 1, // dense array of cells covering the scene box
    ACCEL_HASH_GRID = 2, // cells stored in a hash table, sized by photon count
    ACCEL_STOCHASTIC_HASH_GRID = 3, // hash grid keeping one photon per bucket
    ACCEL_RECTIFIED_HASH_GRID = 4, // the same, the photon picked by reservoir sampling
    ACCEL_MLHASH = 5, // full hash grid over every cell level a lookup uses
    ACCEL_MLRECT = 6, // the same, one photon per bucket by reservoir sampling
    // Splatting (family C): no photon map at all.  The visible points go into
    // the structure, and every photon, as it is deposited, adds its flux to the
    // visible points it lies within (see VisiblePointGrid).
    ACCEL_SPLAT = 7, // visible points in one hash grid, cells F x the largest radius
    ACCEL_MLSPLAT = 8, // visible points each on the level that fits F x its radius
    ACCEL_STOCHSPLAT = 9, // mlsplat's grid, each deposit adding to one point it reaches
    // Adaptive LoD hash-grid SPPM (family D): a method of its own, with its
    // own programs, kernels and buffers (sppm_lepski.h).  Only the name lives
    // here; nothing in the other families' code branches on it.
    ACCEL_SPPM_LEPSKI = 10,
    // RTPM (family E): the photons in an acceleration structure, gathered by
    // an infinitely short ray (Kern et al. 2023).  It gathers at the camera
    // pass's gather points like the structures above, but its photon pass,
    // structures and collection pass are its own (rtpm.h).
    ACCEL_RTPM = 11
};

// The grids that file every photon once per cell level in use, so that each
// lookup can search the level that fits its own radius (see levelOf).
RT_INLINE bool isMultiLevelGrid(int accel)
{
    return accel == ACCEL_MLHASH || accel == ACCEL_MLRECT;
}

RT_INLINE bool isSplatting(int accel)
{
    return accel == ACCEL_SPLAT || accel == ACCEL_MLSPLAT || accel == ACCEL_STOCHSPLAT;
}

// The two hash grids that keep one photon per bucket.  They are filled by the
// photon pass itself rather than by a build afterwards, so the device needs to
// ask this while tracing as well as while gathering.
RT_INLINE bool isStochasticHashGrid(int accel)
{
    return accel == ACCEL_STOCHASTIC_HASH_GRID || accel == ACCEL_RECTIFIED_HASH_GRID;
}

// Photon grid — the uniform grid and the three hash grids share this layout.
//
// All of them divide space into cubic cells of side cell_size and search the
// cells the query's sphere overlaps (cellRange below).  The uniform grid and
// the hash grids differ in how a cell is turned into an index (an entry):
//   uniform grid:  index = the cell's position in a dense dims.x*dims.y*dims.z
//                  array, so every cell of the scene box has its own entry
//   hash grids:    index = hashCell(cell) & hash_mask, so the table has a fixed
//                  number of entries (buckets) and several cells can share one
//
// and the hash grids differ from each other only in what a bucket holds:
//   full hash grid:        every photon that hashed there — as for the uniform
//                          grid, entry e is grid_photon_ids[cell_start[e] ..
//                          cell_start[e+1]).  One array of offsets rather than
//                          a start/count pair: for the uniform grid this is the
//                          dense part, which grows as r^-3 as the radius
//                          shrinks, so it is worth keeping to a single array.
//   stochastic hash grid:  one photon, bucket_photon[e], standing in for all
//                          bucket_count[e] photons that hashed there
//   rectified stochastic:  the same, but the photon stays in photon_map and the
//                          bucket keeps only its slot, in bucket_pick[e]
//
// The multi-level grids use the same fields but cells of their own: level l's
// cells (side level_s0 * 2^l from level_origin, both in Params), hashed with
// hashCell4(cell, l) & hash_mask.  A photon is filed once per level in use,
// and an entry is its index | level << LEVEL_SHIFT rather than a bare index:
//   multi-level hash grid:  entries grouped by bucket, as in the full hash grid
//   multi-level rectified:  one entry per bucket, in bucket_pick, and the
//                           number filed there in bucket_count
struct PhotonGrid {
    float3 aabb_min; // origin of cell coordinates (uniform grid: also the box min)
    float3 aabb_max; // uniform grid only: box max
    int3 dims; // uniform grid only: number of cells in X,Y,Z
    float cell_size; // >= gather_radius (the *maximum* radius a gather may use)
    unsigned int hash_mask; // hash grids only: number of buckets - 1 (a power of two - 1)

    int* cell_start; // [cells + 1] (uniform grid) or [buckets + 1] (full hash grid)
    int* grid_photon_ids; // [num_stored] indices into photon_map, grouped by entry
    // --inline-positions (full and multi-level hash grid): the photon of entry
    // k's position at positions[k], next to the entries, so that a lookup
    // tests level, cell and distance without touching the photon map and
    // loads a photon only once it is within r.  nullptr = not built.
    float4* positions;

    // Stochastic hash grids only.
    int* bucket_count; // [buckets] photons that hashed into the bucket this pass
    Photon* bucket_photon; // [buckets] stochastic: whichever photon was written last
    unsigned long long* bucket_pick; // [buckets] rectified: key << 32 | slot of the pick
};

// Cell coordinate of a position along one axis.
//
// The build (cuda_passes.cu) and the lookup (device.cu) must compute this
// identically.  A photon filed under one cell and searched for under another is
// lost without any error, so both structures and both ends call this one
// function rather than repeating the expression.
RT_INLINE int photonCellCoord(float p, float origin, float cellSize)
{
    return (int)floorf((p - origin) / cellSize);
}

// The cells a sphere of radius r around x overlaps, along one axis: every
// cell c with lo <= c <= hi.  This one function is what the queries visit,
// what culling marks and what splatting files a visible point under, so the
// marked cells are exactly the visited ones.
//
// Exact for any cell size: a point within r of x lies between x - r and
// x + r on every axis, and rounding, subtraction, division and floor are all
// monotone, so its photonCellCoord lands between lo and hi in float
// arithmetic too, not just in exact arithmetic.  Cells between r and 2r wide
// give 2 or 3 cells per axis, cells at least 2r wide 1 or 2, and narrower
// cells more; rounding at the two ends can add one at most.
RT_INLINE void cellRange(float x, float r, float origin, float cellSize, int& lo, int& hi)
{
    lo = photonCellCoord(x - r, origin, cellSize);
    hi = photonCellCoord(x + r, origin, cellSize);
}

RT_INLINE void cellRange3(float3 x, float r, float3 origin, float cellSize, int3& lo, int3& hi)
{
    cellRange(x.x, r, origin.x, cellSize, lo.x, hi.x);
    cellRange(x.y, r, origin.y, cellSize, lo.y, hi.y);
    cellRange(x.z, r, origin.z, cellSize, lo.z, hi.z);
}

// The final mixing step of MurmurHash3: every input bit affects every output
// bit, which is what makes the low bits of a hash usable as a table index.
RT_INLINE unsigned int fmix32(unsigned int h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

// Hash of a cell, for the hash grid.
//
// The first line is the spatial hash of Teschner et al. (2003): multiply each
// coordinate by a large prime and XOR the products.  The table size is a power
// of two and the bucket is taken from the low bits (hash & mask), and the low
// bits of that product are a weak mix, so the result is passed through fmix32.
// Unsigned arithmetic throughout, so overflow wraps instead of being
// undefined, and negative coordinates hash like any others.
RT_INLINE unsigned int hashCell(int ix, int iy, int iz)
{
    unsigned int h = ((unsigned int)ix * 73856093u)
        ^ ((unsigned int)iy * 19349663u)
        ^ ((unsigned int)iz * 83492791u);
    return fmix32(h);
}

// Cell levels — shared by photon culling and the multi-level hash grids.
//
// Level l divides space into cubic cells of side s0 * 2^l, every level counted
// from one fixed origin (the scene box minimum), so a cell of level l+1 is
// exactly eight cells of level l.  A lookup of radius r uses the smallest level
// whose cells are at least r wide; its cells are then narrower than 2r, and
// its sphere overlaps 2 or 3 of them per axis (cellRange).
#define MAX_LEVELS 32

// value = photon index | level << 27 in the multi-level grids' entries.
#define LEVEL_SHIFT 27
#define LEVEL_INDEX_MASK ((1u << LEVEL_SHIFT) - 1u)

// Exact: scaling by a power of two only changes the exponent.
RT_INLINE float levelCellSize(float s0, int level)
{
    return ldexpf(s0, level);
}

// The smallest level in [0, MAX_LEVELS) whose cells are at least r wide.
//
// log2 only supplies the first guess.  Rounding in it could land one level
// low — cells narrower than r, and photons silently missed — so the guess is
// corrected against the exact cell size.  That also makes the answer a pure
// function of (r, s0): the host and the device compute the same level for the
// same radius, whatever their log2 does in the last bit.
RT_INLINE int levelOf(float r, float s0)
{
    if (!(r > 0.f))
        return 0;
    int l = (int)ceilf(log2f(r / s0));
    l = (l < 0) ? 0 : ((l > MAX_LEVELS - 1) ? MAX_LEVELS - 1 : l);
    while (l < MAX_LEVELS - 1 && levelCellSize(s0, l) < r)
        ++l;
    while (l > 0 && levelCellSize(s0, l - 1) >= r)
        --l;
    return l;
}

// The level-aware cell hash.  The level goes in before the final mix, so the
// same integer cell on two levels lands in unrelated buckets.
RT_INLINE unsigned int hashCell4(int ix, int iy, int iz, int level)
{
    return fmix32(hashCell(ix, iy, iz) ^ ((unsigned int)level * 0x9E3779B1u));
}

// Photon culling mask: one bit per (cell, level), addressed by a hash of the
// cell folded to `maskBits` bits.  Several cells can share a bit; that only
// ever keeps a photon nobody needed, never drops one that somebody does.
//
// The hash tables take their buckets from the low bits of hashCell4, so the
// mask must not: with the same low bits, every cell kept by a mask collision
// lands in exactly a bucket that a lookup of the colliding marked cell
// visits, and its photons cost every such lookup a test.  Mixing once more
// makes the bit independent of the bucket.
RT_INLINE unsigned int cullMaskBit(int ix, int iy, int iz, int level, unsigned int maskBits)
{
    return fmix32(hashCell4(ix, iy, iz, level) ^ 0x85EBCA6Bu) & ((1u << maskBits) - 1u);
}

RT_INLINE bool cullMaskTest(const unsigned int* mask, unsigned int maskBits,
    int ix, int iy, int iz, int level)
{
    const unsigned int b = cullMaskBit(ix, iy, iz, level, maskBits);
    return ((mask[b >> 5] >> (b & 31u)) & 1u) != 0u;
}

// PCG random numbers and the seed mixer every stream starts from (see the
// stream table in device.cu).  Here rather than there because the
// multi-level rectified grid's insertion kernel, plain CUDA, draws from them
// too.
RT_INLINE unsigned int pcg(unsigned int& s)
{
    s = s * 747796405u + 2891336453u;
    unsigned int w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

RT_INLINE unsigned int hashSeed3(unsigned int a, unsigned int b, unsigned int c)
{
    unsigned int s = a * 0x9E3779B1u + b * 0x85EBCA6Bu + c * 0xC2B2AE35u
        + 0x165667B1u;
    s ^= s >> 16;
    s *= 0x7FEB352Du;
    s ^= s >> 15;
    s *= 0x846CA68Bu;
    s ^= s >> 16;
    return s;
}

#define RNG_STREAM_MLRECT 0x510E527Fu
#define RNG_STREAM_STOCHSPLAT 0x9B05688Cu

// The random key of photon i's insertion on `level` into the multi-level
// rectified grid: a stream of its own, seeded by (i, level, frame, seed), so it
// owes nothing to the photon paths (whose random numbers stay untouched) or to
// the order in which threads happen to insert.  i < 2^27 and level < 32, so
// (i << 5) | level names each insertion uniquely.
RT_INLINE unsigned int mlReservoirKey(unsigned int i, int level, int frame, int seed)
{
    unsigned int s = hashSeed3((i << 5) | (unsigned int)level, (unsigned int)frame,
        (unsigned int)seed ^ RNG_STREAM_MLRECT);
    return pcg(s);
}

#ifdef __CUDACC__
// One insertion into a multi-level rectified bucket: count it, and keep it as
// the representative if its random key is the smallest so far.
//
// The keys are drawn independently of everything else (mlReservoirKey), so
// every insertion into a bucket is equally likely to hold the smallest one,
// and atomicMin makes the order in which threads arrive irrelevant: the
// winner is decided by the keys alone.  ("Replace with probability 1/k" would
// not be: with concurrent writers the last one wins, and which one is last is
// the scheduler's choice.)  The insertion's own entry sits in the low half of
// the 64-bit word, so the decision and the record of the winner are one
// atomic.  Shared by the insertion kernel (cuda_passes.cu) and, with
// --mlrect-in-trace, the photon pass (device.cu).
__device__ __forceinline__ void insertRepresentative(int* count, unsigned long long* best,
    unsigned int bucket, unsigned int i, int level, int frame, int seed)
{
    atomicAdd(&count[bucket], 1);
    const unsigned long long key = ((unsigned long long)mlReservoirKey(i, level, frame, seed) << 32)
        | (unsigned long long)(i | ((unsigned int)level << LEVEL_SHIFT));
    atomicMin(&best[bucket], key);
}
#endif

// One per pixel: what the camera pass found, and what the query pass needs in
// order to finish the pixel.
//
// The gather is split into two launches so that the photon lookup can be timed
// — and later swapped for a different acceleration structure — with none of the
// camera walk in the measurement.  It also makes the lookup directly checkable:
// run the query pass twice over the same gather points, once with the grid and
// once brute force, and the two images must agree.
struct GatherPoint {
    float3 pos; // first diffuse vertex the camera path reached
    float3 normal; // face-forward shading normal there
    float3 weight; // multiplies the irradiance: throughput * albedo / pi
    float3 direct; // radiance already settled for this pixel (emission, NEE, sky)
    int max_photon_depth; // photon path-length budget left at this vertex
    int valid; // 0 = the path never reached a diffuse vertex; only `direct` counts
    float r; // gather radius of this lookup (0 when !valid)
    int level; // levelOf(r): the cell level its search runs on
    float t; // length of the camera path to here, specular bounces included
};

// The splatting family's structure: the frame's visible points (gather
// points), each filed under every cell its sphere overlaps (cellRange) on its
// own level, so that a photon finds every visible point it lies within by
// looking only at its own cell.  The layout is the multi-level hash grid's
// with visible points in place of photons: bucket b holds the entries
// values[cell_start[b] .. cell_start[b+1]), each j | level << LEVEL_SHIFT for
// visible point (pixel) j, and a point appears at most once per bucket.
//
// splat:    one cell size, cell_size = F x the frame's largest radius, every
//           point on level index 0;
// mlsplat:  cell_size = 0 and point j on level levelOf(F x r_j), whose cells
//           are s0 * 2^level wide — at least F x r_j, so with F = 2 the
//           sphere overlaps at most 2 cells per axis.
struct VisiblePointGrid {
    int* cell_start; // [buckets + 1]
    unsigned int* values; // [entries]
    // --inline-positions: entry e's point's position and radius, so that a
    // deposit tests the distance without loading the whole gather point.
    float4* points; // [entries] or nullptr
    unsigned int hash_mask; // buckets - 1
    float cell_size; // > 0: the one cell size (splat); 0: per level (mlsplat)
    int num_levels; // the levels points are filed under, ascending
    int levels[MAX_LEVELS];
};

// Cell side of a visible-point grid level.
RT_INLINE float vpCellSize(const VisiblePointGrid& g, float s0, int level)
{
    return g.cell_size > 0.f ? g.cell_size : levelCellSize(s0, level);
}

// Where a lookup's radius comes from.
enum RadiusMode : int {
    RADIUS_GLOBAL = 0, // one radius per frame, gather_radius (the PPM schedule)
    RADIUS_FOOTPRINT = 1 // per pixel: radius_scale * k_pix * the pixel's footprint
};

// The analytic sky.  Used by BOTH the miss shader and the photon emitter so
// that path tracing and photon mapping are lit by exactly the same environment
// — when only the miss shader knew about it, every sky-driven bounce was
// missing from the photon map and the photon-mapped image came out dark.
RT_INLINE float3 skyRadiance(float3 dir, float intensity)
{
    float t = 0.5f * (dir.y + 1.f);
    t = (t < 0.f) ? 0.f : ((t > 1.f) ? 1.f : t);
    return make_float3(((1.f - t) * 1.00f + t * 0.40f) * intensity,
        ((1.f - t) * 0.95f + t * 0.60f) * intensity,
        ((1.f - t) * 0.85f + t * 1.00f) * intensity);
}

// Params
struct Params {
    // Output buffers
    uchar4* frame_buffer;
    float3* accum_buffer;

    // Scene data (device pointers)
    unsigned width;
    unsigned height;
    OptixTraversableHandle handle;

    // Camera
    float3 cam_eye;
    float3 cam_u; // right   (pre-scaled: half-width of the image plane)
    float3 cam_v; // up      (pre-scaled: half-height of the image plane)
    float3 cam_w; // forward to image-plane centre (its length is the plane distance)

    // Path tracing.
    //
    // Photon mapping pins samples_per_pixel to 1 and the camera pass ignores it.
    // Extra camera samples inside one frame all query the *same* photon map at
    // the *same* radius, so they do nothing for the density estimate — the error
    // that dominates a photon-mapped image — while multiplying the cost of the
    // gather.  The same budget spent on more frames buys independent photon maps
    // and a smaller radius from the PPM schedule, and the pixel jitter that
    // would have been the antialiasing is re-drawn every frame anyway.
    int samples_per_pixel;
    int max_depth;
    int frame_index;

    // Random seed for the whole run.  0 reproduces the previous behaviour; two
    // runs of identical settings that differ only in this are statistically
    // independent, which is what lets proof.py measure a render's noise level
    // directly instead of inferring it.
    int seed;

    // Scene geometry
    Triangle* triangles;
    Material* materials;

    // Scene bounds — the bounding sphere doubles as the emitter for sky photons
    // and as the reference for the ray epsilon, so both scale with the scene.
    float3 scene_center;
    float scene_radius;
    float scene_epsilon; // ray tmin / surface offset

    // The box the sphere came from.  Photons only ever land on geometry, so it
    // bounds the photon map too, and the grid build uses it instead of reducing
    // over the photons every frame: it costs a few empty cells where the scene
    // is emptier than its box, and saves a device-to-host round trip inside the
    // build — which at one build per frame is the thing worth saving.
    float3 scene_aabb_min;
    float3 scene_aabb_max;

    // NEE: emissive light list (device pointer, built by host)
    EmissiveTriangle* lights;
    int num_lights;
    float total_light_area; // sum of all light areas (for uniform sampling)

    // Environment light
    float sky_intensity; // 0 disables the sky in *both* render modes
    int emit_sky_photons; // 1 = the environment also shoots photons
    float sky_select_prob; // P(a photon path starts on the environment)

    // Photon map
    Photon* photon_map; // device pointer, size of photon_capacity
    int num_photon_paths; // paths emitted per pass
    int photon_capacity; // photon slots allocated in photon_map
    int* photon_count; // number of deposits attempted this pass (device pointer)
    float gather_radius; // maximum gather radius *for the current frame*
    int adaptive_radius; // 1 = shrink the radius towards target_photons
    int target_photons; // photons the adaptive radius aims to collect

    // Probabilistic PPM (Knaus & Zwicker 2011).  Every frame is an independent
    // photon-mapping estimate and the accumulation buffer averages them, so the
    // only thing needed to drive the bias to zero is a radius that shrinks with
    // the frame index:  r_{i+1} = r_i * sqrt((i + alpha)/(i + 1)),  i = 1..N.
    // The schedule is global and deterministic — no per-pixel statistics — which
    // is exactly why it drops into this renderer's existing accumulate loop.
    //
    // alpha = 2/3 balances the two error terms: averaging N frames leaves
    // bias^2 ~ N^(2*alpha-2) and variance ~ N^(-alpha), and 2*alpha-2 = -alpha
    // at alpha = 2/3, giving MSE ~ N^(-2/3).  It is also the value at which a
    // fixed photon budget is best spent on more frames rather than more photons
    // per frame.  Below 2/3 the radius shrinks faster and noise dominates;
    // above it the blur is what is left over.
    int use_ppm; // 1 = drive gather_radius from the schedule below
    float ppm_alpha; // the alpha in the recurrence
    float ppm_radius_initial; // r_1; gather_radius at frame_index 0
    int store_direct_photons; // 1 = also store first-hit photons (then the
                              // gather must NOT add its own direct lighting)

    // Per-pixel radius.  In footprint mode every lookup gets
    //
    //     r = radius_scale * k_pix * t * pixel_angle,
    //
    // k_pix times the width of surface one pixel covers at the end of a camera
    // path of length t (mirror and glass segments included, since they carry
    // the pixel's cone further), shrunk over the frames by the same schedule as
    // the global radius: radius_scale = r_i / r_1 = ppmRadius(1, alpha, i).
    // Recomputed every frame — the gather point itself moves every frame.
    int radius_mode; // RadiusMode
    float radius_scale; // the schedule's factor for this frame; 1 without PPM
    float k_pix; // radius in pixel footprints
    float pixel_angle; // 2|cam_v| / (|cam_w| height), the angle one pixel spans
    float radius_min; // clamp of a footprint radius: 1e-6 of the scene diagonal
    float radius_max; //   ... and a quarter of it; a safety net only
    float max_gather_radius; // largest r of the frame's lookups (host side; what
                             // a single-level grid has to size its cells by)

    // Photon rejection tolerances — these are what stop a photon stored on one
    // wall from bleeding onto the neighbouring one.
    float photon_normal_tol; // min dot(n_surface, n_photon)
    float photon_plane_tol; // max |dot(photon - x, n)| as a fraction of r

    // Photon grid
    PhotonGrid grid;
    int accel; // AccelType of this frame; the photon pass reads it too, to fill
               // a stochastic hash grid while it traces
    float table_load; // host side: every hash table gets at least this many
                      // buckets per entry it holds (--table-load)
    int inline_positions; // host side: the hash grids and the splatting grid
                          // keep positions next to their entries

    // Cell levels (see levelOf), the same for every frame of a run.
    float level_s0; // side of a level-0 cell; level l has side s0 * 2^l
    float3 level_origin; // where every level's cell coordinates start

    // The levels this frame's lookups use.  The camera pass flags each level
    // a valid gather point lands on; the host reads the 32 flags back and
    // hands the photon pass and the build the list of levels that occur.
    int* used_levels; // [MAX_LEVELS] device flags; nullptr = nobody needs them
    int num_levels;
    int levels[MAX_LEVELS]; // ascending, the first num_levels are valid

    // Photon culling.  The mark pass, after the camera pass, marks the cells
    // every lookup's sphere overlaps at the lookup's own level, and the photon
    // pass stores only photons that land in a marked cell on one of the
    // levels in use.  The mask does not depend on the search structure, so
    // the same culled map serves every one of them.
    int mlrect_in_trace; // 1 = the photon pass fills the multi-level rectified
                         // grid as it stores, instead of a kernel afterwards
    int culling; // 1 = store only photons some lookup of this frame can reach
    unsigned int* cull_mask; // [2^cull_mask_bits / 32] one bit per hashed (cell, level)
    unsigned int cull_mask_bits;
    // --hier-mask: a second mask, of the same size and hash, holding the
    // parent cells of every mark at every coarser level in use.  The photon
    // pass tests levels coarse to fine and culls as soon as neither mask has
    // the photon's cell — then no lookup on that level or a finer one marked a
    // cell around it.  Every photon some lookup can use is still kept; what it
    // keeps fewer of than the flat test are photons that only a hash
    // collision in cull_mask, on a finer level, would have kept.
    int hier_mask;
    unsigned int* cull_parent;
    int* deposit_counts; // [2] culling only: photons deposited, photons kept

    // --fused: the camera walk and the lookup in one program.  It records
    // what the next frame's structure is sized from — the frame's largest
    // radius (float bits, atomicMax) and its levels (used_levels) — and stores
    // its gather points only when asked to (--verify-gather).
    unsigned int* fused_max_r;
    int fused_store_points;

    // The two halves of a split gather (photon mapping only).
    GatherPoint* gather_points; // [width*height], written by __raygen__camera
    float3* query_radiance; // [width*height], written by the query or finalize pass

    // Splatting (accel splat / mlsplat).  The photon pass adds w * power of
    // every (deposit, visible point) pair to the point's flux here; the
    // finalize pass turns the flux into radiance.  splat_stripes copies of the
    // accumulators, photon path i adding to copy i % splat_stripes, so that
    // fewer atomics collide on a busy point; finalize sums the copies.
    VisiblePointGrid vpgrid;
    float splat_cell_factor; // F: points are filed on cells at least F x r wide
    float* splat_acc; // [splat_stripes][width*height][3]
    int splat_stripes;
    unsigned long long* splat_counts; // [3] deposits, deposits reaching a point, pairs

    // --stats: per lookup, photon records tested, photons accepted and cells
    // visited ([width*height] each, -1 where a pixel has no lookup).  With
    // splatting the same counters are kept per deposit instead — visible-point
    // entries tested and pairs made — at slots handed out by stat_slot, up to
    // stat_capacity of them.
    int stats;
    int* stat_tested;
    int* stat_accepted;
    int* stat_cells;
    int* stat_slot;
    int stat_capacity;

    // Render mode flag
    int render_mode; // 0 = path tracer, 1 = photon mapping, 2 = photon tracing only
    int offline_frames; // total number of frames to render in offline mode

    // Photon power scaling factor (artistic override, 1 = physically correct)
    float photon_power_scale;

    // Light visualization buffer (for photon tracing only mode)
    float* lightvis_buffer;
    float lightvis_exposure; // display gain applied to the visualised radiance
    int lightvis_splat_px; // splat radius in pixels (0 = one pixel per photon)
};
