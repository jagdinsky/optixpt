#pragma once
#include "packed_photon.h"
#include "params.h"

// Adaptive LoD hash-grid SPPM ("sppm_lepski", family D).
//
// A camera-first stochastic progressive photon mapper of its own, next to the
// gathering and splatting families:
//
//   stage 1  camera walk to the first diffuse vertex, direct light by NEE
//            (OptiX, sppm_lepski_device.cu)
//   stage 2  photon trace into a flat array of 32-byte photons (OptiX)
//   stage 3  counting sort of the photons into one hash table (CUDA + CUB)
//   stage 4  the gather at five radii at once, a running mean and variance
//            per radius, the choice of the one to show (Lepski's rule), and
//            the image (CUDA)
//
// The radii are fixed — multiples of the pixel's footprint — and it is the
// choice among them that adapts, per pixel, as the estimates firm up; there
// is no radius schedule.  The photons live in one hash table over a grid with
// a level per octave of distance from the camera, and only those in cells
// some gather of the iteration searches are stored at all.
//
// This header is what the three compilations share: the OptiX programs, the
// CUDA kernels (sppm_lepski.cu) and the host (main.cpp, through
// sppm_lepski_host.h).  None of it is used by the other methods.

// The hash table and the active-cell bitset are addressed by the same hash.
#define LEPSKI_HASH_BITS 22
#define LEPSKI_HASH_SIZE (1u << LEPSKI_HASH_BITS) // 4,194,304 buckets

// The bandwidths of the multi-scale estimate: scale k gathers within
// lepskiScale(k) * rBase, each twice the one before.
#define LEPSKI_K_SCALES 5
RT_INLINE float lepskiScale(int k) { return 0.25f * (float)(1 << k); } // 0.25 0.5 1 2 4

// Photon rejection, as fractions: a photon counts only if its surface normal
// agrees with the gather point's, and if it lies in a thin slab around the
// gather point's tangent plane (what keeps light from leaking round corners).
#define LEPSKI_NORMAL_TOL 0.15f // min dot(n_gather, n_photon)
#define LEPSKI_PLANE_TOL 0.25f // max |offset along n|, as a fraction of the largest scale's radius

// What stage 1 leaves for a pixel: where to gather and what to multiply the
// gathered flux density by.  64 bytes.
//
// `weight` is the path throughput times the BSDF at the vertex without the
// cosine, f_r = albedo / pi: the density of photon hits on a surface already
// carries the cosine of their incidence.  (Direct light, by contrast, is
// estimated per solid angle and gets its cosine in directLighting.)
struct LepskiGatherPoint {
    float3 pos; // the first diffuse vertex of the camera path
    float r_base; // radius of scale 1; 0 = nothing to gather here
    float3 normal; // face-forward shading normal
    float t; // length of the camera path to here, specular segments included
    float3 weight; // throughput * albedo / pi
    int info; // lepskiPackInfo: the levels the gather searches, the photon budget
    float3 direct; // radiance settled without the photon map: emission, NEE, sky
    unsigned int wo_oct; // unit vector back along the camera path (2 x snorm16)
};
static_assert(sizeof(LepskiGatherPoint) == 64, "LepskiGatherPoint must be 64 bytes");

// LepskiGatherPoint::info: the first and last LoD level the gather searches,
// and the photon bounce budget left at the vertex (the path-length budget
// the camera walk and the photon share).
RT_INLINE int lepskiPackInfo(int lMin, int lMax, int maxPhotonDepth)
{
    return lMin | (lMax << 8) | (maxPhotonDepth << 16);
}
RT_INLINE int lepskiLevelMin(int info) { return info & 0xFF; }
RT_INLINE int lepskiLevelMax(int info) { return (info >> 8) & 0xFF; }
RT_INLINE int lepskiMaxPhotonDepth(int info) { return (info >> 16) & 0xFF; }

// The running estimate of a pixel's indirect light at every scale.
struct PixelLepskiState {
    float3 meanRGB[LEPSKI_K_SCALES];
    float meanLum[LEPSKI_K_SCALES];
    float m2Lum[LEPSKI_K_SCALES]; // Welford: sum of squared luminance deviations
    unsigned int totalCountMax; // photons gathered within the largest scale, all iterations
};

// What the image shows (--lepski-view).
enum LepskiView : int {
    LEPSKI_VIEW_NORMAL = 0, // direct + indirect at the selected scale
    LEPSKI_VIEW_INDIRECT = 1, // indirect at the selected scale
    LEPSKI_VIEW_HEATMAP = 2, // the selected scale, colour-coded
    LEPSKI_VIEW_FIXED = 3 // direct + indirect at scale 1 (index 2), no selection
};

// Device counters of one iteration (LepskiParams::counters).
enum LepskiCounter : int {
    LEPSKI_COUNT_STORED = 0, // photons appended (may exceed max_stored: then saturated)
    LEPSKI_COUNT_CANDIDATES = 1, // deposits that could have been stored
    LEPSKI_COUNT_ACTIVE = 2, // ... of which landed in an active cell
    LEPSKI_COUNTERS = 4
};

// The method's own launch parameters.  The OptiX programs read them from
// their ray-generation SBT record (so Params, which every existing program
// is compiled against, is left as it is); the kernels get them by value.
struct LepskiParams {
    // The image
    unsigned int width;
    unsigned int height;
    float3* accum_buffer; // [W*H] iteration x (running mean): the exporter divides by the frame count
    uchar4* frame_buffer; // [W*H] tone-mapped running mean, or nullptr
    float3* frame_radiance; // [W*H] this iteration's own estimate, or nullptr

    int iteration; // M >= 1, counted from the last reset
    int view; // LepskiView
    float kappa; // half-width of the confidence intervals, in standard errors

    // --stats: per pixel, photons the gather read, photons within the largest
    // scale, buckets visited (-1 where the pixel gathers nothing), or nullptr.
    int* stat_tested;
    int* stat_accepted;
    int* stat_cells;

    // The LoD grid: level l holds the photons whose distance from the camera
    // is in [d0 2^l, d0 2^(l+1)) (level 0: everything nearer than 2 d0), in
    // cells of side h0 2^l.
    float3 origin; // cell coordinates are counted from here (the scene box minimum)
    float d0;
    float h0;
    int max_lod; // number of levels

    // The radius: scale 1 is base_scale pixel footprints at normal incidence.
    float base_scale;

    // Photon storage
    int culling; // 0 = every photon is stored, whatever cell it lands in
    float flux_threshold; // luminance below which a photon is stored by roulette; 0 = never
    unsigned int* active_bits; // [HASH_SIZE / 32] the buckets some gather searches

    // Stage buffers
    LepskiGatherPoint* gather_points; // [W*H]
    PackedPhoton* raw_photons; // [max_stored] in the order they were deposited
    unsigned int* raw_hashes; // [max_stored] bucket of each
    unsigned int max_stored;
    unsigned int* counters; // [LEPSKI_COUNTERS]
    unsigned int* cell_count; // [HASH_SIZE + 1] photons per bucket; the last stays 0
    unsigned int* cell_start; // [HASH_SIZE + 1] exclusive scan of the counts
    unsigned int* cell_cursor; // [HASH_SIZE + 1] the scatter's write positions
    PackedPhoton* sorted_photons; // [max_stored] grouped by bucket
    PixelLepskiState* pixel_state; // [W*H]
    float3* sum_direct; // [W*H] sum of `direct` over the iterations
};

// The method's device memory.  Allocated once for an image size and a photon
// capacity and kept until either changes: an iteration allocates nothing.
struct LepskiBuffers {
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int max_stored = 0;
    LepskiGatherPoint* gather_points = nullptr;
    PackedPhoton* raw_photons = nullptr;
    unsigned int* raw_hashes = nullptr;
    PackedPhoton* sorted_photons = nullptr;
    unsigned int* active_bits = nullptr;
    unsigned int* counters = nullptr;
    unsigned int* cell_count = nullptr;
    unsigned int* cell_start = nullptr;
    unsigned int* cell_cursor = nullptr;
    PixelLepskiState* pixel_state = nullptr;
    float3* sum_direct = nullptr;
    void* cub_temp = nullptr; // CUB's working storage for the scan
    size_t cub_temp_bytes = 0;
    void* stats_temp = nullptr; // ... and for the statistics' reduction
    size_t stats_temp_bytes = 0;
    void* stats_out = nullptr; // the reduction's one-element result
    size_t bytes = 0; // all of the above
};

// Makes the buffers fit a width x height image and max_stored photons.
// Returns true if it had to (re)allocate: the running estimate is then gone,
// and the caller starts again at iteration 1.
bool lepskiAllocate(LepskiBuffers& b, unsigned int width, unsigned int height,
    unsigned int maxStored);
void lepskiFree(LepskiBuffers& b);

// Points L at the buffers.
void lepskiBind(const LepskiBuffers& b, LepskiParams& L);

// The start of an iteration: clears the active-cell bitset and the counters,
// which stages 1 and 2 fill.  (The bucket counts are stage 3's alone and are
// cleared there, so that their clearing is timed with the build.)
void lepskiBeginIteration(const LepskiParams& L, cudaStream_t stream);

// Clears the photon counters alone: a second photon pass over the same
// camera stage (--verify-gather).
void lepskiClearCounters(const LepskiParams& L, cudaStream_t stream);

// Stage 3: the first `stored` raw photons, grouped by bucket, into
// sorted_photons, and the buckets' offsets into cell_start.
void lepskiBuildGrid(const LepskiParams& L, const LepskiBuffers& b, unsigned int stored,
    cudaStream_t stream);

// Stage 4: the gather, the running estimate and the image.
void lepskiGather(const LepskiParams& L, cudaStream_t stream);

// --verify-gather: this iteration's gather on its own, into `out` [W*H] and
// with nothing else written — through the grid, or (`brute`) testing every
// stored photon for every pixel.  A pixel gets the sum of its estimates at
// all the scales, so a photon missing from any of them shows.
void lepskiProbe(const LepskiParams& L, bool brute, float3* out, cudaStream_t stream);

// What the iteration's gathers reach for, over the pixels that gather: the
// radius of the largest scale.  Reduced on the device; statistics only.
struct LepskiRadiusStats {
    float max_r = 0.f;
    double sum_r = 0.0;
    double sum_px = 0.0; // the radius in pixel footprints, r / (t * pixel_angle)
    long long valid = 0;
};
void lepskiRadiusStats(const LepskiParams& L, const LepskiBuffers& b, float pixelAngle,
    cudaStream_t stream, LepskiRadiusStats& out);

// How many pixels the running estimate currently shows at each scale, and (the
// last entry) how many have not gathered a photon yet.  Statistics only.
void lepskiScaleHistogram(const LepskiParams& L, const LepskiBuffers& b,
    unsigned int histogram[LEPSKI_K_SCALES + 1], cudaStream_t stream);

// Registers and local memory per thread of the method's kernels, for the
// timing record.
struct LepskiKernelInfo {
    const char* name;
    int registers;
    size_t local_bytes;
};
int lepskiKernelInfo(LepskiKernelInfo out[4]);

#ifndef __CUDACC__
#include <cstdlib>
#include <iostream>
#include <string>

// The method's command-line options (--lepski-*).
//
// Two defaults are not the values the method was first specified with (h0 =
// 0.08 d0, kappa = 2), because both measured badly (experiment E8):
//
//   h0     The gather's cost goes with the area of a cell.  At 0.08 d0 the
//          cells are 10 to 20 times wider than the largest radius at normal
//          incidence — wide enough to leave it unclamped down to the cosine
//          floor — and a gather reads about 3800 photons at 1280x720.  At
//          0.02 it reads about 250; only radii at grazing angles are clamped.
//   kappa  While the small scales hold few photons, their sample variance
//          is too small exactly when their estimate is low, so a low estimate
//          vetoes the larger scales and the image comes out dark (1.5% of
//          its brightness on the Cornell box at 16 frames, gone by about
//          1000).  At 4 standard errors the bias is within 0.1%, and the
//          error is lower at every frame count measured.
struct LepskiOptions {
    float d0 = 0.01f; // LoD distance unit, as a fraction of the scene diagonal
    float h0 = 0.02f; // level-0 cell side, as a fraction of d0
    int maxLod = 12; // LoD levels
    float fluxRR = 0.1f; // storage roulette below this fraction of the mean emitted photon flux
    bool culling = true; // store only photons in cells a gather searches
    long long maxPhotons = 0; // photon slots; 0 = the most a pass can deposit (capped)
    float kappa = 4.f; // confidence interval half-width of the scale selection
    int view = LEPSKI_VIEW_NORMAL; // LepskiView
};

inline const char* lepskiViewName(int view)
{
    switch (view) {
    case LEPSKI_VIEW_INDIRECT:
        return "indirect";
    case LEPSKI_VIEW_HEATMAP:
        return "heatmap";
    case LEPSKI_VIEW_FIXED:
        return "fixed";
    default:
        return "normal";
    }
}

// Consumes argv[i] (and its value) if it is one of the method's options.
inline bool lepskiParseOption(LepskiOptions& o, int argc, char** argv, int& i)
{
    const std::string a = argv[i];
    if (a.compare(0, 9, "--lepski-") != 0)
        return false;
    auto bad = [&](const char* expects) {
        std::cerr << a << " expects " << expects << "\n";
        std::exit(2);
    };
    if (i + 1 >= argc)
        bad("a value");
    const std::string v = argv[++i];
    if (a == "--lepski-d0") {
        o.d0 = (float)atof(v.c_str());
        if (!(o.d0 > 0.f && o.d0 <= 1.f))
            bad("a fraction of the scene diagonal in (0, 1]");
    } else if (a == "--lepski-h0") {
        // No wider than d0: that is what keeps a gather within two levels.
        o.h0 = (float)atof(v.c_str());
        if (!(o.h0 > 0.f && o.h0 <= 1.f))
            bad("a fraction of d0 in (0, 1]");
    } else if (a == "--lepski-max-lod") {
        o.maxLod = atoi(v.c_str());
        if (o.maxLod < 1 || o.maxLod > 24)
            bad("a number of levels in 1..24");
    } else if (a == "--lepski-flux-rr") {
        o.fluxRR = (float)atof(v.c_str());
        if (!(o.fluxRR >= 0.f))
            bad("a fraction of the mean photon flux, 0 to turn the roulette off");
    } else if (a == "--lepski-culling") {
        if (v != "on" && v != "off")
            bad("on or off");
        o.culling = (v == "on");
    } else if (a == "--lepski-max-photons") {
        o.maxPhotons = atoll(v.c_str());
        if (o.maxPhotons < 0 || o.maxPhotons > (1LL << 30))
            bad("a number of photon slots, 0 for automatic");
    } else if (a == "--lepski-kappa") {
        o.kappa = (float)atof(v.c_str());
        if (!(o.kappa > 0.f))
            bad("a positive number of standard errors");
    } else if (a == "--lepski-view") {
        if (v == "normal")
            o.view = LEPSKI_VIEW_NORMAL;
        else if (v == "indirect")
            o.view = LEPSKI_VIEW_INDIRECT;
        else if (v == "heatmap")
            o.view = LEPSKI_VIEW_HEATMAP;
        else if (v == "fixed")
            o.view = LEPSKI_VIEW_FIXED;
        else
            bad("normal, indirect, heatmap or fixed");
    } else {
        std::cerr << "Unknown option: " << a << "\n";
        std::exit(2);
    }
    return true;
}
#endif // !__CUDACC__

#ifdef __CUDACC__
#include "gather.h" // float3 operators, dot, normalize, length

// The bucket of cell (ix, iy, iz) on LoD level `level`.
__device__ __forceinline__ unsigned int SpatialHash4D(int ix, int iy, int iz, int level)
{
    unsigned int h = (unsigned int)ix * 73856093u
        ^ (unsigned int)iy * 19349663u
        ^ (unsigned int)iz * 83492791u
        ^ (unsigned int)level * 25165843u;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h & (LEPSKI_HASH_SIZE - 1u);
}

// The 2 x 2 x 2 block of cells of side h that a gather around x searches,
// along one axis: cells b and b + 1.  x lies between b + 0.5 and b + 1.5
// cells from the origin, so the block reaches at least h / 2 to either side
// of it: it holds every photon within any radius up to h / 2.  The photon's
// own cell is photonCellCoord, the function the other grids use.
__device__ __forceinline__ int lepskiBlockCoord(float x, float origin, float h)
{
    return (int)floorf((x - origin) / h - 0.5f);
}

__device__ __forceinline__ float lepskiLuminance(float3 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// The LoD level of a point `dist` from the camera: floor(log2(dist / d0)),
// kept within [0, maxLod).  ilogbf reads the exponent, so it is that floor
// exactly, and the level never goes down as the distance goes up — which is
// what lets a gather bound the levels of the photons around it by the levels
// of its nearest and farthest reach.
__device__ __forceinline__ int ComputeLoDLevel(float dist, float d0, int maxLod)
{
    const float ratio = fmaxf(dist, d0) / d0;
    const int lvl = ilogbf(ratio);
    return min(max(lvl, 0), maxLod - 1);
}

// Exact: scaling by a power of two only changes the exponent.
__device__ __forceinline__ float lepskiCellSize(float h0, int level)
{
    return ldexpf(h0, level);
}

// The bucket a photon at p is filed in: one level, from its distance to the
// camera, and its own cell there.
__device__ __forceinline__ unsigned int lepskiPhotonBucket(const LepskiParams& L, float3 p,
    float3 camPos)
{
    const int level = ComputeLoDLevel(length(p - camPos), L.d0, L.max_lod);
    const float h = lepskiCellSize(L.h0, level);
    return SpatialHash4D(photonCellCoord(p.x, L.origin.x, h),
        photonCellCoord(p.y, L.origin.y, h),
        photonCellCoord(p.z, L.origin.z, h), level);
}

// The radius of scale 1 at a gather point, and the levels its gather has to
// search.
//
// The radius is the pixel's footprint on the surface: base_scale pixels'
// width at the end of a camera path of length `pathLength` (specular
// segments included), stretched by the surface's tilt against the path, the
// cosine floored at 0.2.
//
// It is then clamped so that rMax, the radius of the largest scale, is at
// most 0.49 cells of the finest level any photon within rMax can be on.
// Photons within rMax of x are between camDist - rMax and camDist + rMax from
// the camera (triangle inequality), so their levels run from lMin to lMax
// below; every one of those levels has cells at least as wide as the level
// the clamp used, so its 2 x 2 x 2 block around x (lepskiBlockCoord) holds
// all of them.  The reach is padded by a part in a thousand so that the
// rounding of a distance at a level boundary cannot move a photon out of
// that range.  With h0 <= d0 the range is never more than two levels: rMax
// is under half a level-lMin cell, and level lMin + 1 is twice lMin's whole
// depth.
__device__ __forceinline__ float lepskiRadius(const LepskiParams& L, float pathLength,
    float pixelAngle, float cosTheta, float camDist, float rMin, int& lMin, int& lMax)
{
    const float maxScale = lepskiScale(LEPSKI_K_SCALES - 1);
    const float pad = 1.001f;

    const float pixelWorldSize = pathLength * pixelAngle;
    const float rBaseRaw = fmaxf(L.base_scale * pixelWorldSize / fmaxf(cosTheta, 0.2f), rMin);

    const int lEst = ComputeLoDLevel(camDist - pad * maxScale * rBaseRaw, L.d0, L.max_lod);
    const float rBase = fminf(rBaseRaw, 0.49f * lepskiCellSize(L.h0, lEst) / maxScale);
    const float rMax = maxScale * rBase;

    lMin = ComputeLoDLevel(camDist - pad * rMax, L.d0, L.max_lod);
    lMax = min(ComputeLoDLevel(camDist + pad * rMax, L.d0, L.max_lod), lMin + 1);
    return rBase;
}

// The buckets a gather around x searches — the 2 x 2 x 2 block of every level
// from lMin to lMax — each once: two cells can share a bucket, and its
// photons must not count twice.  At most 16; returns how many.  Stage 1 marks
// exactly these and stage 4 visits exactly these (integer arithmetic on the
// same divisions, so the two compilations agree).
__device__ __forceinline__ int lepskiGatherBuckets(const LepskiParams& L, float3 x, int lMin,
    int lMax, unsigned int* buckets)
{
    int count = 0;
    for (int level = lMin; level <= lMax; ++level) {
        const float h = lepskiCellSize(L.h0, level);
        const int bx = lepskiBlockCoord(x.x, L.origin.x, h);
        const int by = lepskiBlockCoord(x.y, L.origin.y, h);
        const int bz = lepskiBlockCoord(x.z, L.origin.z, h);
        for (int c = 0; c < 8; ++c) {
            const unsigned int bucket = SpatialHash4D(bx + (c & 1), by + ((c >> 1) & 1),
                bz + (c >> 2), level);
            bool seen = false;
            for (int i = 0; i < count; ++i)
                seen |= (buckets[i] == bucket);
            if (!seen)
                buckets[count++] = bucket;
        }
    }
    return count;
}
#endif // __CUDACC__
