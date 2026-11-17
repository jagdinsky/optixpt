#pragma once
#include "packed_photon.h"
#include "params.h"

// RTPM, the hardware ray tracing progressive photon mapper of Kern, Brüll and
// Grosch, "Accelerating Photon Mapping for Hardware-Based Ray Tracing" (JCGT
// 12(1), 2023) — family E, --accel rtpm.
//
// What is the paper's:
//
//   - the photon map is an acceleration structure: every photon a box of its
//     radius, rebuilt every iteration, never refitted (sec. 3.5);
//   - the gather is an inverse radius search: an infinitely short ray from
//     the gather point visits the boxes that contain it, an intersection
//     program keeps the photons whose sphere holds the point, and an any-hit
//     program adds them up (sec. 3.6, after Evangelou et al. 2021);
//   - two photon maps, caustic (L S+ D) and global (everything else), each
//     with its own radius, and a rejection step that stores only a fraction
//     of the global photons and scales the survivors up (secs. 3.4, 3.5);
//   - Photon Culling: a hash-addressed mask of the cells around this
//     iteration's gather points, cells as wide as a photon's diameter, and no
//     photon stored outside them (sec. 3.7);
//   - Stochastic Evaluation: the any-hit program only keeps a reservoir of a
//     few photon indices in the payload, and those are evaluated once the
//     traversal is over (sec. 3.8).
//
// What is this renderer's, so that the method converges to the image every
// other method here converges to and can be checked photon for photon:
//
//   - the transport: the camera pass and its gather points (cameraWalk),
//     direct light by next event estimation, no first-hit photons, and the
//     path-length budget shared between the camera walk and the photon;
//   - the estimate: photonUsable's tests and the cone filter (gather.h),
//     where the paper sums f_r * flux over a disc — with the defaults below
//     (one radius, nothing rejected, everything evaluated) the method renders
//     the full hash grid's image, and only the search differs;
//   - the photon count of an iteration is read back and the structures are
//     built over exactly that many boxes, where the paper sizes the build
//     from the iteration before to avoid the read-back (every pass here is
//     timed with a synchronisation anyway);
//   - the two structures are traced by their own handles, where the paper
//     puts them in one top-level structure and tells them apart by instance
//     mask.
//
// This header is what the three compilations share: the OptiX programs
// (rtpm_device.cu), the CUDA kernels (rtpm.cu) and the host (main.cpp,
// through rtpm_host.h).  None of it is used by the other methods.

enum RtpmMapIndex : int {
    RTPM_CAUSTIC = 0, // photons that reached their surface from the light through specular bounces only
    RTPM_GLOBAL = 1, // every other photon
    RTPM_MAPS = 2
};

// Stochastic Evaluation keeps its photon list in the ray payload; this many
// slots are set aside for it (the paper used 3 throughout).
#define RTPM_MAX_LIST 3

// Device counters of one iteration (RtpmParams::counters).  The first two
// are the maps' append counters, indexed by RtpmMapIndex.
enum RtpmCounter : int {
    RTPM_COUNT_CANDIDATES = 2, // deposits past the first hit
    RTPM_COUNT_VISIBLE = 3, // ... of which in a marked cell (all of them without culling)
    RTPM_COUNTERS = 4
};

// One photon map.
struct RtpmMap {
    OptixAabb* aabbs; // [capacity] the photons' boxes, the build's input
    PackedPhoton* photons; // [capacity] the photons, in the same order
    unsigned int capacity;
    unsigned int stored; // photons of this iteration (set once the pass has been read back)
    float box_radius; // half the side of a box: the largest radius any gather uses with this map
    float radius_scale; // a gather's radius with this map, as a multiple of the gather point's
    OptixTraversableHandle handle; // the structure built over `stored` boxes
};

// The method's own launch parameters.  The ray-generation programs read them
// from their SBT record (so Params, which every existing program is compiled
// against, is left as it is); the kernels get them by value.
struct RtpmParams {
    RtpmMap maps[RTPM_MAPS];
    unsigned int* counters; // [RTPM_COUNTERS]

    // Photon Culling
    int culling;
    unsigned char* cull_mask; // [1 << cull_bits] one byte per hashed cell
    unsigned int cull_bits;
    float cull_cell; // cell side: the diameter of a global photon's box
    float3 origin; // cell coordinates are counted from here (the scene box minimum)

    float reject; // probability that a global photon is not stored
    int list_size; // Stochastic Evaluation: photons evaluated per map; 0 = all of them
    int uniform_kernel; // 1 = the paper's estimate, every photon in the disc alike; 0 = the cone filter
    int stats; // --stats: boxes visited and photons accepted, per pixel
};

// What the photon programs get from their hit group's SBT record: the map
// whose structure the ray is in.
struct RtpmMapRecord {
    const PackedPhoton* photons;
    int list_size;
    int uniform_kernel;
    int stats;
};

// The ray payload of a collection ray.  The gather point's side goes in, the
// sum comes out; with Stochastic Evaluation the list comes out instead.
enum RtpmPayload : int {
    RTPM_P_FLUX = 0, // 0..2: weighted flux
    RTPM_P_ACCEPTED = 3, // photons that passed every test
    RTPM_P_NORMAL = 4, // 4..6: the gather point's normal
    RTPM_P_R2 = 7, // squared radius of this gather with this map
    RTPM_P_MAX_DEPTH = 8, // photon bounce budget left at the gather point
    RTPM_P_LIST = 9, // 9..11: Stochastic Evaluation's photon indices
    RTPM_P_RNG = 12, // ... and its random state
    RTPM_P_TESTED = 13, // boxes the traversal handed to the intersection program
    RTPM_P_PLANE_TOL = 14,
    RTPM_P_INV_R = 15
};

// The method's device memory, allocated once for a photon capacity and kept
// until it changes: an iteration allocates nothing.  (The acceleration
// structures' buffers are the host's, rtpm_host.h: they are OptiX's.)
struct RtpmBuffers {
    OptixAabb* aabbs[RTPM_MAPS] = {};
    PackedPhoton* photons[RTPM_MAPS] = {};
    unsigned int capacity[RTPM_MAPS] = {};
    unsigned int* counters = nullptr;
    unsigned char* cull_mask = nullptr;
    unsigned int cull_bits = 0;
    size_t bytes = 0; // all of the above
};

// Makes the buffers fit; true if it had to (re)allocate.
bool rtpmAllocate(RtpmBuffers& b, unsigned int causticCapacity, unsigned int globalCapacity,
    unsigned int cullBits);
void rtpmFree(RtpmBuffers& b);

// Points R at the buffers.
void rtpmBind(const RtpmBuffers& b, RtpmParams& R);

void rtpmClearCounters(const RtpmParams& R, cudaStream_t stream);

// Photon Culling, the camera side: clears the mask and marks the eight cells
// nearest to every valid gather point (params.gather_points).
void rtpmMarkCells(const Params& params, const RtpmParams& R, cudaStream_t stream);

// --verify-gather: the collection by brute force — every stored photon of
// both maps tested for every pixel, with the tests and weights of the any-hit
// program — into `out` [W*H].
void rtpmBruteCollect(const Params& params, const RtpmParams& R, float3* out, cudaStream_t stream);

// Registers and local memory per thread of the method's kernels.
struct RtpmKernelInfo {
    const char* name;
    int registers;
    size_t local_bytes;
};
int rtpmKernelInfo(RtpmKernelInfo out[2]);

#ifndef __CUDACC__
#include <cstdlib>
#include <iostream>
#include <string>

// The method's command-line options (--rtpm-*).
//
// The defaults make the method an exact structure — every photon stored and
// evaluated, one radius, the cone filter — so that it renders what the full
// hash grid renders and the two differ in the search alone.  The paper's own
// test configuration is --rtpm-culling on --rtpm-reject 0.7 --rtpm-stochastic
// 3 --rtpm-kernel uniform, with a caustic radius chosen per scene
// (--rtpm-caustic-scale).
struct RtpmOptions {
    bool culling = false; // Photon Culling
    int cullBits = 22; // its mask has 2^cullBits cells (the paper's 2^22)
    float reject = 0.f; // rejection probability of global photons (the paper's tests: 0.7)
    int listSize = 0; // Stochastic Evaluation's list (the paper's tests: 3); 0 = evaluate all
    float causticScale = 1.f; // caustic radius as a fraction of the global radius
    long long maxPhotons = 0; // photon slots of the global map; 0 = the most a pass can deposit (capped)
    bool uniformKernel = false; // the paper's kernel (its equation 1) instead of the cone filter
    int build = 0; // the structures' build: 0 fastest to build, 1 OptiX's default, 2 fastest to trace
};

inline const char* rtpmBuildName(int build)
{
    return build == 2 ? "trace" : (build == 1 ? "default" : "fast");
}

// Consumes argv[i] (and its value) if it is one of the method's options.
inline bool rtpmParseOption(RtpmOptions& o, int argc, char** argv, int& i)
{
    const std::string a = argv[i];
    if (a.compare(0, 7, "--rtpm-") != 0)
        return false;
    auto bad = [&](const char* expects) {
        std::cerr << a << " expects " << expects << "\n";
        std::exit(2);
    };
    if (i + 1 >= argc)
        bad("a value");
    const std::string v = argv[++i];
    if (a == "--rtpm-culling") {
        if (v != "on" && v != "off")
            bad("on or off");
        o.culling = (v == "on");
    } else if (a == "--rtpm-mask-bits") {
        o.cullBits = atoi(v.c_str());
        if (o.cullBits < 10 || o.cullBits > 28)
            bad("a number of hash bits in 10..28");
    } else if (a == "--rtpm-reject") {
        o.reject = (float)atof(v.c_str());
        if (!(o.reject >= 0.f && o.reject <= 0.99f))
            bad("a probability in [0, 0.99]");
    } else if (a == "--rtpm-stochastic") {
        o.listSize = atoi(v.c_str());
        if (o.listSize < 0 || o.listSize > RTPM_MAX_LIST)
            bad("a list size in 0..3 (0 evaluates every photon)");
    } else if (a == "--rtpm-caustic-scale") {
        // No larger than 1: the culling cells are sized by the global radius.
        o.causticScale = (float)atof(v.c_str());
        if (!(o.causticScale > 0.f && o.causticScale <= 1.f))
            bad("a fraction of the global radius in (0, 1]");
    } else if (a == "--rtpm-kernel") {
        if (v != "cone" && v != "uniform")
            bad("cone or uniform");
        o.uniformKernel = (v == "uniform");
    } else if (a == "--rtpm-build") {
        if (v == "fast")
            o.build = 0;
        else if (v == "default")
            o.build = 1;
        else if (v == "trace")
            o.build = 2;
        else
            bad("fast, default or trace");
    } else if (a == "--rtpm-max-photons") {
        o.maxPhotons = atoll(v.c_str());
        if (o.maxPhotons < 0 || o.maxPhotons > (1LL << 28))
            bad("a number of photon slots, 0 for automatic");
    } else {
        std::cerr << "Unknown option: " << a << "\n";
        std::exit(2);
    }
    return true;
}
#endif // !__CUDACC__

#ifdef __CUDACC__
// The culling mask's cells.  A cell is as wide as a global photon's box, so
// the eight cells nearest to a gather point — the 2 x 2 x 2 block that
// starts at rtpmBlockCoord — hold every photon within half a cell of it, the
// box radius.  A photon is looked up in the one cell it lies in
// (photonCellCoord).
__device__ __forceinline__ int rtpmBlockCoord(float x, float origin, float cell)
{
    return (int)floorf((x - origin) / cell - 0.5f);
}

__device__ __forceinline__ unsigned int rtpmCullIndex(int ix, int iy, int iz, unsigned int bits)
{
    return hashCell(ix, iy, iz) & ((1u << bits) - 1u);
}

// photonUsable (gather.h) for a packed photon whose squared distance d2 to
// the gather point x has already been found to be within the radius: the
// same four tests in the same arithmetic.  `diff` is photon - x.
__device__ __forceinline__ bool rtpmUsable(const PackedPhoton& ph, float3 diff, float3 n,
    float planeTol, float normalTol, int maxPhotonDepth)
{
    const unsigned int nd = ph.nOct;
    // the path-length budget shared with the camera walk
    if (unpackDepth(nd) > maxPhotonDepth)
        return false;
    if (dotRn(n, unpackNormal(nd)) < normalTol)
        return false;
    if (fabsf(dotRn(diff, n)) > planeTol)
        return false;
    // the photon must have arrived on the side being shaded
    if (dotRn(n, UnpackOctahedral(ph.wiOct)) <= 1e-4f)
        return false;
    return true;
}

// A photon's weight in the density estimate, and the estimate's
// normalisation.  The cone filter 1 - d / r is the gathers' (gather.h) and
// integrates to pi r^2 / 3 over the disc.  The uniform kernel is the paper's
// equation (1): every photon in the disc counts alike, over its area pi r^2.
// It matters for Stochastic Evaluation in particular: a photon picked from
// the list stands for all that were found, which is a far better stand-in
// when they all weigh the same.
__device__ __forceinline__ float rtpmWeight(float d2, float invR, int uniformKernel)
{
    return uniformKernel ? 1.f : coneWeight(d2, invR);
}

__device__ __forceinline__ float rtpmNormalisation(float r, int uniformKernel)
{
    return (uniformKernel ? 1.f : 3.f) / (RT_PI * r * r);
}
#endif // __CUDACC__
