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
    ACCEL_HASH_GRID = 2 // cells stored in a hash table, sized by photon count
};

// Photon grid — the uniform grid and the hash grid share this layout.
//
// Both divide space into cubic cells of side cell_size and search the 3x3x3
// cells around the query point.  They differ only in how a cell is turned into
// an index into cell_start:
//   uniform grid:  index = the cell's position in a dense dims.x*dims.y*dims.z
//                  array, so every cell of the scene box has its own entry
//   hash grid:     index = hashCell(cell) & hash_mask, so the table has a fixed
//                  number of entries (buckets) and several cells can share one
//
// Entry e holds grid_photon_ids[cell_start[e] .. cell_start[e+1]).  One array
// of offsets rather than a start/count pair: for the uniform grid this is the
// dense part, which grows as r^-3 as the radius shrinks, so it is worth keeping
// to a single array.
struct PhotonGrid {
    float3 aabb_min; // origin of cell coordinates (uniform grid: also the box min)
    float3 aabb_max; // uniform grid only: box max
    int3 dims; // uniform grid only: number of cells in X,Y,Z
    float cell_size; // >= gather_radius (the *maximum* radius a gather may use)
    unsigned int hash_mask; // hash grid only: number of buckets - 1 (a power of two - 1)

    int* cell_start; // [cells + 1] (uniform grid) or [buckets + 1] (hash grid)
    int* grid_photon_ids; // [num_stored] indices into photon_map, grouped by entry
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

// Hash of a cell, for the hash grid.
//
// The first line is the spatial hash of Teschner et al. (2003): multiply each
// coordinate by a large prime and XOR the products.  The table size is a power
// of two and the bucket is taken from the low bits (hash & mask), and the low
// bits of that product are a weak mix, so the result is passed through the
// final mixing step of MurmurHash3 (fmix32), which makes every input bit affect
// every output bit.  Unsigned arithmetic throughout, so overflow wraps instead
// of being undefined, and negative coordinates hash like any others.
RT_INLINE unsigned int hashCell(int ix, int iy, int iz)
{
    unsigned int h = ((unsigned int)ix * 73856093u)
        ^ ((unsigned int)iy * 19349663u)
        ^ ((unsigned int)iz * 83492791u);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

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

    // Photon rejection tolerances — these are what stop a photon stored on one
    // wall from bleeding onto the neighbouring one.
    float photon_normal_tol; // min dot(n_surface, n_photon)
    float photon_plane_tol; // max |dot(photon - x, n)| as a fraction of r

    // Photon grid
    PhotonGrid grid;
    int accel; // AccelType the gather uses for this frame

    // The two halves of a split gather (photon mapping only).
    GatherPoint* gather_points; // [width*height], written by __raygen__camera
    float3* query_radiance; // [width*height], written by __raygen__query

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
