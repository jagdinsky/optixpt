#pragma once
#include <cuda_runtime.h>
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

// Uniform photon grid
struct PhotonGrid {
    float3 aabb_min; // world-space AABB min
    float3 aabb_max; // world-space AABB max
    int3 dims; // number of cells in X,Y,Z
    float cell_size; // = gather_radius (the *maximum* radius a gather may use)

    int* cell_start; // [dims.x*dims.y*dims.z] starting range in grid_photon_ids
    int* cell_count; // [dims.x*dims.y*dims.z] number of photons in the cell
    int* grid_photon_ids; // [num_stored] indices into photon_map (sorted by cell)
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

    // Path tracing
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
    float gather_radius; // maximum gather radius
    int adaptive_radius; // 1 = shrink the radius towards target_photons
    int target_photons; // photons the adaptive radius aims to collect
    int store_direct_photons; // 1 = also store first-hit photons (then the
                              // gather must NOT add its own direct lighting)

    // Photon rejection tolerances — these are what stop a photon stored on one
    // wall from bleeding onto the neighbouring one.
    float photon_normal_tol; // min dot(n_surface, n_photon)
    float photon_plane_tol; // max |dot(photon - x, n)| as a fraction of r

    // Photon grid
    PhotonGrid grid;
    int use_grid; // 0 = brute-force, 1 = grid lookup

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
