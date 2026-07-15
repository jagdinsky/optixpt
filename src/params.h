#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <vector_types.h>

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
    float area; // triangle area
    int tri_idx; // index into params.triangles (for normal lookup)
};

// Photon data structure
struct Photon {
    float3 pos; // world-space position
    float3 power; // Φ = power = radiance * area (for surface photons) or radiance (for volume photons)
    float3 dir; // world-space incident direction (pointing towards the surface for surface photons)
    // 12+12+12 = 36 byte -> 48 with alignment for coalesced access
    float _pad;
};

// Uniform photon grid
struct PhotonGrid {
    float3 aabb_min; // world-space AABB min
    float3 aabb_max; // world-space AABB max
    int3 dims; // number of cells in X,Y,Z
    float cell_size; // = gather_radius

    int* cell_start; // [dims.x*dims.y*dims.z] starting range in grid_photon_ids
    int* cell_count; // [dims.x*dims.y*dims.z] number of photons ячейке
    int* grid_photon_ids; // [num_stored] indices into photon_map (sorted by cell)
};

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
    float3 cam_u; // right  (pre-scaled)
    float3 cam_v; // down   (pre-scaled)
    float3 cam_w; // forward to image-plane centre

    // Path tracing
    int samples_per_pixel;
    int max_depth;
    int frame_index;

    // Scene geometry
    Triangle* triangles;
    Material* materials;

    // NEE: emissive light list (device pointer, built by host)
    EmissiveTriangle* lights;
    int num_lights;
    float total_light_area; // sum of all light areas (for uniform sampling)

    // Photon map
    Photon* photon_map; // device pointer, size of num_photons
    int num_photons; // number of photons sent
    int num_stored; // number of photons stored in the map, <= num_photons (device pointer -> single int)
    int* photon_count; // &num_stored on GPU
    float gather_radius; // radius of gathering (initial value, can be adapted)
    int photons_per_light; // how many photons per light source

    // Photon grid
    PhotonGrid grid;
    int use_grid; // 0 = brute-force, 1 = grid lookup

    // Render mode flag
    int render_mode; // 0 = path tracer, 1 = photon mapping
    int offline_frames; // total number of frames to render in offline mode

    // Photon power scaling factor (to adjust brightness)
    float photon_power_scale;
};