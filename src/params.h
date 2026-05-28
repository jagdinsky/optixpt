#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// ── Triangle ──────────────────────────────────────────────────────────────────
// Added: uv0/uv1/uv2 — per-vertex texture coordinates read from TEXCOORD_0.
// When a primitive has no UV attribute all three are zero, which is harmless
// (the device code falls back to mat.albedo/mat.emission when texObj == 0).

struct Triangle {
    float3 v0, v1, v2;   // vertices
    float3 n0, n1, n2;   // per-vertex normals
    float2 uv0, uv1, uv2; // per-vertex UVs  (NEW)
    int    mat_id;
};

// ── Material ──────────────────────────────────────────────────────────────────
// Added: base_color_tex / emissive_tex — opaque 64-bit CUDA texture object
// handles.  A value of 0 means "no texture; use the flat albedo/emission
// float3 instead."  The handles are created on the host via
// cudaCreateTextureObject() and stored here so the device can call
// tex2D<float4>() directly without any indirection.

struct Material {
    float3 albedo;                       // flat fallback color
    float3 emission;                     // flat fallback emission
    cudaTextureObject_t base_color_tex;  // 0 = no texture  (NEW)
    cudaTextureObject_t emissive_tex;    // 0 = no texture  (NEW)
};

// ── Params ────────────────────────────────────────────────────────────────────

struct Params {
    // Output buffers
    uchar4* frame_buffer;   // final 8-bit RGBA image written each frame
    float3* accum_buffer;   // HDR running sum for progressive accumulation

    unsigned width;
    unsigned height;
    OptixTraversableHandle handle;

    // Camera
    float3 cam_eye;
    float3 cam_u;   // right  (pre-scaled by half image-plane width)
    float3 cam_v;   // down   (pre-scaled by half image-plane height)
    float3 cam_w;   // forward to image-plane centre

    // Path tracing
    int samples_per_pixel;
    int max_depth;
    int frame_index;

    // Scene geometry
    Triangle* triangles;
    Material* materials;
};