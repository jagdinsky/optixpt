#pragma once
#include <cuda_runtime.h>
#include <optix.h>
#include <vector_types.h>

struct Triangle {
    float3 v0, v1, v2; // vertices
    float3 n0, n1, n2; // per-vertex normals (for interpolation)
    int mat_id;
};

struct Material {
    float3 albedo;
    float3 emission;
};

struct Params {
    // ── Output buffers ────────────────────────────────────────────────────────
    uchar4* frame_buffer;
    // The final image: one uchar4 (R,G,B,A bytes 0-255) per pixel.
    // Written at the very end of raygen after averaging + gamma correction.
    // This is what gets saved to output.ppm.
    float3* accum_buffer;
    // A floating-point accumulation buffer — one float3 per pixel.
    // Exists for progressive rendering: if you render 64 samples across
    // multiple launches (e.g. 4 launches × 16 samples), you ADD each
    // launch's result here instead of overwriting. At display time you
    // divide by total samples.
    unsigned width;
    unsigned height;
    OptixTraversableHandle handle;
    // The BVH handle — this is what optixTrace uses to traverse the scene.
    // Considered as a pointer to your GPU-side acceleration structure.

    // ── Camera ────────────────────────────────────────────────────────────────
    float3 cam_eye;
    // The camera position in world space. Rays originate from here.
    float3 cam_u;
    // The camera's RIGHT vector, pre-scaled by half the image plane width.
    // When you compute a ray direction as:
    //   dir = cam_w + (2*u - 1)*cam_u + (2*v - 1)*cam_v
    // multiplying by (2*u-1) maps pixel U from [0,1] to [-1,+1].
    // cam_u's length controls the field of view horizontally.
    float3 cam_v;
    // The camera's UP vector, pre-scaled by half the image plane height.
    // Same idea as cam_u but for the vertical axis.
    float3 cam_w;
    // The vector from the camera eye TO the center of the image plane.
    // Its length is the focal distance (1.0 = image plane 1 unit away).
    // This is the "forward" direction all rays are biased toward.
    // Together cam_eye + cam_w + cam_u + cam_v fully define a pinhole camera.
    // No projection matrix needed — rays are computed analytically per pixel.

    // ── Path tracing settings ─────────────────────────────────────────────────
    int samples_per_pixel;
    int max_depth;
    int frame_index;

    // ── Scene data pointers ─────────────────────────────────────────────────
    Triangle* triangles;
    Material* materials;
};
