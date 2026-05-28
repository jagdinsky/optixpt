#include <optix.h>
#include <cuda_runtime.h>

#include "params.h"

extern "C" __constant__ Params params;

// ─── float3 helpers ───────────────────────────────────────────────────────────

__device__ __forceinline__ float3 operator+(float3 a,float3 b){return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ __forceinline__ float3 operator-(float3 a,float3 b){return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ __forceinline__ float3 operator*(float  t,float3 v){return make_float3(t*v.x,t*v.y,t*v.z);}
__device__ __forceinline__ float3 operator*(float3 v,float  t){return make_float3(t*v.x,t*v.y,t*v.z);}
__device__ __forceinline__ float3 operator*(float3 a,float3 b){return make_float3(a.x*b.x,a.y*b.y,a.z*b.z);}
__device__ __forceinline__ float3& operator+=(float3& a,float3 b){a.x+=b.x;a.y+=b.y;a.z+=b.z;return a;}
__device__ __forceinline__ float   dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ __forceinline__ float3  cross(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ __forceinline__ float3  normalize(float3 v){float inv=1.f/sqrtf(dot(v,v));return make_float3(v.x*inv,v.y*inv,v.z*inv);}

// ─── SBT structs ──────────────────────────────────────────────────────────────

struct MissData   { float3 bg_color; };
struct RayGenData { };

// ─── PCG RNG ──────────────────────────────────────────────────────────────────

__device__ unsigned int pcg(unsigned int& s)
{
    s = s*747796405u+2891336453u;
    unsigned int w = ((s>>((s>>28u)+4u))^s)*277803737u;
    return (w>>22u)^w;
}
__device__ float randf(unsigned int& rng){ return (pcg(rng)&0xFFFFFF)/float(0x1000000); }

// ─── Cosine-weighted hemisphere sample ────────────────────────────────────────

__device__ float3 cosineSampleHemisphere(float r1,float r2)
{
    float phi=2.f*M_PIf*r1, s=sqrtf(r2);
    return make_float3(cosf(phi)*s, sinf(phi)*s, sqrtf(1.f-r2));
}

// ─── ONB ──────────────────────────────────────────────────────────────────────

__device__ void onb(const float3& n,float3& t,float3& b)
{
    t = (fabsf(n.x)>0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
    b = normalize(cross(n,t));
    t = cross(b,n);
}
__device__ float3 toWorld(float3 l,float3 n,float3 t,float3 b){ return l.x*t+l.y*b+l.z*n; }

// ─── Texture sampling ─────────────────────────────────────────────────────────
//
// sampleTex wraps tex2D<float4> with a zero-handle guard.
// cudaTextureObject_t is a 64-bit integer; 0 is the sentinel "no texture".
// When texObj == 0 the function returns the flat fallback color instead,
// so materials without textures work identically to before.
//
// tex2D() can only be called from raygen / closesthit / miss —
// i.e. from within an OptiX kernel launched via optixLaunch().  It cannot
// be called from regular __global__ CUDA kernels.  Here we call it in
// __closesthit__ch, which is the natural place: the hit knows which triangle
// (and therefore which material + UV) was struck.

__device__ __forceinline__ float3 sampleTex(cudaTextureObject_t texObj,
                                             float u, float v,
                                             float3 fallback)
{
    if (texObj == 0) return fallback;
    float4 s = tex2D<float4>(texObj, u, v);
    // tex2D returns linear float4; tiny_gltf already decoded JPEG/PNG to
    // 8-bit and we upload as UNORM (see uploadTexture in main.cpp), so the
    // driver converts to [0,1] float automatically.
    return make_float3(s.x, s.y, s.z);
}

// ─── Payload layout ───────────────────────────────────────────────────────────
//
//  Slot  Contents                 Written by
//  ────  ───────────────────────  ──────────
//  p0-2  radiance      (float3)   closesthit / miss
//  p3-5  attenuation   (float3)   closesthit / miss
//  p6    done flag      (uint)    miss  (sets to 1)
//  p7-9  hit_pos        (float3)  closesthit
//  p10-12 shading n     (float3)  closesthit
//
// Total: 13 slots — matches numPayloadValues = 13 in createModule().
//
// Textures are sampled directly in closesthit (see below) so we do NOT
// need extra payload slots for UV or mat_id — closesthit already knows
// all of that from optixGetPrimitiveIndex() / optixGetTriangleBarycentrics().

// ─── Raygen ───────────────────────────────────────────────────────────────────

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx   = optixGetLaunchIndex();
    const int   pixel = idx.y * params.width + idx.x;

    unsigned int rng = pixel*1973u + (unsigned int)params.frame_index*9277u + 4801u;

    float3 result = make_float3(0.f,0.f,0.f);

    for (int s = 0; s < params.samples_per_pixel; ++s) {
        float pu = (idx.x + randf(rng)) / params.width;
        float pv = (idx.y + randf(rng)) / params.height;

        float3 origin = params.cam_eye;
        float3 dir    = normalize(params.cam_w
                                + (2.f*pu-1.f)*params.cam_u
                                + (2.f*pv-1.f)*params.cam_v);

        float3 radiance    = make_float3(0.f,0.f,0.f);
        float3 attenuation = make_float3(1.f,1.f,1.f);
        bool   done        = false;

        for (int depth = 0; depth < params.max_depth && !done; ++depth) {
            unsigned int p0=__float_as_uint(radiance.x),
                         p1=__float_as_uint(radiance.y),
                         p2=__float_as_uint(radiance.z);
            unsigned int p3=__float_as_uint(attenuation.x),
                         p4=__float_as_uint(attenuation.y),
                         p5=__float_as_uint(attenuation.z);
            unsigned int p6=0u;
            unsigned int p7=0u,p8=0u,p9=0u;
            unsigned int p10=0u,p11=0u,p12=0u;

            optixTrace(params.handle, origin, dir, 1e-3f, 1e16f, 0.f,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                       0,1,0,
                       p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12);

            radiance    = make_float3(__uint_as_float(p0),__uint_as_float(p1),__uint_as_float(p2));
            attenuation = make_float3(__uint_as_float(p3),__uint_as_float(p4),__uint_as_float(p5));
            done        = (p6==1u);

            if (!done) {
                float3 hit_pos = make_float3(__uint_as_float(p7), __uint_as_float(p8),  __uint_as_float(p9));
                float3 n       = make_float3(__uint_as_float(p10),__uint_as_float(p11), __uint_as_float(p12));

                float3 t_vec, b_vec;
                onb(n, t_vec, b_vec);
                float3 local = cosineSampleHemisphere(randf(rng), randf(rng));
                origin = hit_pos;
                dir    = normalize(toWorld(local, n, t_vec, b_vec));

                if (depth >= 3) {
                    float q = fmaxf(attenuation.x, fmaxf(attenuation.y, attenuation.z));
                    if (randf(rng) > q) break;
                    attenuation.x /= q;
                    attenuation.y /= q;
                    attenuation.z /= q;
                }
            }
        }

        result.x += radiance.x;
        result.y += radiance.y;
        result.z += radiance.z;
    }

    result.x /= params.samples_per_pixel;
    result.y /= params.samples_per_pixel;
    result.z /= params.samples_per_pixel;

    // Progressive accumulation
    float3 accumulated;
    if (params.frame_index == 0) {
        accumulated = result;
    } else {
        float3 prev = params.accum_buffer[pixel];
        accumulated = make_float3(prev.x+result.x, prev.y+result.y, prev.z+result.z);
    }
    params.accum_buffer[pixel] = accumulated;

    float  nf   = (float)(params.frame_index+1);
    float3 mean = make_float3(accumulated.x/nf, accumulated.y/nf, accumulated.z/nf);
    mean.x = sqrtf(fminf(fmaxf(mean.x,0.f),1.f));
    mean.y = sqrtf(fminf(fmaxf(mean.y,0.f),1.f));
    mean.z = sqrtf(fminf(fmaxf(mean.z,0.f),1.f));

    params.frame_buffer[pixel] = make_uchar4(
        (unsigned char)(mean.x*255.f),
        (unsigned char)(mean.y*255.f),
        (unsigned char)(mean.z*255.f), 255u);
}

// ─── Closesthit ───────────────────────────────────────────────────────────────
//
// Key change: texture sampling now happens here.
//
// Why here and not in raygen?
//   optixGetPrimitiveIndex() and optixGetTriangleBarycentrics() are only
//   valid inside a hit/anyhit/intersection program.  Raygen has no access
//   to them.  We therefore sample the texture in closesthit where we know
//   exactly which triangle was hit and what the barycentric weights are,
//   then store the already-tinted radiance/attenuation in the payload for
//   raygen to read back.
//
// UV interpolation:
//   Barycentric coords (b1, b2) from OptiX satisfy:
//     P = (1-b1-b2)*V0 + b1*V1 + b2*V2
//   So w = 1-b1-b2 is the weight of vertex 0.  We apply the same weights
//   to the per-vertex UV coordinates stored in Triangle.uv0/uv1/uv2.
//
// Texture * factor multiplication:
//   glTF spec says the effective base color is:
//     baseColor = texture(baseColorTexture) * baseColorFactor
//   We handle this by multiplying the sampled value by mat.albedo.
//   When there is no texture, sampleTex() returns mat.albedo and
//   multiplying by it again would double-darken, so we use 1.0 as the
//   factor in that case — the conditional is folded into sampleTex().

extern "C" __global__ void __closesthit__ch()
{
    const int    triIdx = optixGetPrimitiveIndex();
    const float2 bary   = optixGetTriangleBarycentrics();
    const float  w      = 1.f - bary.x - bary.y;
    const Triangle& tri = params.triangles[triIdx];

    // Interpolated shading normal
    float3 n       = normalize(w*tri.n0 + bary.x*tri.n1 + bary.y*tri.n2);
    float3 ray_dir = optixGetWorldRayDirection();
    if (dot(n, ray_dir) > 0.f) n = make_float3(-n.x,-n.y,-n.z);

    // Interpolated UV  ← NEW
    float uv_u = w*tri.uv0.x + bary.x*tri.uv1.x + bary.y*tri.uv2.x;
    float uv_v = w*tri.uv0.y + bary.x*tri.uv1.y + bary.y*tri.uv2.y;

    // Material lookup
    const Material& mat = params.materials[tri.mat_id];

    // Sample base color texture (or use flat albedo if no texture)
    // glTF: effective = texture_sample * baseColorFactor
    // When texObj==0, sampleTex returns mat.albedo, so the *= mat.albedo
    // below would apply it twice.  We therefore pass make_float3(1,1,1) as
    // the fallback and multiply by mat.albedo unconditionally afterward.
    float3 albedo;
    if (mat.base_color_tex != 0) {
        float4 s = tex2D<float4>(mat.base_color_tex, uv_u, uv_v);
        // multiply sampled color by the factor stored in mat.albedo
        albedo = make_float3(s.x * mat.albedo.x,
                             s.y * mat.albedo.y,
                             s.z * mat.albedo.z);
    } else {
        albedo = mat.albedo;
    }

    // Sample emissive texture
    float3 emission;
    if (mat.emissive_tex != 0) {
        float4 s = tex2D<float4>(mat.emissive_tex, uv_u, uv_v);
        emission = make_float3(s.x * mat.emission.x,
                               s.y * mat.emission.y,
                               s.z * mat.emission.z);
    } else {
        emission = mat.emission;
    }

    // Read payload
    float3 radiance    = make_float3(__uint_as_float(optixGetPayload_0()),
                                     __uint_as_float(optixGetPayload_1()),
                                     __uint_as_float(optixGetPayload_2()));
    float3 attenuation = make_float3(__uint_as_float(optixGetPayload_3()),
                                     __uint_as_float(optixGetPayload_4()),
                                     __uint_as_float(optixGetPayload_5()));

    // Update payload
    radiance    = radiance + attenuation * emission;
    attenuation = attenuation * albedo;

    float3 hit_pos = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_3(__float_as_uint(attenuation.x));
    optixSetPayload_4(__float_as_uint(attenuation.y));
    optixSetPayload_5(__float_as_uint(attenuation.z));
    // p6 stays 0 — ray continues
    optixSetPayload_7(__float_as_uint(hit_pos.x));
    optixSetPayload_8(__float_as_uint(hit_pos.y));
    optixSetPayload_9(__float_as_uint(hit_pos.z));
    optixSetPayload_10(__float_as_uint(n.x));
    optixSetPayload_11(__float_as_uint(n.y));
    optixSetPayload_12(__float_as_uint(n.z));
}

// ─── Miss ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void __miss__ms()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());

    float3 radiance    = make_float3(__uint_as_float(optixGetPayload_0()),
                                     __uint_as_float(optixGetPayload_1()),
                                     __uint_as_float(optixGetPayload_2()));
    float3 attenuation = make_float3(__uint_as_float(optixGetPayload_3()),
                                     __uint_as_float(optixGetPayload_4()),
                                     __uint_as_float(optixGetPayload_5()));

    radiance.x += attenuation.x * data->bg_color.x;
    radiance.y += attenuation.y * data->bg_color.y;
    radiance.z += attenuation.z * data->bg_color.z;

    optixSetPayload_0(__float_as_uint(radiance.x));
    optixSetPayload_1(__float_as_uint(radiance.y));
    optixSetPayload_2(__float_as_uint(radiance.z));
    optixSetPayload_6(1u);
}