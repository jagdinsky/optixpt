#include <optix.h>
#include "params.h"

extern "C"
{
    __constant__ Params params;
}

struct RayGenData
{
};
struct MissData
{
    float3 bg_color;
};
struct HitData
{
    float3 color;
};

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const int pixel = idx.y * params.width + idx.x;

    float u = (idx.x + 0.5f) / params.width;
    float v = (idx.y + 0.5f) / params.height;

    float3 origin = make_float3(u - 0.5f, v - 0.5f, -1.f);
    float3 direction = make_float3(0.f, 0.f, 1.f);

    unsigned int r, g, b;
    optixTrace(params.handle,
               origin, direction,
               0.001f, 1e16f,
               0.f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               0, 1, 0,
               r, g, b);

    params.frame_buffer[pixel] = make_uchar4(r, g, b, 255);
}

extern "C" __global__ void __miss__ms()
{
    MissData *data = reinterpret_cast<MissData *>(optixGetSbtDataPointer());
    optixSetPayload_0((unsigned int)(data->bg_color.x * 255));
    optixSetPayload_1((unsigned int)(data->bg_color.y * 255));
    optixSetPayload_2((unsigned int)(data->bg_color.z * 255));
}

extern "C" __global__ void __closesthit__ch()
{
    HitData *data = reinterpret_cast<HitData *>(optixGetSbtDataPointer());
    optixSetPayload_0((unsigned int)(data->color.x * 255));
    optixSetPayload_1((unsigned int)(data->color.y * 255));
    optixSetPayload_2((unsigned int)(data->color.z * 255));
}