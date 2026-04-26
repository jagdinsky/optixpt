// params.h  (put this in ~/renderer/src/)
#pragma once
#include <optix.h>
#include <cuda_runtime.h>

struct Params
{
    uchar4 *frame_buffer;
    unsigned width;
    unsigned height;
    OptixTraversableHandle handle;
};