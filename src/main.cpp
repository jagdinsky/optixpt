#include "cuda_passes.h"
#include "params.h"
#include "rtpm.h"
#include "sppm_lepski.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Error macros
#define CUDA_CHECK(call)                                                                                       \
    do {                                                                                                       \
        cudaError_t e = (call);                                                                                \
        if (e != cudaSuccess) {                                                                                \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n"; \
            exit(1);                                                                                           \
        }                                                                                                      \
    } while (0)
#define OPTIX_CHECK(call)                                                                   \
    do {                                                                                    \
        OptixResult r = (call);                                                             \
        if (r != OPTIX_SUCCESS) {                                                           \
            std::cerr << "OptiX error " << __FILE__ << ":" << __LINE__ << " " << r << "\n"; \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)


static bool gCameraChanged = false;
static double gLastX = 0.0, gLastY = 0.0;
static bool gFirstMouse = true;
static float gYaw;
static float gPitch;
static bool gGuiMode = false;
static constexpr size_t kMaxPhotonSlots = 32'000'000;   // 32M * 52B = 1.66 GB


// Mouse callback to control camera orientation. Updates gYaw and gPitch based on mouse movement, and sets gCameraChanged to true when the camera is updated
static void mouseCallback(GLFWwindow*, double xpos, double ypos)
{
    if (gGuiMode || ImGui::GetIO().WantCaptureMouse) {
        gLastX = xpos;
        gLastY = ypos;
        gFirstMouse = false;
        return;
    }
    if (gFirstMouse) {
        gLastX = xpos;
        gLastY = ypos;
        gFirstMouse = false;
    }
    float dx = (float)(xpos - gLastX) * 0.15f;
    float dy = (float)(ypos - gLastY) * 0.15f;
    gLastX = xpos;
    gLastY = ypos;
    gYaw += dx;
    gPitch = std::max(-89.f, std::min(89.f, gPitch - dy));
    gCameraChanged = true;
}

// Recomputes the camera basis vectors (cam_u, cam_v, cam_w) based on the current gYaw and gPitch angles
static void rebuildCameraVectors(Params& p)
{
    if (p.width <= 0 || p.height <= 0) {
        std::cerr << "Invalid render resolution in rebuildCameraVectors\n";
        return;
    }

    float yr = gYaw * M_PI / 180.f;
    float pr = gPitch * M_PI / 180.f;

    // forward в glTF (Y-up, right-handed)
    float3 fwd = make_float3(cosf(pr) * cosf(yr), sinf(pr), cosf(pr) * sinf(yr));
    float len2 = fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z;
    if (len2 < 1e-20f) {
        fwd = make_float3(0.f, 0.f, 1.f);
    } else {
        float inv = 1.0f / sqrtf(len2);
        fwd = make_float3(fwd.x * inv, fwd.y * inv, fwd.z * inv);
    }

    float3 worldUp = make_float3(0.f, 1.f, 0.f);

    // right = fwd × worldUp, then normalize
    float3 right = make_float3(
        fwd.y * worldUp.z - fwd.z * worldUp.y,
        fwd.z * worldUp.x - fwd.x * worldUp.z,
        fwd.x * worldUp.y - fwd.y * worldUp.x);
    float rlen2 = right.x * right.x + right.y * right.y + right.z * right.z;
    if (rlen2 < 1e-20f) {
        right = make_float3(1.f, 0.f, 0.f);
    } else {
        float inv = 1.0f / sqrtf(rlen2);
        right = make_float3(right.x * inv, right.y * inv, right.z * inv);
    }

    // up = right × fwd
    float3 up = make_float3(
        right.y * fwd.z - right.z * fwd.y,
        right.z * fwd.x - right.x * fwd.z,
        right.x * fwd.y - right.y * fwd.x);

    float aspect = float(p.width) / float(p.height);
    float fovY = 45.f * M_PI / 180.f;
    float h = tanf(fovY * 0.5f);

    p.cam_w = fwd;
    p.cam_u = make_float3(right.x * h * aspect, right.y * h * aspect, right.z * h * aspect);
    p.cam_v = make_float3(up.x * h, up.y * h, up.z * h);

    // The angle one pixel spans: the image plane's height 2|cam_v| over its
    // distance |cam_w|, split into `height` rows — the convention
    // __raygen__lightvis projects with.  A pixel at distance t then covers a
    // patch of surface about t * pixel_angle wide.
    const float vLen = sqrtf(p.cam_v.x * p.cam_v.x + p.cam_v.y * p.cam_v.y + p.cam_v.z * p.cam_v.z);
    const float wLen = sqrtf(p.cam_w.x * p.cam_w.x + p.cam_w.y * p.cam_w.y + p.cam_w.z * p.cam_w.z);
    p.pixel_angle = 2.f * vLen / (wLen * float(p.height));
}

// Handles keyboard input to move the camera
static void handleKeys(GLFWwindow* window, Params& p, float speed)
{
    float yr = gYaw * (M_PI / 180.f), pr = gPitch * (M_PI / 180.f);
    float3 fwd = make_float3(cosf(pr) * cosf(yr), sinf(pr), cosf(pr) * sinf(yr));
    float flen = sqrtf(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z);
    fwd = make_float3(fwd.x / flen, fwd.y / flen, fwd.z / flen);
    float3 right = make_float3(-fwd.z, 0.f, fwd.x);
    float rlen = sqrtf(right.x * right.x + right.z * right.z);
    right = make_float3(right.x / rlen, 0.f, right.z / rlen);

    bool moved = false;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        p.cam_eye.x += fwd.x * speed;
        p.cam_eye.y += fwd.y * speed;
        p.cam_eye.z += fwd.z * speed;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        p.cam_eye.x -= fwd.x * speed;
        p.cam_eye.y -= fwd.y * speed;
        p.cam_eye.z -= fwd.z * speed;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        p.cam_eye.x -= right.x * speed;
        p.cam_eye.z -= right.z * speed;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        p.cam_eye.x += right.x * speed;
        p.cam_eye.z += right.z * speed;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        p.cam_eye.y += speed;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        p.cam_eye.y -= speed;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        gYaw -= 1.5f;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        gYaw += 1.5f;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        gPitch += 1.5f;
        moved = true;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        gPitch -= 1.5f;
        moved = true;
    }
    gPitch = std::max(-89.f, std::min(89.f, gPitch));
    if (moved)
        gCameraChanged = true;
}

// Renderer state data structure to hold OptiX objects, device pointers, etc
struct RendererState {
    OptixDeviceContext context = nullptr;
    OptixModule ptxModule = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixProgramGroup raygenGroup = nullptr;
    OptixProgramGroup missGroup = nullptr; // primary miss
    OptixProgramGroup shadowMissGroup = nullptr; // shadow miss
    OptixProgramGroup hitGroup = nullptr;
    OptixProgramGroup photonRaygenGroup = nullptr;
    OptixProgramGroup cameraRaygenGroup = nullptr; // gather pass 1
    // (gather pass 2, the photon lookup, is a CUDA kernel: launchQueryKernel)

    // The "trace" pipeline: every program that traces rays — the path
    // tracer, the camera and photon passes, the photon visualisation — and
    // their miss and hit programs.  OptiX gives every program of a pipeline
    // the same register count, which is why the lookups are not in it.
    OptixPipeline pipeline = nullptr;
    // Splatting's photon pass, with the same miss and hit programs: a pipeline
    // of its own, so that its code costs the storing photon pass no registers.
    OptixProgramGroup splatRaygenGroup = nullptr;
    OptixPipeline splatPipeline = nullptr;
    CUdeviceptr drg_splat = 0;
    // ... and stochastic splatting's, in a pipeline of its own as well.
    OptixProgramGroup stochsplatRaygenGroup = nullptr;
    OptixPipeline stochsplatPipeline = nullptr;
    CUdeviceptr drg_stochsplat = 0;
    // --fused: the camera walk and the lookup in one program, one per
    // structure (fusedIndex), each in a pipeline of its own.
    OptixProgramGroup fusedRaygenGroups[5] = {};
    OptixPipeline fusedPipelines[5] = {};
    CUdeviceptr drg_fused[5] = {};
    OptixShaderBindingTable sbt = {};
    OptixTraversableHandle gasHandle = 0;

    CUdeviceptr dGasOutputBuffer = 0;
    CUdeviceptr dVertices = 0;
    CUdeviceptr dTriangles = 0;
    CUdeviceptr dMaterials = 0;
    CUdeviceptr dLights = 0; // EmissiveTriangle array
    CUdeviceptr dParams = 0;

    CUdeviceptr drg_photon = 0;
    CUdeviceptr drg_camera = 0;

    // Working memory for the photon-grid build (device side, see cuda_passes.h)
    GridScratch grid;

    // Split-gather buffers, one entry per pixel
    CUdeviceptr dGatherPoints = 0;
    CUdeviceptr dQueryRadiance = 0;

    uint32_t numVertices = 0;
    uint32_t numMaterials = 0;

    std::vector<Material> hostMaterials;
    std::vector<uint32_t> sbtOffsets;

    std::vector<EmissiveTriangle> hostLights;

    // Texture tracking
    std::vector<cudaArray_t> texArrays;
    std::vector<cudaTextureObject_t> texObjects;

    // Light visualization launch group and device pointer for photon tracing only mode
    OptixProgramGroup lightvisRaygenGroup = nullptr;
    CUdeviceptr drg_lightvis = 0;

    // The levels in use and photon culling (see Params::culling).  Allocated
    // the first time a frame needs them — culling can be switched on from the
    // GUI — and kept after that.
    int* dUsedLevels = nullptr; // [MAX_LEVELS]
    int* dDepositCounts = nullptr; // [2]
    unsigned int* dCullMask = nullptr;
    unsigned int* dCullParent = nullptr; // --hier-mask
    size_t cullMaskBytes = 0;
};

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

// glTF buffer helpers to read vertex attributes and indices from the tinygltf::Model and tinygltf::Accessor structures.
// These functions compute the appropriate byte offsets and strides to access the data correctly, and return it in the desired format (float3 for positions/normals, float2 for UVs, uint32_t for indices)
static float3 gltfFloat3(const tinygltf::Model& m, const tinygltf::Accessor& acc, size_t i)
{
    const auto& bv = m.bufferViews[acc.bufferView];
    const auto& buf = m.buffers[bv.buffer];
    size_t stride = acc.ByteStride(bv) ? acc.ByteStride(bv) : sizeof(float3);
    const float* p = reinterpret_cast<const float*>(buf.data.data() + bv.byteOffset + acc.byteOffset + i * stride);
    return make_float3(p[0], p[1], p[2]);
}
static float2 gltfFloat2(const tinygltf::Model& m, const tinygltf::Accessor& acc, size_t i)
{
    const auto& bv = m.bufferViews[acc.bufferView];
    const auto& buf = m.buffers[bv.buffer];
    size_t stride = acc.ByteStride(bv) ? acc.ByteStride(bv) : sizeof(float2);
    const float* p = reinterpret_cast<const float*>(buf.data.data() + bv.byteOffset + acc.byteOffset + i * stride);
    return make_float2(p[0], p[1]);
}
static uint32_t gltfIndex(const tinygltf::Model& m, const tinygltf::Accessor& acc, size_t i)
{
    const auto& bv = m.bufferViews[acc.bufferView];
    const auto& buf = m.buffers[bv.buffer];
    const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset;
    switch (acc.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return base[i];
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return reinterpret_cast<const uint16_t*>(base)[i];
    default:
        return reinterpret_cast<const uint32_t*>(base)[i];
    }
}

// Texture upload
static cudaTextureObject_t uploadTexture(RendererState& state,
    const unsigned char* rgba8,
    int w, int h)
{
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &fmt, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, rgba8, w * 4, w * 4, h, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    state.texArrays.push_back(cuArray);
    state.texObjects.push_back(texObj);
    return texObj;
}

struct RunArgs {
    std::string sceneFile = "scene.glb";
    bool offline = false;
    bool photon = false;
    bool lightvis = false; // photon-tracing-only visualisation
    int frames = 0; // 0 = keep the built-in default
    int photonPaths = 0; // 0 = keep the built-in default
    int maxDepth = 0; // 0 = keep the built-in default
    int seed = 0; // random seed; same settings + different seed = independent run
    float skyIntensity = -1.f; // <0 = keep the built-in default
    int skyPhotons = -1; // <0 = keep the built-in default
    float ppmAlpha = -1.f; // <0 = default (2/3); 0 = disable the schedule
    // r_1 and the target r_N, both as a fraction of scene_radius.  <0 = unset.
    // Giving the *final* radius is usually what you want: the schedule fixes
    // r_N/r_1 for a given frame count, so naming r_N pins the blur you are
    // actually going to look at and lets r_1 fall out of it.
    float radius = -1.f;
    float finalRadius = -1.f;
    std::string cameraFile = "camera.txt";
    std::string outputFile = "../build/output.exr";
    // 32-bit float channels instead of half.  Half keeps 11 significant bits,
    // so two renders that differ only by float-summation order can land on
    // either side of a half rounding step and look 1e-3 apart; comparing images
    // at 1e-5 needs the full float.
    bool floatExr = false;

    // Offline resolution.  0 = keep the default; small frames are what make a
    // brute-force comparison run affordable.
    int width = 0;
    int height = 0;

    // Gather-side switches, all aimed at the acceleration-structure comparison.
    std::string accel = "grid"; // grid | hash | stochastic | rectified | brute
    bool verifyGather = false; // run every query pass twice, structure vs brute force
    int warmupFrames = 1; // frames left out of the timing summary
    std::string timingJson; // where to write the per-frame pass timings

    // Photon culling: store only photons some lookup of the frame can reach.
    bool culling = false;
    int maskBits = 24; // culling mask of 2^maskBits bits (24 = 2 MiB)
    bool verifyCulling = false; // trace every frame culled and not, compare
    float s0 = -1.f; // level-0 cell side in world units; <0 = sceneDiag * 2^-18

    // Per-pixel radius: r = s * k_pix * (the surface one pixel covers).
    std::string radiusMode = "global"; // global | footprint
    float kPix = 4.f;
    std::string debugExr; // prefix for the per-pixel debug images of the last frame
    int firstFrame = 0; // frame index of the first frame rendered (offline)
    long long rectBuckets = 0; // multi-level rectified table size; 0 = automatic
    bool selftestReservoir = false; // run the reservoir unit test and exit
    bool stats = false; // per-lookup counters, summarised every frame
    bool mlrectInTrace = false; // fill mlrect in the photon pass, not a kernel
    float tableLoad = 4.f; // buckets per entry of every hash table
    // OptiX validation: extra checks, and a synchronisation after every
    // launch, so off unless asked for — it inflates every pass time.
    bool validation = false;

    // Error against time.  At every checkpoint (a number of frames, counted
    // from 1 within the run) the accumulated image is saved and the GPU time
    // spent so far is recorded; --checkpoints without a list means every
    // power of two up to --frames.  --dump-frames saves the radiance of single
    // frames, not accumulated.
    std::vector<int> checkpoints;
    bool checkpointsPow2 = false;
    std::vector<int> dumpFrames;

    // Splatting: visible points are filed on cells at least F x r wide, and
    // the flux is gathered in K copies of the accumulators.
    float splatCellFactor = 2.f;
    int splatStripes = 1;

    // Optional: positions next to the entries of the hash grids and the
    // splatting grid, tested before a photon or gather point is loaded.
    bool inlinePositions = false;
    // Optional: a mask of parent cells, so that the photon pass rejects most
    // photons at the coarsest level in use (culling only).
    bool hierMask = false;
    // Optional, family A: the camera walk and the lookup in one program, the
    // structure built first from the previous frame's radii and levels.
    bool fused = false;

    // The adaptive LoD hash-grid SPPM's own options (--lepski-*).
    LepskiOptions lepski;
    // RTPM's own options (--rtpm-*).
    RtpmOptions rtpm;
};

// "4,8,16", "1-64" or "1-1024:16" (every 16th frame from 1), and any
// comma-separated mix of these; false if `s` is not such a list.
static bool parseFrameList(const std::string& s, std::vector<int>& out)
{
    std::vector<int> frames;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        int a = 0, b = 0, step = 1;
        char dash = 0, colon = 0;
        std::istringstream t(tok);
        if (!(t >> a) || a < 1)
            return false;
        b = a;
        if (t >> dash) {
            if (dash != '-' || !(t >> b) || b < a)
                return false;
            if (t >> colon) {
                if (colon != ':' || !(t >> step) || step < 1)
                    return false;
            }
        }
        if (t >> dash) // anything left over
            return false;
        for (int f = a; f <= b; f += step)
            frames.push_back(f);
    }
    if (frames.empty())
        return false;
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    out = frames;
    return true;
}

static RunArgs parseArgs(int argc, char** argv) {
    RunArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--offline") {
            args.offline = true;
        } else if (a == "--photon") {
            args.photon = true;
        } else if (a == "--lightvis") {
            args.lightvis = true;
        } else if (a == "--frames" && i + 1 < argc) {
            args.frames = std::max(1, atoi(argv[++i]));
        } else if (a == "--paths" && i + 1 < argc) {
            args.photonPaths = std::max(1, atoi(argv[++i]));
        } else if (a == "--depth" && i + 1 < argc) {
            args.maxDepth = std::max(1, atoi(argv[++i]));
        } else if (a == "--seed" && i + 1 < argc) {
            args.seed = atoi(argv[++i]);
        } else if (a == "--sky" && i + 1 < argc) {
            args.skyIntensity = (float)atof(argv[++i]);
        } else if (a == "--sky-photons" && i + 1 < argc) {
            args.skyPhotons = atoi(argv[++i]) ? 1 : 0;
        } else if (a == "--ppm-alpha" && i + 1 < argc) {
            // 0 turns the schedule off (fixed radius); otherwise clamped to the
            // open interval the recurrence is defined on.
            float v = (float)atof(argv[++i]);
            args.ppmAlpha = (v <= 0.f) ? 0.f : std::min(v, 0.999f);
        } else if (a == "--no-ppm") {
            args.ppmAlpha = 0.f;
        } else if (a == "--radius" && i + 1 < argc) {
            args.radius = (float)atof(argv[++i]);
        } else if (a == "--final-radius" && i + 1 < argc) {
            args.finalRadius = (float)atof(argv[++i]);
        } else if (a == "--camera" && i + 1 < argc) {
            args.cameraFile = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            args.outputFile = argv[++i];
        } else if (a == "--float-exr") {
            args.floatExr = true;
        } else if (a == "--res" && i + 2 < argc) {
            args.width = std::max(1, atoi(argv[++i]));
            args.height = std::max(1, atoi(argv[++i]));
        } else if (a == "--accel" && i + 1 < argc) {
            args.accel = argv[++i];
        } else if (a == "--brute-force") {
            args.accel = "brute"; // kept as a shorthand for --accel brute
        } else if (a == "--verify-gather") {
            args.verifyGather = true;
        } else if (a == "--warmup" && i + 1 < argc) {
            args.warmupFrames = std::max(0, atoi(argv[++i]));
        } else if (a == "--timing-json" && i + 1 < argc) {
            args.timingJson = argv[++i];
        } else if (a == "--culling" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v != "on" && v != "off") {
                std::cerr << "--culling expects on or off, not " << v << "\n";
                std::exit(2);
            }
            args.culling = (v == "on");
        } else if (a == "--mask-bits" && i + 1 < argc) {
            args.maskBits = std::max(10, std::min(31, atoi(argv[++i])));
        } else if (a == "--verify-culling") {
            args.verifyCulling = true;
        } else if (a == "--s0" && i + 1 < argc) {
            args.s0 = (float)atof(argv[++i]);
        } else if (a == "--radius-mode" && i + 1 < argc) {
            args.radiusMode = argv[++i];
            if (args.radiusMode != "global" && args.radiusMode != "footprint") {
                std::cerr << "--radius-mode expects global or footprint, not "
                          << args.radiusMode << "\n";
                std::exit(2);
            }
        } else if (a == "--k-pix" && i + 1 < argc) {
            args.kPix = std::max(1e-3f, (float)atof(argv[++i]));
        } else if (a == "--debug-exr" && i + 1 < argc) {
            args.debugExr = argv[++i];
        } else if (a == "--first-frame" && i + 1 < argc) {
            args.firstFrame = std::max(0, atoi(argv[++i]));
        } else if (a == "--rect-buckets" && i + 1 < argc) {
            args.rectBuckets = atoll(argv[++i]);
            if (args.rectBuckets < 1024 || args.rectBuckets > (1LL << 28)
                || (args.rectBuckets & (args.rectBuckets - 1)) != 0) {
                std::cerr << "--rect-buckets expects a power of two in [2^10, 2^28]\n";
                std::exit(2);
            }
        } else if (a == "--selftest-reservoir") {
            args.selftestReservoir = true;
        } else if (a == "--stats") {
            args.stats = true;
        } else if (a == "--mlrect-in-trace") {
            args.mlrectInTrace = true;
        } else if (a == "--table-load" && i + 1 < argc) {
            args.tableLoad = (float)atof(argv[++i]);
            if (!(args.tableLoad >= 0.25f && args.tableLoad <= 64.f)) {
                std::cerr << "--table-load expects buckets per entry in [0.25, 64]\n";
                std::exit(2);
            }
        } else if (a == "--validation" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v != "on" && v != "off") {
                std::cerr << "--validation expects on or off, not " << v << "\n";
                std::exit(2);
            }
            args.validation = (v == "on");
        } else if (a == "--checkpoints") {
            // The list is optional; the next argument is taken as one only if
            // it reads as one, so a scene file after the flag stays a scene.
            std::vector<int> list;
            if (i + 1 < argc && parseFrameList(argv[i + 1], list)) {
                args.checkpoints = list;
                ++i;
            } else if (i + 1 < argc && std::string(argv[i + 1]) == "pow2") {
                args.checkpointsPow2 = true;
                ++i;
            } else {
                args.checkpointsPow2 = true;
            }
        } else if (a == "--dump-frames" && i + 1 < argc) {
            if (!parseFrameList(argv[++i], args.dumpFrames)) {
                std::cerr << "--dump-frames expects a list of frames like 1,2,8 or 1-64:4\n";
                std::exit(2);
            }
        } else if (a == "--splat-cell-factor" && i + 1 < argc) {
            args.splatCellFactor = (float)atof(argv[++i]);
            if (args.splatCellFactor != 1.f && args.splatCellFactor != 2.f) {
                std::cerr << "--splat-cell-factor expects 1 or 2\n";
                std::exit(2);
            }
        } else if (a == "--inline-positions") {
            args.inlinePositions = true;
        } else if (a == "--hier-mask") {
            args.hierMask = true;
        } else if (a == "--fused") {
            args.fused = true;
        } else if (a == "--splat-stripes" && i + 1 < argc) {
            args.splatStripes = atoi(argv[++i]);
            if (args.splatStripes < 1 || args.splatStripes > 64) {
                std::cerr << "--splat-stripes expects 1..64\n";
                std::exit(2);
            }
        } else if (lepskiParseOption(args.lepski, argc, argv, i)) {
            // one of --lepski-*
        } else if (rtpmParseOption(args.rtpm, argc, argv, i)) {
            // one of --rtpm-*
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "Unknown option: " << a << "\n"
                      << "If this option was added recently, the binary is out "
                         "of date — rebuild with:\n"
                      << "    make -j$(nproc)\n";
            std::exit(2);
        } else if (!a.empty()) {
            args.sceneFile = a;
        }
    }
    return args;
}

struct CameraFileState {
    float3 eye = make_float3(0.f, 0.f, 0.f);
    float yaw = 0.f;
    float pitch = 0.f;
};

static CameraFileState loadCameraFile(const std::string& filename) {
    CameraFileState c;

    const std::string filepath = "../scenes/" + filename;

    std::ifstream in(filepath);
    if (!in) {
        std::cerr << "Failed to open camera file: " << filename << "\n";
        return c;
    }

    if (!(in >> c.eye.x >> c.eye.y >> c.eye.z >> c.yaw >> c.pitch)) {
        std::cerr << "Invalid camera file format: " << filename << "\n";
        return c;
    }

    return c;
}

// Scene loader
void loadScene(const std::string& filename, Scene& scene, RendererState& state)
{
    const std::string filepath = "../scenes/" + filename;
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;
    bool ok = (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".glb") == 0)
        ? loader.LoadBinaryFromFile(&model, &err, &warn, filepath)
        : loader.LoadASCIIFromFile(&model, &err, &warn, filepath);
    if (!warn.empty())
        std::cerr << "[tinyGLTF warn] " << warn << "\n";
    if (!ok) {
        std::cerr << "[tinyGLTF] " << err << "\n";
        exit(1);
    }

    // Upload textures
    std::vector<cudaTextureObject_t> imageTexObjs(model.images.size(), 0);
    for (size_t i = 0; i < model.images.size(); ++i) {
        const tinygltf::Image& img = model.images[i];
        if (img.width <= 0 || img.height <= 0 || img.component < 3)
            continue;
        std::vector<uint8_t> rgba8;
        if (img.component == 4) {
            rgba8.assign(img.image.begin(), img.image.end());
        } else {
            rgba8.resize(img.width * img.height * 4);
            for (int p = 0; p < img.width * img.height; ++p) {
                rgba8[p * 4 + 0] = img.image[p * 3 + 0];
                rgba8[p * 4 + 1] = img.image[p * 3 + 1];
                rgba8[p * 4 + 2] = img.image[p * 3 + 2];
                rgba8[p * 4 + 3] = 255;
            }
        }
        imageTexObjs[i] = uploadTexture(state, rgba8.data(), img.width, img.height);
        std::cout << " Uploaded image " << i << " (" << img.width << "x" << img.height << ")\n";
    }

    auto resolveTexture = [&](int texIndex) -> cudaTextureObject_t {
        if (texIndex < 0)
            return 0;
        const tinygltf::Texture& tex = model.textures[texIndex];
        if (tex.source < 0 || tex.source >= (int)imageTexObjs.size())
            return 0;
        return imageTexObjs[tex.source];
    };

    // Build materials
    for (const auto& gmat : model.materials) {
        Material m = {};
        const auto& pbr = gmat.pbrMetallicRoughness;
        if (gmat.name.find("mirror") != std::string::npos || gmat.name.find("Mirror") != std::string::npos)
            m.matType = MAT_MIRROR;
        else if (gmat.name.find("glass") != std::string::npos || gmat.name.find("Glass") != std::string::npos)
            m.matType = MAT_GLASS;
        else
            m.matType = MAT_DIFFUSE;
        m.albedo = make_float3((float)pbr.baseColorFactor[0],
            (float)pbr.baseColorFactor[1],
            (float)pbr.baseColorFactor[2]);
        m.base_color_tex = resolveTexture(pbr.baseColorTexture.index);
        m.emission = make_float3((float)gmat.emissiveFactor[0],
            (float)gmat.emissiveFactor[1],
            (float)gmat.emissiveFactor[2]);
        m.emissive_tex = resolveTexture(gmat.emissiveTexture.index);
        scene.materials.push_back(m);
    }
    if (scene.materials.empty()) {
        Material f = {};
        f.albedo = make_float3(0.6f, 0.4f, 0.2f);
        f.emission = make_float3(0, 0, 0);
        scene.materials.push_back(f);
    }

    const int maxMat = (int)scene.materials.size() - 1;

    auto flatNormal = [](float3 v0, float3 v1, float3 v2) {
        float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        float3 n = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
        float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        float inv = len > 0 ? 1.f / len : 0.f;
        return make_float3(inv * n.x, inv * n.y, inv * n.z);
    };

    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1)
                continue;
            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end())
                continue;
            const auto& posAcc = model.accessors[posIt->second];

            bool hasN = false;
            tinygltf::Accessor normAcc;
            auto normIt = prim.attributes.find("NORMAL");
            if (normIt != prim.attributes.end()) {
                normAcc = model.accessors[normIt->second];
                hasN = true;
            }

            bool hasUV = false;
            tinygltf::Accessor uvAcc;
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt != prim.attributes.end()) {
                uvAcc = model.accessors[uvIt->second];
                hasUV = true;
            }

            int matId = (prim.material >= 0) ? std::min(prim.material, maxMat) : 0;

            auto makeTri = [&](uint32_t i0, uint32_t i1, uint32_t i2) {
                Triangle tri;
                tri.v0 = gltfFloat3(model, posAcc, i0);
                tri.v1 = gltfFloat3(model, posAcc, i1);
                tri.v2 = gltfFloat3(model, posAcc, i2);
                if (hasN) {
                    tri.n0 = gltfFloat3(model, normAcc, i0);
                    tri.n1 = gltfFloat3(model, normAcc, i1);
                    tri.n2 = gltfFloat3(model, normAcc, i2);
                } else {
                    tri.n0 = tri.n1 = tri.n2 = flatNormal(tri.v0, tri.v1, tri.v2);
                }
                if (hasUV) {
                    tri.uv0 = gltfFloat2(model, uvAcc, i0);
                    tri.uv1 = gltfFloat2(model, uvAcc, i1);
                    tri.uv2 = gltfFloat2(model, uvAcc, i2);
                } else {
                    tri.uv0 = tri.uv1 = tri.uv2 = make_float2(0, 0);
                }
                tri.mat_id = matId;
                scene.triangles.push_back(tri);
            };

            if (prim.indices >= 0) {
                const auto& idxAcc = model.accessors[prim.indices];
                for (size_t t = 0; t < idxAcc.count / 3; ++t)
                    makeTri(gltfIndex(model, idxAcc, t * 3),
                        gltfIndex(model, idxAcc, t * 3 + 1),
                        gltfIndex(model, idxAcc, t * 3 + 2));
            } else {
                for (size_t t = 0; t < posAcc.count / 3; ++t)
                    makeTri((uint32_t)(t * 3), (uint32_t)(t * 3 + 1), (uint32_t)(t * 3 + 2));
            }
        }
    }
    std::cout << "Loaded " << scene.triangles.size() << " triangles, "
              << scene.materials.size() << " materials.\n";
}

// Build emissive light list
// Iterates over all triangles, checks if the material emits, and builds
// the EmissiveTriangle list.  Called after loadScene() so that both
// scene.triangles and scene.materials are fully populated.
void buildLightList(const Scene& scene, RendererState& state)
{
    state.hostLights.clear();
    for (int i = 0; i < (int)scene.triangles.size(); ++i) {
        const Triangle& tri = scene.triangles[i];
        const Material& mat = scene.materials[tri.mat_id];
        float3 em = mat.emission;
        if (em.x + em.y + em.z < 1e-5f)
            continue;

        float3 e1 = make_float3(tri.v1.x - tri.v0.x, tri.v1.y - tri.v0.y, tri.v1.z - tri.v0.z);
        float3 e2 = make_float3(tri.v2.x - tri.v0.x, tri.v2.y - tri.v0.y, tri.v2.z - tri.v0.z);
        float3 cr = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
        float area = 0.5f * sqrtf(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
        if (area < 1e-10f)
            continue;

        float inv = 1.f / (2.f * area);
        float3 gn = make_float3(cr.x * inv, cr.y * inv, cr.z * inv);

        float3 sn = make_float3(tri.n0.x + tri.n1.x + tri.n2.x,
                                tri.n0.y + tri.n1.y + tri.n2.y,
                                tri.n0.z + tri.n1.z + tri.n2.z);
        if (gn.x * sn.x + gn.y * sn.y + gn.z * sn.z < 0.f)
            gn = make_float3(-gn.x, -gn.y, -gn.z);

        EmissiveTriangle lt;
        lt.v0 = tri.v0;
        lt.v1 = tri.v1;
        lt.v2 = tri.v2;
        lt.emission = em;
        lt.area = area;
        lt.normal = gn;
        lt.tri_idx = i;
        state.hostLights.push_back(lt);
    }
    std::cout << "Light list: " << state.hostLights.size() << " emissive triangles.\n";
}

static float luminance(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// Scene bounding sphere.  It anchors the ray epsilon, gives the gather radius a
// scene-relative default, and — most importantly — is the disc the environment
// shoots its photons from.
void computeSceneBounds(const Scene& scene, Params& p)
{
    if (scene.triangles.empty()) {
        p.scene_center = make_float3(0.f, 0.f, 0.f);
        p.scene_radius = 1.f;
        p.scene_epsilon = 1e-3f;
        p.scene_aabb_min = make_float3(-1.f, -1.f, -1.f);
        p.scene_aabb_max = make_float3(1.f, 1.f, 1.f);
        return;
    }

    float3 bmin = scene.triangles[0].v0;
    float3 bmax = bmin;
    auto grow = [&](const float3& v) {
        bmin.x = std::min(bmin.x, v.x);
        bmax.x = std::max(bmax.x, v.x);
        bmin.y = std::min(bmin.y, v.y);
        bmax.y = std::max(bmax.y, v.y);
        bmin.z = std::min(bmin.z, v.z);
        bmax.z = std::max(bmax.z, v.z);
    };
    for (const auto& t : scene.triangles) {
        grow(t.v0);
        grow(t.v1);
        grow(t.v2);
    }

    p.scene_center = make_float3(0.5f * (bmin.x + bmax.x),
        0.5f * (bmin.y + bmax.y),
        0.5f * (bmin.z + bmax.z));
    // Kept, not just reduced to a sphere: the photon grid bins against this box.
    p.scene_aabb_min = bmin;
    p.scene_aabb_max = bmax;

    float dx = bmax.x - bmin.x, dy = bmax.y - bmin.y, dz = bmax.z - bmin.z;
    float diag = std::sqrt(dx * dx + dy * dy + dz * dz);

    // 1% of slack keeps sky photons starting strictly outside the geometry.
    p.scene_radius = std::max(0.5f * diag, 1e-4f) * 1.01f;
    p.scene_epsilon = std::max(1e-4f, diag * 5e-5f);

    std::cout << "Scene bounds: radius " << p.scene_radius
              << ", epsilon " << p.scene_epsilon << "\n";
}

// Photon paths are split between the emissive triangles and the environment in
// proportion to the flux each of them pushes into the scene, so neither emitter
// ends up starved of photons.
float computeSkySelectProb(const Params& p, const std::vector<EmissiveTriangle>& lights)
{
    float lightFlux = 0.f;
    for (const auto& lt : lights)
        lightFlux += luminance(lt.emission) * lt.area * RT_PI; // Φ = L·A·π

    float skyFlux = 0.f;
    if (p.emit_sky_photons && p.sky_intensity > 0.f && p.scene_radius > 0.f) {
        // Φ_env = πR² ∫ L(ω) dω, integrated numerically over the sphere with
        // cosθ stratified so the samples are uniform in solid angle.
        const int NT = 64, NP = 128;
        double sum = 0.0;
        for (int i = 0; i < NT; ++i) {
            double ct = 1.0 - 2.0 * (i + 0.5) / NT;
            double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
            for (int j = 0; j < NP; ++j) {
                double phi = 2.0 * M_PI * (j + 0.5) / NP;
                float3 d = make_float3((float)(st * std::cos(phi)),
                    (float)ct,
                    (float)(st * std::sin(phi)));
                sum += luminance(skyRadiance(d, p.sky_intensity));
            }
        }
        double meanL = sum / double(NT * NP);
        skyFlux = float(meanL * 4.0 * M_PI) * RT_PI * p.scene_radius * p.scene_radius;
    }

    if (skyFlux <= 0.f)
        return 0.f;
    if (lightFlux <= 0.f)
        return 1.f;
    // Never starve either emitter completely.
    return std::min(0.9f, std::max(0.1f, skyFlux / (skyFlux + lightFlux)));
}

// GPU upload (scene data + light list)
void uploadSceneBuffers(const Scene& scene, RendererState& state)
{
    size_t tb = scene.triangles.size() * sizeof(Triangle);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dTriangles), tb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dTriangles), scene.triangles.data(), tb, cudaMemcpyHostToDevice));

    size_t mb = scene.materials.size() * sizeof(Material);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dMaterials), mb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dMaterials), scene.materials.data(), mb, cudaMemcpyHostToDevice));

    // Build vertex array for BVH
    std::vector<float3> verts;
    verts.reserve(scene.triangles.size() * 3);
    for (const auto& t : scene.triangles) {
        verts.push_back(t.v0);
        verts.push_back(t.v1);
        verts.push_back(t.v2);
    }
    size_t vb = verts.size() * sizeof(float3);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dVertices), vb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dVertices), verts.data(), vb, cudaMemcpyHostToDevice));
    state.numVertices = (uint32_t)verts.size();

    // Upload light list
    if (!state.hostLights.empty()) {
        size_t lb = state.hostLights.size() * sizeof(EmissiveTriangle);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dLights), lb));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dLights), state.hostLights.data(), lb, cudaMemcpyHostToDevice));
    }

    state.numMaterials = (uint32_t)scene.materials.size();
    state.hostMaterials = scene.materials;
    state.sbtOffsets.reserve(scene.triangles.size());
    for (const auto& t : scene.triangles)
        state.sbtOffsets.push_back((uint32_t)t.mat_id);
}

// OptiX context.  Validation mode adds checks to every OptiX call and makes
// every optixLaunch synchronise, so it is only switched on when asked for
// (--validation on): with it, every pass time includes that overhead.
void createContext(RendererState& state, bool validation)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions opt = {};
    opt.validationMode = validation ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                    : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    opt.logCallbackFunction = [](unsigned int, const char*, const char* msg, void*) { std::cerr << "[OptiX] " << msg << "\n"; };
    opt.logCallbackLevel = 4;
    CUcontext cu;
    cuCtxGetCurrent(&cu);
    OPTIX_CHECK(optixDeviceContextCreate(cu, &opt, &state.context));
}

// BVH / GAS
void buildAccel(RendererState& state)
{
    const uint32_t MC = std::max(1u, state.numMaterials);
    std::vector<unsigned int> flags(MC, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    CUdeviceptr dSbt;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbt), state.sbtOffsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbt), state.sbtOffsets.data(), state.sbtOffsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    OptixBuildInput ti = {};
    ti.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    ti.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    ti.triangleArray.vertexStrideInBytes = sizeof(float3);
    ti.triangleArray.numVertices = state.numVertices;
    ti.triangleArray.vertexBuffers = &state.dVertices;
    ti.triangleArray.flags = flags.data();
    ti.triangleArray.numSbtRecords = MC;
    ti.triangleArray.sbtIndexOffsetBuffer = dSbt;
    ti.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    ti.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sz;
    optixAccelComputeMemoryUsage(state.context, &ao, &ti, 1, &sz);
    CUdeviceptr dTmp, dOut;
    size_t cso = sz.outputSizeInBytes;
    cudaMalloc(reinterpret_cast<void**>(&dTmp), sz.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&dOut), cso + 8);
    OptixAccelEmitDesc ed;
    ed.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    ed.result = dOut + cso;
    optixAccelBuild(state.context, 0, &ao, &ti, 1, dTmp, sz.tempSizeInBytes, dOut, sz.outputSizeInBytes, &state.gasHandle, &ed, 1);
    cudaFree(reinterpret_cast<void*>(dTmp));
    size_t cs;
    cudaMemcpy(&cs, reinterpret_cast<void*>(ed.result), sizeof(size_t), cudaMemcpyDeviceToHost);
    if (cs < sz.outputSizeInBytes) {
        cudaMalloc(reinterpret_cast<void**>(&state.dGasOutputBuffer), cs);
        optixAccelCompact(state.context, 0, state.gasHandle, state.dGasOutputBuffer, cs, &state.gasHandle);
        cudaFree(reinterpret_cast<void*>(dOut));
    } else {
        state.dGasOutputBuffer = dOut;
    }
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dSbt)));
}

// Module
void createModule(RendererState& state)
{
    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    state.pipelineCompileOptions = {};
    state.pipelineCompileOptions.usesMotionBlur = false;
    state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipelineCompileOptions.numPayloadValues = 16;
    state.pipelineCompileOptions.numAttributeValues = 2;
    state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    state.pipelineCompileOptions.usesPrimitiveTypeFlags = (unsigned int)OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    state.pipelineCompileOptions.allowOpacityMicromaps = 0;

    std::ifstream f("device.ptx", std::ios::binary);
    std::string ptx(std::istreambuf_iterator<char>(f), {});
    char log[2048];
    size_t ls = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(state.context, &mco, &state.pipelineCompileOptions,
        ptx.c_str(), ptx.size(), log, &ls, &state.ptxModule));
}

// Program groups
// 0. raygen
// 1. primary miss
// 2. shadow miss
// 3. hitgroup
// 4. photon tracing raygen
// 5. camera (gather pass 1) raygen
// 6. light visualisation raygen

void createProgramGroups(RendererState& state)
{
    OptixProgramGroupOptions pgo = {};
    char log[2048];
    size_t ls = sizeof(log);

    // Raygen
    OptixProgramGroupDesc rd = {};
    rd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rd.raygen.module = state.ptxModule;
    rd.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &rd, 1, &pgo, log, &ls, &state.raygenGroup));

    // Primary miss
    OptixProgramGroupDesc md = {};
    md.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    md.miss.module = state.ptxModule;
    md.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &md, 1, &pgo, log, &ls, &state.missGroup));

    // Shadow miss
    OptixProgramGroupDesc smd = {};
    smd.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    smd.miss.module = state.ptxModule;
    smd.miss.entryFunctionName = "__miss__shadow";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &smd, 1, &pgo, log, &ls, &state.shadowMissGroup));

    // Hitgroup
    OptixProgramGroupDesc hd = {};
    hd.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hd.hitgroup.moduleCH = state.ptxModule;
    hd.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &hd, 1, &pgo, log, &ls, &state.hitGroup));

    // Photon tracing raygen
    OptixProgramGroupDesc photonRd = {};
    photonRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    photonRd.raygen.module = state.ptxModule;
    photonRd.raygen.entryFunctionName = "__raygen__photon";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &photonRd, 1, &pgo,
        log, &ls, &state.photonRaygenGroup));

    // Gather pass 1 — camera walk
    OptixProgramGroupDesc cameraRd = {};
    cameraRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cameraRd.raygen.module = state.ptxModule;
    cameraRd.raygen.entryFunctionName = "__raygen__camera";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &cameraRd, 1, &pgo,
        log, &ls, &state.cameraRaygenGroup));

    // Light visualization raygen
    OptixProgramGroupDesc lightvisRd = {};
    lightvisRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    lightvisRd.raygen.module = state.ptxModule;
    lightvisRd.raygen.entryFunctionName = "__raygen__lightvis";
    OptixProgramGroupOptions opts = {};
    OPTIX_CHECK(optixProgramGroupCreate(
        state.context, &lightvisRd, 1, &pgo, log, &ls, &state.lightvisRaygenGroup));

    // Splatting's photon pass
    OptixProgramGroupDesc splatRd = {};
    splatRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    splatRd.raygen.module = state.ptxModule;
    splatRd.raygen.entryFunctionName = "__raygen__photon_splat";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &splatRd, 1, &pgo,
        log, &ls, &state.splatRaygenGroup));

    // Stochastic splatting's photon pass
    OptixProgramGroupDesc stochsplatRd = {};
    stochsplatRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    stochsplatRd.raygen.module = state.ptxModule;
    stochsplatRd.raygen.entryFunctionName = "__raygen__photon_stochsplat";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &stochsplatRd, 1, &pgo,
        log, &ls, &state.stochsplatRaygenGroup));

    // The fused camera-and-lookup programs
    const char* fusedNames[5] = { "__raygen__fused_uniform", "__raygen__fused_hash",
        "__raygen__fused_rect", "__raygen__fused_mlhash", "__raygen__fused_mlrect" };
    for (int i = 0; i < 5; ++i) {
        OptixProgramGroupDesc fusedRd = {};
        fusedRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        fusedRd.raygen.module = state.ptxModule;
        fusedRd.raygen.entryFunctionName = fusedNames[i];
        OPTIX_CHECK(optixProgramGroupCreate(state.context, &fusedRd, 1, &pgo,
            log, &ls, &state.fusedRaygenGroups[i]));
    }
}

// Links `groups` into a pipeline and sizes its stacks.
static OptixPipeline linkPipeline(RendererState& state, const OptixProgramGroup* groups,
    uint32_t count)
{
    OptixPipelineLinkOptions lo = {};
    lo.maxTraceDepth = 8;

    char log[2048];
    size_t ls = sizeof(log);
    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK(optixPipelineCreate(
        state.context,
        &state.pipelineCompileOptions,
        &lo,
        groups, count,
        log, &ls,
        &pipeline));
    if (ls > 1)
        std::cerr << "[OptiX Pipeline] " << log << "\n";

    OptixStackSizes ss = {};
    for (uint32_t i = 0; i < count; ++i)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(groups[i], &ss, pipeline));

    uint32_t fromTraversal, fromState, continuation;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &ss,
        lo.maxTraceDepth, // maxTraceDepth
        0, 0, // CC depth, DC depth
        &fromTraversal, &fromState, &continuation));

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        fromTraversal,
        fromState,
        continuation,
        2 // IAS + GAS depth
        ));
    return pipeline;
}

// Pipelines
void createPipeline(RendererState& state)
{
    // Every program group whose SBT record can be launched must be linked into
    // the pipeline — __raygen__lightvis was missing here, so "Photon Tracing
    // Only" was launching a record from an unlinked group.
    const OptixProgramGroup trace[] = {
        state.raygenGroup,
        state.missGroup,
        state.shadowMissGroup,
        state.hitGroup,
        state.photonRaygenGroup,
        state.cameraRaygenGroup,
        state.lightvisRaygenGroup
    };
    state.pipeline = linkPipeline(state, trace, (uint32_t)std::size(trace));

    const OptixProgramGroup splat[] = {
        state.splatRaygenGroup,
        state.missGroup,
        state.shadowMissGroup,
        state.hitGroup
    };
    state.splatPipeline = linkPipeline(state, splat, (uint32_t)std::size(splat));

    const OptixProgramGroup stochsplat[] = {
        state.stochsplatRaygenGroup,
        state.missGroup,
        state.shadowMissGroup,
        state.hitGroup
    };
    state.stochsplatPipeline = linkPipeline(state, stochsplat, (uint32_t)std::size(stochsplat));

    for (int i = 0; i < 5; ++i) {
        const OptixProgramGroup fused[] = {
            state.fusedRaygenGroups[i],
            state.missGroup,
            state.shadowMissGroup,
            state.hitGroup
        };
        state.fusedPipelines[i] = linkPipeline(state, fused, (uint32_t)std::size(fused));
    }
}

// SBT layout (miss records must match missSbtOffset used in optixTrace):
//   miss[0] = primary miss   (missSbtOffset=0 in primary optixTrace)
//   miss[1] = shadow miss    (missSbtOffset=1 in shadow  optixTrace)

template <typename T>
struct SbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
struct RayGenData { };
struct MissData {
    float3 bgColor;
};
struct HitData {
    float3 albedo;
    float3 emission;
};
inline size_t roundUp(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// SBT setup
void createSBT(RendererState& state)
{
    state.sbt = {};

    // Raygen record
    SbtRecord<RayGenData> rg;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenGroup, &rg));
    CUdeviceptr drg;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&drg), sizeof(rg)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(drg), &rg, sizeof(rg), cudaMemcpyHostToDevice));
    state.sbt.raygenRecord = drg;

    // Miss records: two entries — primary (index 0) and shadow (index 1)
    std::vector<SbtRecord<MissData>> missRecs(2);
    // primary miss: white background
    missRecs[0].data.bgColor = make_float3(1.0f, 1.0f, 1.0f); // sky intensity
    OPTIX_CHECK(optixSbtRecordPackHeader(state.missGroup, &missRecs[0]));
    // shadow miss: data unused (only sets payload 0 = 1)
    missRecs[1].data.bgColor = make_float3(0, 0, 0);
    OPTIX_CHECK(optixSbtRecordPackHeader(state.shadowMissGroup, &missRecs[1]));

    size_t missStride = roundUp(sizeof(SbtRecord<MissData>), OPTIX_SBT_RECORD_ALIGNMENT);
    CUdeviceptr dms;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dms), missStride * 2));
    // Copy record 0
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dms), &missRecs[0], sizeof(missRecs[0]), cudaMemcpyHostToDevice));
    // Copy record 1 at aligned stride offset
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dms + missStride), &missRecs[1], sizeof(missRecs[1]), cudaMemcpyHostToDevice));

    state.sbt.missRecordBase = dms;
    state.sbt.missRecordStrideInBytes = (unsigned int)missStride;
    state.sbt.missRecordCount = 2;

    // Hitgroup records (one per material)
    const uint32_t MC = std::max(1u, state.numMaterials);
    std::vector<SbtRecord<HitData>> hrs(MC);
    for (uint32_t i = 0; i < MC; ++i) {
        OPTIX_CHECK(optixSbtRecordPackHeader(state.hitGroup, &hrs[i]));
        hrs[i].data.albedo = (i < state.hostMaterials.size()) ? state.hostMaterials[i].albedo : make_float3(0.5f, 0.5f, 0.5f);
        hrs[i].data.emission = (i < state.hostMaterials.size()) ? state.hostMaterials[i].emission : make_float3(0, 0, 0);
    }
    size_t hb = hrs.size() * sizeof(SbtRecord<HitData>);
    CUdeviceptr dhr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dhr), hb));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dhr), hrs.data(), hb, cudaMemcpyHostToDevice));
    state.sbt.hitgroupRecordBase = dhr;
    state.sbt.hitgroupRecordStrideInBytes = (unsigned int)roundUp(sizeof(SbtRecord<HitData>), OPTIX_SBT_RECORD_ALIGNMENT);
    state.sbt.hitgroupRecordCount = MC;

    state.sbt.callablesRecordBase = 0;
    state.sbt.callablesRecordStrideInBytes = 0;
    state.sbt.callablesRecordCount = 0;

    // Photon raygen SBT record
    SbtRecord<RayGenData> rgPhoton = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.photonRaygenGroup, &rgPhoton));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_photon), sizeof(rgPhoton)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_photon),
        &rgPhoton, sizeof(rgPhoton), cudaMemcpyHostToDevice));

    // Gather pass 1 (camera) raygen SBT record
    SbtRecord<RayGenData> rgCamera = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.cameraRaygenGroup, &rgCamera));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_camera), sizeof(rgCamera)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_camera),
        &rgCamera, sizeof(rgCamera), cudaMemcpyHostToDevice));

    // Splatting's photon pass
    SbtRecord<RayGenData> rgSplat = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.splatRaygenGroup, &rgSplat));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_splat), sizeof(rgSplat)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_splat),
        &rgSplat, sizeof(rgSplat), cudaMemcpyHostToDevice));
    SbtRecord<RayGenData> rgStochsplat = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.stochsplatRaygenGroup, &rgStochsplat));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_stochsplat), sizeof(rgStochsplat)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_stochsplat),
        &rgStochsplat, sizeof(rgStochsplat), cudaMemcpyHostToDevice));
    for (int i = 0; i < 5; ++i) {
        SbtRecord<RayGenData> rgFused = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(state.fusedRaygenGroups[i], &rgFused));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_fused[i]), sizeof(rgFused)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_fused[i]),
            &rgFused, sizeof(rgFused), cudaMemcpyHostToDevice));
    }

    // Light visualization raygen SBT record
    SbtRecord<RayGenData> rgLightvis = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.lightvisRaygenGroup, &rgLightvis));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_lightvis), sizeof(rgLightvis)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_lightvis),
        &rgLightvis, sizeof(rgLightvis), cudaMemcpyHostToDevice));
}

// OpenGL fullscreen quad
static GLuint gQuadVAO = 0, gQuadVBO = 0, gQuadProg = 0;
static void initQuad()
{
    const char* vs = "#version 330 core\nlayout(location=0) in vec2 pos;\nout vec2 uv;\nvoid main(){ uv=pos*0.5+0.5; gl_Position=vec4(pos,0,1); }\n";
    const char* fs = "#version 330 core\nin vec2 uv; out vec4 col; uniform sampler2D tex;\nvoid main(){ col=texture(tex,uv); }\n";
    auto compile = [](GLenum t, const char* src) {
        GLuint s = glCreateShader(t);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[512];
            glGetShaderInfoLog(s, 512, nullptr, buf);
            std::cerr << "Shader: " << buf << "\n";
        }
        return s;
    };
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    gQuadProg = glCreateProgram();
    glAttachShader(gQuadProg, v);
    glAttachShader(gQuadProg, f);
    glLinkProgram(gQuadProg);
    float q[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    glGenVertexArrays(1, &gQuadVAO);
    glGenBuffers(1, &gQuadVBO);
    glBindVertexArray(gQuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gQuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(q), q, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
}
static void drawQuad(GLuint tex)
{
    glUseProgram(gQuadProg);
    glUniform1i(glGetUniformLocation(gQuadProg, "tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBindVertexArray(gQuadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

// GL window init
static GLFWwindow* initGL()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* w = glfwCreateWindow(1280, 720, "OptiX Renderer", nullptr, nullptr);
    glfwMakeContextCurrent(w);
    glfwSetCursorPosCallback(w, mouseCallback);
    glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        std::cerr << "GLEW: " << glewGetErrorString(glewErr) << "\n";
        exit(1);
    }
    return w;
}

// The probabilistic-PPM radius for frame `frameIndex` (0-based), i.e. r_i with
// i = frameIndex + 1 in  r_{i+1} = r_i * sqrt((i + alpha)/(i + 1)).
//
// Evaluated in closed form rather than by repeated multiplication.  Telescoping
// the product gives
//
//     (r_i / r_1)^2 = [Gamma(i+a)/Gamma(1+a)] * [Gamma(2)/Gamma(i+1)]
//
// which costs the same as the incremental update but does not accumulate float
// drift over thousands of frames and — the reason it matters here — is a pure
// function of frame_index.  The realtime loop resets frame_index whenever the
// camera moves, and this way the radius resets with it for free instead of
// needing its own piece of state kept in sync.
//
// Note how slowly this falls: r_i / r_1 ~ i^(-(1-alpha)/2), so at alpha = 2/3 a
// thousand frames buy a factor of 3.  The schedule guarantees the bias vanishes;
// it does not rescue a badly chosen r_1, which still has to be roughly the
// radius you want.
static float ppmRadius(float r1, float alpha, int frameIndex)
{
    const int i = frameIndex + 1;
    if (i <= 1)
        return r1;
    const double logRatio = (std::lgamma(i + (double)alpha) - std::lgamma(1.0 + (double)alpha))
        - (std::lgamma(i + 1.0) - std::lgamma(2.0));
    return r1 * (float)std::exp(0.5 * logRatio);
}

// Set gather_radius for the frame about to be traced.  Must run *before*
// launchPhotonTracing: buildPhotonGrid bins the map at this radius.
static void updatePPMRadius(Params& p)
{
    if (p.use_ppm)
        p.gather_radius = ppmRadius(p.ppm_radius_initial, p.ppm_alpha, p.frame_index);
    // The same schedule as a factor, for the per-pixel radius.
    p.radius_scale = p.use_ppm ? ppmRadius(1.f, p.ppm_alpha, p.frame_index) : 1.f;
}

// The host-side grid build this replaced.  Kept, commented, because it is the
// reference the GPU build was checked against: CUB's radix sort is stable and
// the photon ids go in ascending, so it groups a cell's photons in exactly the
// order this counting sort did.  What it cost was the thing worth removing —
// the whole photon map came back over PCIe every frame, three host passes ran
// over it, and the result went back up again, all of it per frame because PPM
// rebuilds the grid every frame.  See buildPhotonGridGPU in cuda_passes.cu.
//
// void buildPhotonGrid(Params& params,
//     CUdeviceptr& dGridCellStart,
//     CUdeviceptr& dGridCellCount,
//     CUdeviceptr& dGridPhotonIds)
// {
//     // get photon count from GPU
//     int storedCount = 0;
//     CUDA_CHECK(cudaMemcpy(&storedCount, params.photon_count,
//         sizeof(int), cudaMemcpyDeviceToHost));
//     storedCount = std::min(storedCount, params.photon_capacity);
//
//     if (storedCount == 0) {
//         params.use_grid = 0;
//         return;
//     }
//
//     std::vector<Photon> hostPhotons(storedCount);
//     CUDA_CHECK(cudaMemcpy(hostPhotons.data(), params.photon_map,
//         storedCount * sizeof(Photon),
//         cudaMemcpyDeviceToHost));
//
//     // AABB bounds of all photons
//     float3 bmin = hostPhotons[0].pos;
//     float3 bmax = hostPhotons[0].pos;
//     for (int i = 1; i < storedCount; ++i) {
//         const float3& p = hostPhotons[i].pos;
//         bmin.x = std::min(bmin.x, p.x);
//         bmax.x = std::max(bmax.x, p.x);
//         bmin.y = std::min(bmin.y, p.y);
//         bmax.y = std::max(bmax.y, p.y);
//         bmin.z = std::min(bmin.z, p.z);
//         bmax.z = std::max(bmax.z, p.z);
//     }
//     // one cell extra in each direction to avoid boundary issues
//     float r = params.gather_radius;
//     bmin.x -= r;
//     bmin.y -= r;
//     bmin.z -= r;
//     bmax.x += r;
//     bmax.y += r;
//     bmax.z += r;
//
//     // The grid is binned at `cs`, normally the gather radius itself.  The axis
//     // count is capped, and the cap has to be absorbed by *growing the cell*
//     // rather than by clamping dims alone: the host below clamps an out-of-range
//     // photon into the edge cell, while the device derives its index from
//     // grid.cell_size and skips anything outside dims, so if the two disagree the
//     // gather silently loses photons.  Growing cs instead keeps them consistent
//     // and stays correct, because the device's 3x3x3 walk covers the search
//     // sphere for any cs >= r (it is only slower, more photons per cell to test).
//     //
//     // This matters now that the PPM schedule shrinks r every frame: cell count
//     // grows as r^-3, so a factor-3 radius reduction is 27x the cells.
//     const int kMaxCellsPerAxis = 1024;
//     const float extX = bmax.x - bmin.x, extY = bmax.y - bmin.y, extZ = bmax.z - bmin.z;
//     float cs = r;
//     cs = std::max(cs, extX / kMaxCellsPerAxis);
//     cs = std::max(cs, extY / kMaxCellsPerAxis);
//     cs = std::max(cs, extZ / kMaxCellsPerAxis);
//
//     // grid dimensions (number of cells in each direction)
//     int3 dims;
//     dims.x = std::min(kMaxCellsPerAxis, std::max(1, (int)std::ceil(extX / cs)));
//     dims.y = std::min(kMaxCellsPerAxis, std::max(1, (int)std::ceil(extY / cs)));
//     dims.z = std::min(kMaxCellsPerAxis, std::max(1, (int)std::ceil(extZ / cs)));
//
//     int totalCells = dims.x * dims.y * dims.z;
//     std::cout << "[PhotonGrid] dims = " << dims.x << "x" << dims.y << "x" << dims.z
//               << "  cells = " << totalCells
//               << "  photons = " << storedCount << "\n";
//
//     // Count photons in each cell
//     auto cellIndex = [&](float3 pos) -> int {
//         int ix = (int)std::floor((pos.x - bmin.x) / cs);
//         int iy = (int)std::floor((pos.y - bmin.y) / cs);
//         int iz = (int)std::floor((pos.z - bmin.z) / cs);
//         ix = std::max(0, std::min(ix, dims.x - 1));
//         iy = std::max(0, std::min(iy, dims.y - 1));
//         iz = std::max(0, std::min(iz, dims.z - 1));
//         return iz * dims.y * dims.x + iy * dims.x + ix;
//     };
//
//     std::vector<int> cellCount(totalCells, 0);
//     for (int i = 0; i < storedCount; ++i)
//         cellCount[cellIndex(hostPhotons[i].pos)]++;
//
//     // Prefix sum -> cell_start
//     std::vector<int> cellStart(totalCells, 0);
//     for (int c = 1; c < totalCells; ++c)
//         cellStart[c] = cellStart[c - 1] + cellCount[c - 1];
//
//     // Filling grid_photon_ids
//     std::vector<int> gridPhotonIds(storedCount);
//     std::vector<int> insertCursor(cellStart); // копия для записи
//
//     for (int i = 0; i < storedCount; ++i) {
//         int c = cellIndex(hostPhotons[i].pos);
//         gridPhotonIds[insertCursor[c]++] = i;
//     }
//
//     // Upload to GPU.  This runs once per frame now, so the buffers are kept
//     // and only grown when they no longer fit.
//     // One capacity per buffer — sharing a counter between two pointers would let
//     // the second one skip a realloc it actually needed.
//     static size_t startCapacity = 0;
//     static size_t countCapacity = 0;
//     static size_t idCapacity = 0;
//
//     auto ensure = [](CUdeviceptr& ptr, size_t& capacity, size_t needed) {
//         if (ptr && capacity >= needed)
//             return;
//         if (ptr)
//             CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
//         capacity = needed + needed / 4; // a little headroom
//         CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), capacity * sizeof(int)));
//     };
//
//     ensure(dGridCellStart, startCapacity, (size_t)totalCells);
//     ensure(dGridCellCount, countCapacity, (size_t)totalCells);
//     ensure(dGridPhotonIds, idCapacity, (size_t)storedCount);
//
//     CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dGridCellStart),
//         cellStart.data(), totalCells * sizeof(int),
//         cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dGridCellCount),
//         cellCount.data(), totalCells * sizeof(int),
//         cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dGridPhotonIds),
//         gridPhotonIds.data(), storedCount * sizeof(int),
//         cudaMemcpyHostToDevice));
//
//     // Update params
//     params.grid.aabb_min = bmin;
//     params.grid.aabb_max = bmax;
//     params.grid.dims = dims;
//     params.grid.cell_size = cs;
//     params.grid.cell_start = reinterpret_cast<int*>(dGridCellStart);
//     params.grid.cell_count = reinterpret_cast<int*>(dGridCellCount);
//     params.grid.grid_photon_ids = reinterpret_cast<int*>(dGridPhotonIds);
//     params.use_grid = 1;
// }

static bool saveEXR(const std::string& filename,
                    const std::vector<float3>& hdr,
                    int width, int height, bool fullFloat = false) {
    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y; // vertical flip
        for (int x = 0; x < width; ++x) {
            int dst = y * width + x;
            int src = srcY * width + x;

            images[0][dst] = hdr[src].x; // R
            images[1][dst] = hdr[src].y; // G
            images[2][dst] = hdr[src].z; // B
        }
    }

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float*> image_ptrs = {
        images[2].data(), // B
        images[1].data(), // G
        images[0].data()  // R
    };
    image.images = reinterpret_cast<unsigned char**>(image_ptrs.data());
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    strncpy(header.channels[0].name, "B", 255);
    strncpy(header.channels[1].name, "G", 255);
    strncpy(header.channels[2].name, "R", 255);

    header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < 3; ++i) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = fullFloat ? TINYEXR_PIXELTYPE_FLOAT
                                                    : TINYEXR_PIXELTYPE_HALF;
    }

    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            std::cerr << "SaveEXR error: " << err << "\n";
            FreeEXRErrorMessage(err);
        }
        return false;
    }

    return true;
}

// Per-pass timing.
//
// The events are timestamps on the stream, so what they measure is the wall
// clock between two points of the frame — device work and any gap where the GPU
// sat waiting for the host to enqueue the next piece.  That is the honest
// number for a structure that is rebuilt every frame: the build's host side is
// part of its cost.
struct GpuTimer {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    void init()
    {
        if (!begin) {
            CUDA_CHECK(cudaEventCreate(&begin));
            CUDA_CHECK(cudaEventCreate(&end));
        }
    }
    void start(CUstream stream)
    {
        init();
        CUDA_CHECK(cudaEventRecord(begin, stream));
    }
    float stopMs(CUstream stream)
    {
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        return ms;
    }
    void destroy()
    {
        if (begin) {
            cudaEventDestroy(begin);
            cudaEventDestroy(end);
            begin = end = nullptr;
        }
    }
};

// One frame's worth of measurements, in milliseconds.
struct FrameTimings {
    float photon_trace = 0.f;
    float build = 0.f; // acceleration-structure build
    float camera = 0.f;
    float mark = 0.f; // culling only: clearing the mask and marking the cells
    float query = 0.f;
    float finalize = 0.f; // splatting: flux -> radiance (its build and photon
                          // pass are `build` and `photon_trace`)
    float fused = 0.f; // --fused: the camera walk and the lookup in one pass
    float accum = 0.f;
    long long vp_entries = 0; // splatting: (visible point, cell) entries filed
    long long vp_unique = 0; //   ... after removing a point's repeats in a bucket
    long long reached = 0; //   ... deposits that reached a visible point
    long long pairs = 0; //   ... (deposit, visible point) pairs = atomic triples
    float radius = 0.f;
    long long cells = 0; // uniform grid only
    long long buckets = 0; // hash grids only
    int photons = 0; // stored
    long long deposits = 0; // photons deposited before culling (= photons without it)
    float culled = 0.f; // fraction of the deposits culling turned away
    float r_max = 0.f; // largest lookup radius of the frame (in footprint mode
    float r_mean = 0.f; // the reduction over the gather points provides these
    float r_px_mean = 0.f; // three; in global mode r is the same everywhere)
    long long entries = 0; // multi-level grids: (photon, level) pairs filed
    long long bytes = 0; // memory of the structure the lookup reads
    int num_levels = 0; // levels in use (read back only when something needs them)
    int level_min = -1;
    int level_max = -1;
    QueryStatSummary tested; // --stats: photon records each lookup tested
    QueryStatSummary accepted; //   ... and how many it accepted
    QueryStatSummary cells_visited; //   ... and how many cells it visited
    float total() const { return photon_trace + build + camera + mark + query + finalize + fused + accum; }
};

// Selected with --culling or the GUI; params.culling is what a frame uses,
// which is this only while the photon map is retraced for the frame's own
// lookups (photon mapping, retrace on).
static bool gCulling = false;

// --rect-buckets: the multi-level rectified grid's table size (a power of
// two); 0 = sized from the photon count.
static long long gRectBuckets = 0;

// --mlrect-in-trace: the photon pass fills the multi-level rectified grid.
static bool gMlrectInTrace = false;

// --fused: camera walk and lookup in one program (offline, family A).
static bool gFused = false;

// The fused program of a structure, or -1.
static int fusedIndex(AccelType a)
{
    switch (a) {
    case ACCEL_UNIFORM_GRID:
        return 0;
    case ACCEL_HASH_GRID:
        return 1;
    case ACCEL_RECTIFIED_HASH_GRID:
        return 2;
    case ACCEL_MLHASH:
        return 3;
    case ACCEL_MLRECT:
        return 4;
    default:
        return -1;
    }
}

// What the timings were measured on, recorded with them: a Debug build or
// OptiX validation moves every pass time, and so do the GPU and its driver.
struct RunEnvironment {
    std::string buildType;
    bool validation = false;
    std::string gpu;
    std::string driver;
    int cudaDriverApi = 0;
    int cudaRuntime = 0;
};
static RunEnvironment gEnv;

#ifndef RENDERER_BUILD_TYPE
#define RENDERER_BUILD_TYPE ""
#endif

static RunEnvironment queryEnvironment(bool validation)
{
    RunEnvironment e;
    e.buildType = RENDERER_BUILD_TYPE[0] ? RENDERER_BUILD_TYPE : "unspecified";
    e.validation = validation;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        e.gpu = prop.name;
    cudaDriverGetVersion(&e.cudaDriverApi);
    cudaRuntimeGetVersion(&e.cudaRuntime);
    // The kernel module's own version string, "... Kernel Module  595.99.02  ...".
    std::ifstream in("/proc/driver/nvidia/version");
    std::string line;
    if (std::getline(in, line)) {
        const size_t k = line.find("Kernel Module");
        if (k != std::string::npos) {
            std::istringstream words(line.substr(k + 13));
            words >> e.driver;
        }
    }
    if (e.driver.empty())
        e.driver = "CUDA driver API " + std::to_string(e.cudaDriverApi);
    return e;
}

static std::string jsonString(const std::string& s)
{
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    return out + "\"";
}

// The structure the gather searches the photon map with, set by --accel or the
// GUI.  params.accel is what the device actually uses for a given frame; it is
// this value except when there are no photons to build over, and except while
// --verify-gather runs its brute-force reference.
//
// ACCEL_BRUTE_FORCE builds nothing and tests every photon in the map.  It is
// slow by construction: it is the reference the structures are checked against.
//
// The two stochastic hash grids are not exact structures but estimators: they
// keep one photon per bucket, so they do not find the same photons as brute
// force, only the same sum in expectation (the rectified one) or not even that
// (the original one — its pick is biased, which is what it is here to show).
static AccelType gAccel = ACCEL_UNIFORM_GRID;

static const char* accelName(AccelType a)
{
    switch (a) {
    case ACCEL_BRUTE_FORCE:
        return "brute";
    case ACCEL_HASH_GRID:
        return "hash";
    case ACCEL_STOCHASTIC_HASH_GRID:
        return "stochastic";
    case ACCEL_RECTIFIED_HASH_GRID:
        return "rect";
    case ACCEL_MLHASH:
        return "mlhash";
    case ACCEL_MLRECT:
        return "mlrect";
    case ACCEL_SPLAT:
        return "splat";
    case ACCEL_MLSPLAT:
        return "mlsplat";
    case ACCEL_STOCHSPLAT:
        return "stochsplat";
    case ACCEL_SPPM_LEPSKI:
        return "sppm_lepski";
    case ACCEL_RTPM:
        return "rtpm";
    default:
        return "uniform";
    }
}

static const char* accelLabel(AccelType a)
{
    switch (a) {
    case ACCEL_BRUTE_FORCE:
        return "brute force";
    case ACCEL_HASH_GRID:
        return "hash grid";
    case ACCEL_STOCHASTIC_HASH_GRID:
        return "stochastic hash grid";
    case ACCEL_RECTIFIED_HASH_GRID:
        return "rectified stochastic hash grid";
    case ACCEL_MLHASH:
        return "multi-level hash grid";
    case ACCEL_MLRECT:
        return "multi-level rectified hash grid";
    case ACCEL_SPLAT:
        return "splatting into a hash grid of visible points";
    case ACCEL_MLSPLAT:
        return "splatting into a multi-level hash grid of visible points";
    case ACCEL_STOCHSPLAT:
        return "stochastic splatting into one of the visible points reached";
    case ACCEL_SPPM_LEPSKI:
        return "adaptive LoD hash-grid SPPM";
    case ACCEL_RTPM:
        return "RTPM: photon acceleration structures, collected by ray";
    default:
        return "uniform grid";
    }
}

// The structures that keep one photon per bucket estimate the sum a lookup
// is after rather than computing it, so there is no image to compare them
// with pixel by pixel; they are checked by averaging many frames instead.
static bool estimatesOnly(AccelType a)
{
    return isStochasticHashGrid(a) || a == ACCEL_MLRECT || a == ACCEL_STOCHSPLAT;
}

// <family>/<accel>/<radius mode>, the name a configuration goes by in logs and
// reports: family A gathers from the full photon map, family B from a map
// culled to what the frame's lookups can reach, family C splats the photons
// into the visible points (culled implicitly: only photons that reach one
// count).  E.g. B/mlhash/footprint, C/mlsplat/footprint.  Family D is the
// adaptive LoD hash-grid SPPM, a method of its own: D/sppm_lepski/footprint.
// Family E is RTPM, the photons in acceleration structures: E/rtpm/global.
static const char* familyName(AccelType a, bool culling)
{
    if (a == ACCEL_SPPM_LEPSKI)
        return "D";
    if (a == ACCEL_RTPM)
        return "E";
    return isSplatting(a) ? "C" : (culling ? "B" : "A");
}

static std::string configName(AccelType a, bool culling, int radiusMode)
{
    return std::string(familyName(a, culling)) + "/" + accelName(a)
        + (radiusMode == RADIUS_FOOTPRINT ? "/footprint" : "/global");
}

static bool parseAccel(const std::string& s, AccelType& out)
{
    if (s == "grid" || s == "uniform") {
        out = ACCEL_UNIFORM_GRID;
    } else if (s == "hash") {
        out = ACCEL_HASH_GRID;
    } else if (s == "stochastic") {
        out = ACCEL_STOCHASTIC_HASH_GRID;
    } else if (s == "rect" || s == "rectified") {
        out = ACCEL_RECTIFIED_HASH_GRID;
    } else if (s == "brute") {
        out = ACCEL_BRUTE_FORCE;
    } else if (s == "mlhash") {
        out = ACCEL_MLHASH;
    } else if (s == "mlrect") {
        out = ACCEL_MLRECT;
    } else if (s == "splat") {
        out = ACCEL_SPLAT;
    } else if (s == "mlsplat") {
        out = ACCEL_MLSPLAT;
    } else if (s == "stochsplat") {
        out = ACCEL_STOCHSPLAT;
    } else if (s == "sppm_lepski" || s == "lepski") {
        out = ACCEL_SPPM_LEPSKI;
    } else if (s == "rtpm") {
        out = ACCEL_RTPM;
    } else {
        return false;
    }
    return true;
}

// One path-traced frame; returns its GPU time, timed like the photon-mapping
// passes (for --checkpoints).
float launchPathTracing(RendererState& state, Params& p, CUstream& stream) {
    static GpuTimer timer;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                                &p, sizeof(Params),
                                cudaMemcpyHostToDevice, stream));
    timer.start(stream);
    OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                            state.dParams, sizeof(Params),
                            &state.sbt, p.width, p.height, 1));
    const float ms = timer.stopMs(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return ms;
}

static bool tracePhotonsOnce(RendererState& state, Params& p, CUstream stream, int& deposits)
{
    CUDA_CHECK(cudaMemsetAsync(p.photon_count, 0, sizeof(int), stream));
    if (p.culling)
        CUDA_CHECK(cudaMemsetAsync(p.deposit_counts, 0, 2 * sizeof(int), stream));

    OptixShaderBindingTable sbtPhoton = state.sbt;
    sbtPhoton.raygenRecord = state.drg_photon;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
                               cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(state.pipeline, stream, state.dParams, sizeof(Params),
                            &sbtPhoton, (unsigned)p.num_photon_paths, 1, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemcpy(&deposits, p.photon_count, sizeof(int), cudaMemcpyDeviceToHost));
    return deposits <= p.photon_capacity;
}

// Traces one full photon pass and rebuilds the lookup grid.
// Returns the number of photons actually stored.
//
// The stochastic hash grids turn the order around: the photon pass fills them
// as it deposits, so their table is cleared *before* the trace and nothing is
// built after it.  Clearing is all their "build" is; the filing itself is part
// of photon_trace, as in Davidovič et al. (2014, table V).
int launchPhotonTracing(RendererState& state, Params& p, CUstream& stream,
    FrameTimings* timings = nullptr)
{
    static GpuTimer traceTimer, buildTimer;

    const bool stochastic = isStochasticHashGrid(gAccel);
    // --mlrect-in-trace: the multi-level rectified grid is filled by the
    // photon pass too, so it is prepared before the trace like the stochastic
    // grids and has no build after it.
    const bool rectInTrace = gAccel == ACCEL_MLRECT && gMlrectInTrace;
    p.mlrect_in_trace = rectInTrace ? 1 : 0;
    float buildMs = 0.f;
    GridBuildStats stats;

    // The largest radius any lookup of this frame uses — the camera pass ran
    // first and worked it out.  The single-level grids size their cells by it.
    const float gridRadius = p.max_gather_radius > 0.f ? p.max_gather_radius : p.gather_radius;
    if (stochastic) {
        buildTimer.start(stream);
        prepareStochasticHashGridGPU(p, gAccel, gridRadius, state.grid, stream, &stats);
        buildMs = buildTimer.stopMs(stream);
    } else if (rectInTrace) {
        buildTimer.start(stream);
        prepareMultiLevelRectGridForTraceGPU(p, gRectBuckets, state.grid, stream, &stats);
        buildMs = buildTimer.stopMs(stream);
    } else {
        // __raygen__photon fills a stochastic grid whenever p.accel names
        // one, so the first frame after switching away from one must not trace
        // with it still set.
        p.accel = gAccel;
    }

    traceTimer.start(stream);
    int deposits = 0;
    if (!tracePhotonsOnce(state, p, stream, deposits)) {
        size_t need = size_t(deposits * 1.2) + 1024;
        std::cout << "[Photons] saturated (" << deposits << " > " << p.photon_capacity
                  << "), growing to " << need << " and retracing\n";

        CUDA_CHECK(cudaFree(p.photon_map));
        p.photon_capacity = int(std::min<size_t>(need, kMaxPhotonSlots));
        CUDA_CHECK(cudaMalloc(&p.photon_map, size_t(p.photon_capacity) * sizeof(Photon)));

        // A stochastic table now holds the saturated pass's photons, and is
        // sized from the old capacity.
        if (stochastic)
            prepareStochasticHashGridGPU(p, gAccel, gridRadius, state.grid, stream, &stats);
        else if (rectInTrace)
            prepareMultiLevelRectGridForTraceGPU(p, gRectBuckets, state.grid, stream, &stats);

        if (!tracePhotonsOnce(state, p, stream, deposits))
            std::cerr << "[Photons] STILL saturated — lower num_photon_paths\n";
    }
    const float traceMs = traceTimer.stopMs(stream);

    const int stored = std::min(deposits, p.photon_capacity);

    // Culling statistics, read after the trace is timed: they cost the
    // render nothing.  Without culling every deposit is kept.
    long long allDeposits = deposits;
    if (p.culling) {
        int counts[2] = { 0, 0 };
        CUDA_CHECK(cudaMemcpy(counts, p.deposit_counts, sizeof(counts),
            cudaMemcpyDeviceToHost));
        allDeposits = counts[0];
        // The kept count is photon_count again, tallied per path instead of
        // per slot; if the two disagree, a path skipped its tally.
        if (counts[1] != deposits)
            std::cerr << "\n[Culling] kept " << counts[1] << " but stored "
                      << deposits << " — the per-path counts are off\n";
    }

    if (stochastic) {
        // already filled by the trace
    } else if (gAccel == ACCEL_BRUTE_FORCE) {
        p.accel = ACCEL_BRUTE_FORCE;
    } else if (gAccel == ACCEL_HASH_GRID) {
        buildTimer.start(stream);
        buildPhotonHashGridGPU(p, stored, gridRadius, state.grid, stream, &stats);
        buildMs = buildTimer.stopMs(stream);
    } else if (gAccel == ACCEL_MLHASH) {
        buildTimer.start(stream);
        buildMultiLevelHashGridGPU(p, stored, state.grid, stream, &stats);
        buildMs = buildTimer.stopMs(stream);
    } else if (rectInTrace) {
        // already filled by the trace; the insertions are counted only for
        // the statistics
        if (timings)
            stats.entries = p.culling ? countMultiLevelRectEntries(p, state.grid, stream)
                                      : (long long)stored * p.num_levels;
    } else if (gAccel == ACCEL_MLRECT) {
        buildTimer.start(stream);
        buildMultiLevelRectGridGPU(p, stored, gRectBuckets, state.grid, stream, &stats);
        buildMs = buildTimer.stopMs(stream);
        // With culling the build counts its insertions to size the table,
        // unless --rect-buckets fixed the size; then they are counted here,
        // after the build is timed: it is statistics.
        if (p.culling && p.accel == ACCEL_MLRECT && timings && stats.entries == 0)
            stats.entries = countMultiLevelRectEntries(p, state.grid, stream);
    } else {
        buildTimer.start(stream);
        buildPhotonGridGPU(p, stored, gridRadius, state.grid, stream, &stats);
        buildMs = buildTimer.stopMs(stream);

        // The cell budget is a silent change of what is being measured, so say
        // when it engages — but only when it changes, or a long render would
        // print it once per frame.
        static bool wasCapped = false;
        if (stats.cell_size_capped != wasCapped) {
            wasCapped = stats.cell_size_capped;
            if (wasCapped)
                std::cout << "\n[PhotonGrid] cell budget reached: cell size "
                          << stats.cell_size << " > r = " << stats.radius
                          << " (x" << (stats.cell_size / std::max(stats.radius, 1e-9f))
                          << ") — the gather now walks wider cells\n";
            else
                std::cout << "\n[PhotonGrid] cell size back at the gather radius\n";
        }
    }

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
                          cudaMemcpyHostToDevice));

    if (timings) {
        timings->photon_trace = traceMs;
        timings->build = buildMs;
        timings->radius = gridRadius;
        timings->cells = stats.cells;
        timings->buckets = stats.buckets;
        timings->photons = stored;
        timings->deposits = allDeposits;
        timings->entries = stats.entries;
        timings->bytes = stats.bytes;
        timings->culled = allDeposits > 0
            ? float(1.0 - double(deposits) / double(allDeposits)) : 0.f;
    }
    return stored;
}

// void launchPhotonTracing(RendererState& state, Params& p, CUstream& stream) {
//     CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(p.photon_count),
//                               0, sizeof(int), stream));

//     OptixShaderBindingTable sbtPhoton = state.sbt;
//     sbtPhoton.raygenRecord = state.drg_photon;

//     CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
//                               &p, sizeof(Params),
//                               cudaMemcpyHostToDevice, stream));
//     OPTIX_CHECK(optixLaunch(state.pipeline, stream,
//                             state.dParams, sizeof(Params),
//                             &sbtPhoton, p.num_photon_paths, 1, 1));

//     CUDA_CHECK(cudaStreamSynchronize(stream));     
//     int deposits = 0;
//     CUDA_CHECK(cudaMemcpy(&deposits, p.photon_count, sizeof(int), cudaMemcpyDeviceToHost));
//     std::cout << "[Photons] " << p.num_photon_paths << " paths -> " << deposits
//               << " deposits (" << double(deposits) / p.num_photon_paths
//               << " per path)\n";

//     buildPhotonGrid(p,
//                     state.dGridCellStart,
//                     state.dGridCellCount,
//                     state.dGridPhotonIds);

//     CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dParams),
//                           &p,
//                           sizeof(Params),
//                           cudaMemcpyHostToDevice));

//     CUDA_CHECK(cudaStreamSynchronize(stream));
// }

// One raygen launch over the whole image, with `record` as the entry point.
static float launchPixelPass(RendererState& state, Params& p, CUstream& stream,
    CUdeviceptr record, GpuTimer& timer)
{
    OptixShaderBindingTable sbtPass = state.sbt;
    sbtPass.raygenRecord = record;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                              &p, sizeof(Params),
                              cudaMemcpyHostToDevice, stream));
    timer.start(stream);
    OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                            state.dParams, sizeof(Params),
                            &sbtPass, p.width, p.height, 1));
    const float ms = timer.stopMs(stream);
    return ms;
}

// The device buffers culling needs, sized for p.cull_mask_bits.
static void ensureCullingBuffers(RendererState& state, Params& p)
{
    if (!state.dUsedLevels) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dUsedLevels),
            MAX_LEVELS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dDepositCounts),
            2 * sizeof(int)));
    }
    const size_t maskBytes = (size_t(1) << p.cull_mask_bits) / 8;
    if (state.cullMaskBytes != maskBytes) {
        if (state.dCullMask)
            CUDA_CHECK(cudaFree(state.dCullMask));
        if (state.dCullParent)
            CUDA_CHECK(cudaFree(state.dCullParent));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dCullMask), maskBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dCullParent), maskBytes));
        state.cullMaskBytes = maskBytes;
    }
    p.cull_mask = state.dCullMask;
    p.cull_parent = state.dCullParent;
    p.deposit_counts = state.dDepositCounts;
}

// Gather pass 1: camera walk -> one GatherPoint per pixel.
//
// With culling or a multi-level grid the pass also flags the levels its
// lookups use, which are read back here (32 ints) and handed to the photon
// pass and the build as a list; the cells to keep are marked afterwards, by
// launchMarkPass.  In footprint mode every pixel has its own radius, and
// the largest of them — what a single-level grid has to size its cells by —
// is reduced on the device and read back here.  What a structure needs before
// it can be built is part of its cost, so it is timed with the pass; the same
// reduction run only for the statistics (mean radius, radius in pixels) is
// not.  In global mode without culling none of this runs.
float launchCameraPass(RendererState& state, Params& p, CUstream& stream,
    FrameTimings* timings = nullptr)
{
    static GpuTimer timer, outerTimer;
    // Photon mapping is one camera sample per frame; see Params::samples_per_pixel.
    p.samples_per_pixel = 1;

    // In global mode every lookup uses gather_radius.
    p.max_gather_radius = p.gather_radius;
    if (timings)
        timings->r_max = timings->r_mean = p.gather_radius;

    // mlsplat files its points on levels of its own (the build finds them),
    // but its exact counterpart in --verify-gather, mlhash, needs these.
    const bool needLevels = p.culling != 0 || isMultiLevelGrid(gAccel) || gAccel == ACCEL_MLSPLAT
        || gFused;
    const bool footprint = p.radius_mode == RADIUS_FOOTPRINT;
    // Only a single-level grid needs the largest radius before it can be
    // built — single-level splatting too, whose cells are F times it.
    // Everyone else gets the radii reduced for the statistics only (always in
    // footprint mode, with --stats in global mode too), after the pass's time
    // is taken.
    // RTPM too: its photon boxes are as wide as the largest radius.
    const bool needMaxRadius = footprint
        && (gAccel == ACCEL_UNIFORM_GRID || gAccel == ACCEL_HASH_GRID
            || isStochasticHashGrid(gAccel) || gAccel == ACCEL_SPLAT || gFused
            || gAccel == ACCEL_RTPM);
    const bool wantRadii = footprint || p.stats;

    GatherRadiusStats radii;
    bool haveRadii = false;
    float ms = 0.f;
    if (!needLevels && !needMaxRadius) {
        p.used_levels = nullptr;
        ms = launchPixelPass(state, p, stream, state.drg_camera, timer);
    } else {
        if (needLevels) {
            ensureCullingBuffers(state, p);
            p.used_levels = state.dUsedLevels;
        } else {
            p.used_levels = nullptr;
        }

        outerTimer.start(stream);
        if (needLevels)
            CUDA_CHECK(cudaMemsetAsync(p.used_levels, 0, MAX_LEVELS * sizeof(int), stream));
        launchPixelPass(state, p, stream, state.drg_camera, timer);
        int used[MAX_LEVELS] = {};
        if (needLevels)
            CUDA_CHECK(cudaMemcpyAsync(used, p.used_levels, sizeof(used),
                cudaMemcpyDeviceToHost, stream));
        if (needMaxRadius) {
            reduceGatherRadii(p, state.grid, stream, radii);
            haveRadii = true;
        }
        ms = outerTimer.stopMs(stream);

        if (needLevels) {
            p.num_levels = 0;
            for (int l = 0; l < MAX_LEVELS; ++l)
                if (used[l])
                    p.levels[p.num_levels++] = l;
            if (timings) {
                timings->num_levels = p.num_levels;
                timings->level_min = p.num_levels ? p.levels[0] : -1;
                timings->level_max = p.num_levels ? p.levels[p.num_levels - 1] : -1;
            }
        }
    }

    if (wantRadii && !haveRadii) {
        reduceGatherRadii(p, state.grid, stream, radii);
        haveRadii = true;
    }
    if (haveRadii) {
        // No valid gather point, no lookup: any cell size will do.
        if (footprint && radii.max_r > 0.f)
            p.max_gather_radius = radii.max_r;
        if (timings) {
            timings->r_max = radii.max_r;
            timings->r_mean = radii.valid ? float(radii.sum_r / double(radii.valid)) : 0.f;
            timings->r_px_mean = radii.valid ? float(radii.sum_px / double(radii.valid)) : 0.f;
        }
    }
    return ms;
}

// Photon culling, the lookup side: clear the mask and mark every cell this
// frame's lookups will search.  It runs after the camera pass because the
// marks come from the gather points that pass wrote; see
// markLookupCellsKernel.  Its time is culling's own cost on the lookup side,
// which is why it is a pass of its own and not part of the camera walk.
float launchMarkPass(RendererState& state, Params& p, CUstream& stream)
{
    static GpuTimer timer;
    timer.start(stream);
    CUDA_CHECK(cudaMemsetAsync(p.cull_mask, 0, state.cullMaskBytes, stream));
    if (p.hier_mask)
        CUDA_CHECK(cudaMemsetAsync(p.cull_parent, 0, state.cullMaskBytes, stream));
    launchMarkKernel(p, stream);
    return timer.stopMs(stream);
}

// Gather pass 2: the photon lookup -> one radiance per pixel.  A CUDA kernel
// per structure (launchQueryKernel), timed like the OptiX passes.
float launchQueryPass(RendererState&, Params& p, CUstream& stream)
{
    static GpuTimer timer;
    timer.start(stream);
    launchQueryKernel(p, stream);
    return timer.stopMs(stream);
}

// The splatting family's frame after the camera pass: file the visible points
// (timed as the build), splat every photon deposit into them (the photon
// pass: clearing the accumulators, uploading the parameters and the launch,
// as the storing photon pass is timed with its counter reset and upload), and
// turn their flux into radiance (finalize).  Nothing is stored, so nothing
// waits on a photon count.
static void launchSplatPasses(RendererState& state, Params& p, CUstream& stream, FrameTimings* ft)
{
    static GpuTimer buildTimer, splatTimer, finalizeTimer;

    // splat: one cell size, F times the frame's largest radius (the camera
    // pass reduced it); mlsplat: a level per point, from its own radius.
    p.vpgrid.cell_size = (gAccel == ACCEL_SPLAT) ? p.splat_cell_factor * p.max_gather_radius : 0.f;
    const bool stochastic = gAccel == ACCEL_STOCHSPLAT;
    VpGridStats vs;
    buildTimer.start(stream);
    buildVisiblePointGridGPU(p, state.grid, stream, &vs);
    const float buildMs = buildTimer.stopMs(stream);

    const size_t points = size_t(p.width) * p.height;
    OptixShaderBindingTable sbt = state.sbt;
    sbt.raygenRecord = stochastic ? state.drg_stochsplat : state.drg_splat;
    splatTimer.start(stream);
    CUDA_CHECK(cudaMemsetAsync(p.splat_acc, 0, size_t(p.splat_stripes) * points * 3 * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(p.splat_counts, 0, 3 * sizeof(unsigned long long), stream));
    if (p.stats)
        CUDA_CHECK(cudaMemsetAsync(p.stat_slot, 0, sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
        cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(stochastic ? state.stochsplatPipeline : state.splatPipeline, stream,
        state.dParams, sizeof(Params), &sbt, (unsigned)p.num_photon_paths, 1, 1));
    const float splatMs = splatTimer.stopMs(stream);

    finalizeTimer.start(stream);
    launchSplatFinalize(p, stream);
    const float finalizeMs = finalizeTimer.stopMs(stream);

    if (ft) {
        unsigned long long counts[3] = { 0, 0, 0 };
        CUDA_CHECK(cudaMemcpy(counts, p.splat_counts, sizeof(counts), cudaMemcpyDeviceToHost));
        ft->build = buildMs;
        ft->photon_trace = splatMs;
        ft->finalize = finalizeMs;
        ft->radius = p.max_gather_radius;
        ft->photons = 0; // nothing is stored
        ft->deposits = (long long)counts[0];
        ft->reached = (long long)counts[1];
        ft->pairs = (long long)counts[2];
        ft->vp_entries = vs.entries;
        ft->vp_unique = vs.unique;
        ft->entries = vs.unique;
        ft->buckets = vs.buckets;
        ft->bytes = vs.bytes;
        ft->num_levels = vs.num_levels;
        ft->level_min = vs.level_min;
        ft->level_max = vs.level_max;
    }
}

// What a --fused frame sizes its structure from: the frame before's largest
// radius and PPM factor, and the levels its lookups used.
struct FusedHistory {
    bool valid = false;
    float max_r = 0.f;
    float scale = 1.f;
    int num_levels = 0;
    int levels[MAX_LEVELS] = {};
};

static void rememberForFused(const Params& p, FusedHistory& hist)
{
    hist.valid = true;
    hist.max_r = p.max_gather_radius;
    hist.scale = p.radius_scale;
    hist.num_levels = p.num_levels;
    for (int i = 0; i < MAX_LEVELS; ++i)
        hist.levels[i] = p.levels[i];
}

// One --fused frame (family A): the structure first, sized without this
// frame's gather points — the single-level grids' cells from the frame
// before's largest radius times this frame's PPM shrink s_i / s_{i-1} (in
// global mode the radius is known exactly), the multi-level grids' levels
// from the ones the frame before used, with one more on either side — then the
// camera walk and the lookup in one program (__raygen__fused_*), which
// records this frame's largest radius and levels for the next.  A lookup
// whose cells turn out narrower than its radius, or whose level was not
// built, still finds exactly its photons (see fusedCameraGather).  The
// fused pass is timed with the readback of what it recorded.
static void launchFusedFrame(RendererState& state, Params& p, CUstream& stream, FrameTimings* ft,
    FusedHistory& hist)
{
    static GpuTimer timer;
    p.culling = 0;
    p.max_gather_radius = (p.radius_mode == RADIUS_FOOTPRINT)
        ? hist.max_r * (p.radius_scale / hist.scale) : p.gather_radius;
    int flags[MAX_LEVELS] = {};
    for (int i = 0; i < hist.num_levels; ++i)
        for (int d = -1; d <= 1; ++d) {
            const int l = hist.levels[i] + d;
            if (l >= 0 && l < MAX_LEVELS)
                flags[l] = 1;
        }
    p.num_levels = 0;
    for (int l = 0; l < MAX_LEVELS; ++l)
        if (flags[l])
            p.levels[p.num_levels++] = l;

    launchPhotonTracing(state, p, stream, ft);

    ensureCullingBuffers(state, p);
    p.used_levels = state.dUsedLevels;
    const int k = fusedIndex(gAccel);
    OptixShaderBindingTable sbt = state.sbt;
    sbt.raygenRecord = state.drg_fused[k];
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
        cudaMemcpyHostToDevice, stream));
    int used[MAX_LEVELS] = {};
    unsigned int maxBits = 0;
    timer.start(stream);
    CUDA_CHECK(cudaMemsetAsync(p.used_levels, 0, MAX_LEVELS * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(p.fused_max_r, 0, sizeof(unsigned int), stream));
    OPTIX_CHECK(optixLaunch(state.fusedPipelines[k], stream, state.dParams, sizeof(Params), &sbt,
        p.width, p.height, 1));
    CUDA_CHECK(cudaMemcpyAsync(used, p.used_levels, sizeof(used), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&maxBits, p.fused_max_r, sizeof(maxBits), cudaMemcpyDeviceToHost, stream));
    const float ms = timer.stopMs(stream);

    float maxR = 0.f;
    std::memcpy(&maxR, &maxBits, sizeof(maxR));
    hist.max_r = maxR > 0.f ? maxR : p.max_gather_radius;
    hist.scale = p.radius_scale;
    hist.num_levels = 0;
    for (int l = 0; l < MAX_LEVELS; ++l)
        if (used[l])
            hist.levels[hist.num_levels++] = l;
    if (ft) {
        ft->fused = ms;
        ft->num_levels = p.num_levels;
        ft->level_min = p.num_levels ? p.levels[0] : -1;
        ft->level_max = p.num_levels ? p.levels[p.num_levels - 1] : -1;
        ft->r_max = maxR;
    }
}

// Fold the query result into the accumulation buffer and tone-map it.
float launchAccumulatePass(Params& p, CUstream& stream)
{
    static GpuTimer timer;
    timer.start(stream);
    launchAccumulate(p, stream);
    return timer.stopMs(stream);
}

// Splats the photon map onto the film and tone-maps the result.
//
// The buffer accumulates across frames (it is only cleared on frame 0) so the
// visualisation converges the same way the other two modes do — every pass
// traces a fresh, independent photon map, and averaging them is what removes
// the speckle.
void launchLightVis(RendererState& state, Params& p, CUstream stream,
                    CUdeviceptr dLightvisBuf)
{
    const size_t pixels = size_t(p.width) * size_t(p.height);

    if (p.frame_index == 0) {
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(dLightvisBuf), 0,
                                   pixels * sizeof(float3), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    p.lightvis_buffer = reinterpret_cast<float*>(dLightvisBuf);

    // How many photons are stored
    int storedCount = 0;
    CUDA_CHECK(cudaMemcpy(&storedCount, p.photon_count,
                           sizeof(int), cudaMemcpyDeviceToHost));
    storedCount = std::min(storedCount, p.photon_capacity);
    if (storedCount == 0)
        return;

    // Upload updated params (with new cam_eye, lightvis_buffer)
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dParams),
                           &p, sizeof(Params), cudaMemcpyHostToDevice));

    // OptiX launch: 1-D, storedCount threads
    OptixShaderBindingTable sbtLV = state.sbt;
    sbtLV.raygenRecord = state.drg_lightvis;

    OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                             state.dParams, sizeof(Params),
                             &sbtLV,
                             (unsigned)storedCount, 1, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Average over the passes so far, apply exposure, gamma 2.0.
    //
    // The old code divided by the brightest pixel in the image and then
    // multiplied by photon_power_scale (15), which drove everything above 1/15
    // of the maximum to pure white and made the visualisation independent of
    // the actual photon intensities.  The kernel now emits real radiance, so it
    // is tone-mapped exactly like the path-traced and photon-mapped images.
    std::vector<float3> hostBuf(pixels);
    CUDA_CHECK(cudaMemcpy(hostBuf.data(),
                           reinterpret_cast<void*>(dLightvisBuf),
                           pixels * sizeof(float3),
                           cudaMemcpyDeviceToHost));

    const float scale = p.lightvis_exposure / float(p.frame_index + 1);

    std::vector<uchar4> hostFrame(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        float r = std::sqrt(std::min(std::max(hostBuf[i].x * scale, 0.f), 1.f));
        float g = std::sqrt(std::min(std::max(hostBuf[i].y * scale, 0.f), 1.f));
        float b = std::sqrt(std::min(std::max(hostBuf[i].z * scale, 0.f), 1.f));
        hostFrame[i] = make_uchar4(
            (unsigned char)(r * 255.f),
            (unsigned char)(g * 255.f),
            (unsigned char)(b * 255.f),
            255u);
    }

    CUDA_CHECK(cudaMemcpy(p.frame_buffer,
                           hostFrame.data(),
                           pixels * sizeof(uchar4),
                           cudaMemcpyHostToDevice));
}

// Per-frame timings, averaged over everything past the warm-up.
//
// The first frames are dropped because they are not the steady state being
// compared: frame 0 pays for every first-touch allocation in the build, the
// driver's first launch of each program, and the clocks still ramping up.  It
// is also why exec.py's whole-process time is the wrong number to quote — that
// one includes glTF parsing, the acceleration build and the PTX compile.
static void reportTimings(const std::vector<FrameTimings>& timings,
    const RunArgs& args, const Params& p)
{
    const int n = (int)timings.size();
    const int skip = std::min(args.warmupFrames, std::max(0, n - 1));
    if (n - skip <= 0)
        return;

    FrameTimings sum;
    long long cells = 0;
    long long buckets = 0;
    long long photons = 0;
    long long deposits = 0;
    double rMax = 0.0, rMean = 0.0, rPx = 0.0;
    double entries = 0.0, bytes = 0.0, levels = 0.0;
    int levelMin = MAX_LEVELS, levelMax = -1;
    for (int i = skip; i < n; ++i) {
        sum.photon_trace += timings[i].photon_trace;
        sum.build += timings[i].build;
        sum.camera += timings[i].camera;
        sum.mark += timings[i].mark;
        sum.query += timings[i].query;
        sum.finalize += timings[i].finalize;
        sum.fused += timings[i].fused;
        sum.accum += timings[i].accum;
        sum.vp_entries += timings[i].vp_entries;
        sum.vp_unique += timings[i].vp_unique;
        sum.reached += timings[i].reached;
        sum.pairs += timings[i].pairs;
        cells += timings[i].cells;
        buckets += timings[i].buckets;
        photons += timings[i].photons;
        deposits += timings[i].deposits;
        rMax += timings[i].r_max;
        rMean += timings[i].r_mean;
        rPx += timings[i].r_px_mean;
        entries += double(timings[i].entries);
        bytes += double(timings[i].bytes);
        levels += timings[i].num_levels;
        if (timings[i].level_min >= 0)
            levelMin = std::min(levelMin, timings[i].level_min);
        levelMax = std::max(levelMax, timings[i].level_max);
    }
    const float inv = 1.f / float(n - skip);
    const double invd = 1.0 / double(n - skip);

    std::cout << "[Timing] " << (n - skip) << " frames"
              << " (" << skip << " warm-up skipped)"
              << ", " << p.width << "x" << p.height
              << ", " << configName(gAccel, gCulling, p.radius_mode)
              << " (" << accelLabel(gAccel) << ")"
              << "\n         mean ms/frame:"
              << "\n           camera pass   " << sum.camera * inv;
    if (isSplatting(gAccel)) {
        std::cout << "\n           point grid    " << sum.build * inv
                  << "\n           splat photons " << sum.photon_trace * inv
                  << "\n           finalize      " << sum.finalize * inv;
    } else {
        if (gCulling)
            std::cout << "\n           mark pass     " << sum.mark * inv;
        std::cout << "\n           photon trace  " << sum.photon_trace * inv
                  << "\n           build         " << sum.build * inv
                  << "\n           query pass    " << sum.query * inv;
        if (sum.fused > 0.f)
            std::cout << "\n           fused pass    " << sum.fused * inv
                      << "  (camera walk and lookup, after the first frame)";
    }
    std::cout << "\n           accumulate    " << sum.accum * inv
              << "\n           ------------------------"
              << "\n           total         " << sum.total() * inv
              << "\n         mean photons/frame " << double(photons) * invd;
    if (gAccel == ACCEL_UNIFORM_GRID)
        std::cout << ", mean cells " << double(cells) * invd;
    else if (gAccel != ACCEL_BRUTE_FORCE)
        std::cout << ", mean buckets " << double(buckets) * invd;
    std::cout << "\n";
    if (isSplatting(gAccel))
        std::cout << "         splatting: mean deposits/frame " << double(deposits) * invd
                  << ", " << (deposits > 0 ? 100.0 * double(sum.reached) / double(deposits) : 0.0)
                  << "% reaching a visible point, " << double(sum.pairs) * invd << " pairs"
                  << "; point entries " << double(sum.vp_entries) * invd << " ("
                  << double(sum.vp_unique) * invd << " after de-duplication), culling implicit\n";
    if (gCulling && !isSplatting(gAccel))
        std::cout << "         culling: mean deposits/frame " << double(deposits) * invd
                  << ", culled " << (deposits > 0 ? 100.0 * (1.0 - double(photons) / double(deposits)) : 0.0)
                  << "%\n";
    if (bytes > 0.0)
        std::cout << "         structure: mean " << bytes * invd / (1024.0 * 1024.0) << " MiB"
                  << (entries > 0.0 ? ", mean entries " : "")
                  << (entries > 0.0 ? std::to_string(entries * invd) : std::string())
                  << "\n";
    if (levelMax >= 0)
        std::cout << "         levels: mean " << levels * invd << " in use, levels "
                  << levelMin << ".." << levelMax << "\n";
    if (p.stats) {
        // Means of the per-frame mean, 95th percentile and maximum.
        double tm = 0, tp = 0, tx = 0, am = 0, ap = 0, ax = 0, cm = 0, cp = 0, cx = 0;
        for (int i = skip; i < n; ++i) {
            tm += timings[i].tested.mean;
            tp += timings[i].tested.p95;
            tx += timings[i].tested.max;
            am += timings[i].accepted.mean;
            ap += timings[i].accepted.p95;
            ax += timings[i].accepted.max;
            cm += timings[i].cells_visited.mean;
            cp += timings[i].cells_visited.p95;
            cx += timings[i].cells_visited.max;
        }
        std::cout << "         per lookup (mean / p95 / max): tested " << tm * invd << " / "
                  << tp * invd << " / " << tx * invd << ", accepted " << am * invd << " / "
                  << ap * invd << " / " << ax * invd << ", cells " << cm * invd << " / "
                  << cp * invd << " / " << cx * invd << "\n";
    }
    if (p.radius_mode == RADIUS_FOOTPRINT || p.stats)
        std::cout << "         radius: mean r " << rMean * invd << ", mean max r "
                  << rMax * invd << ", mean radius " << rPx * invd
                  << " pixels\n";
    if (isStochasticHashGrid(gAccel))
        std::cout << "         (build = clearing the table; photons are filed during the trace)\n";
    if (args.verifyGather && !estimatesOnly(gAccel))
        std::cout << "         (query pass not timed: --verify-gather runs it twice)\n";
    const std::string queryKernel = std::string("query_") + accelName(gAccel) + (p.stats ? "_stats" : "");
    for (const KernelResources& k : kernelResources())
        if (queryKernel == k.name)
            std::cout << "         query kernel " << k.name << ": " << k.registers
                      << " registers, " << k.local_bytes << " bytes of local memory per thread\n";
    std::cout << "         " << gEnv.buildType << " build, OptiX validation "
              << (gEnv.validation ? "on" : "off") << ", " << gEnv.gpu << ", driver "
              << gEnv.driver << "\n";
}

// The adaptive LoD hash-grid SPPM's part of the record (sppm_lepski_host.h).
static void lepskiWriteJson(std::ostream& out);
static bool lepskiCullingOn();
// ... and RTPM's (rtpm_host.h).
static void rtpmWriteJson(std::ostream& out);
static bool rtpmCullingOn();

// The same numbers as JSON, one record per frame, for whatever plots the
// comparison ends up needing.  Written by hand rather than through a library
// because it is a dozen fields and the file is an output, not an interface.
static void writeTimingJson(const std::string& path,
    const std::vector<FrameTimings>& timings, const RunArgs& args, const Params& p)
{
    if (path.empty() || timings.empty())
        return;

    std::ofstream out(path);
    if (!out) {
        std::cerr << "[Timing] could not write " << path << "\n";
        return;
    }
    out.precision(6);
    out << std::fixed;
    out << "{\n";
    out << "  \"config\": " << jsonString(configName(gAccel, gCulling, p.radius_mode)) << ",\n";
    out << "  \"family\": \"" << familyName(gAccel, gCulling) << "\",\n";
    out << "  \"accel\": \"" << accelName(gAccel) << "\",\n";
    out << "  \"environment\": {\"build_type\": " << jsonString(gEnv.buildType)
        << ", \"validation\": " << (gEnv.validation ? "true" : "false")
        << ", \"gpu\": " << jsonString(gEnv.gpu)
        << ", \"driver\": " << jsonString(gEnv.driver)
        << ", \"cuda_driver_api\": " << gEnv.cudaDriverApi
        << ", \"cuda_runtime\": " << gEnv.cudaRuntime
        << ", \"optix\": " << OPTIX_VERSION << "},\n";
    // Registers and local memory per thread of the CUDA kernels; the OptiX
    // programs' only show up in a profiler (Nsight Compute).
    out << "  \"kernels\": {";
    {
        const std::vector<KernelResources> ks = kernelResources();
        for (size_t i = 0; i < ks.size(); ++i)
            out << (i ? ", " : "") << "\"" << ks[i].name << "\": {\"registers\": " << ks[i].registers
                << ", \"local_bytes\": " << ks[i].local_bytes << "}";
    }
    out << "},\n";
    out << "  \"table_load\": " << p.table_load << ",\n";
    out << "  \"inline_positions\": " << (p.inline_positions ? "true" : "false") << ",\n";
    out << "  \"hier_mask\": " << (p.hier_mask ? "true" : "false") << ",\n";
    out << "  \"fused\": " << (gFused ? "true" : "false") << ",\n";
    if (isSplatting(gAccel))
        out << "  \"splat_cell_factor\": " << p.splat_cell_factor << ",\n"
            << "  \"splat_stripes\": " << p.splat_stripes << ",\n";
    out << "  \"width\": " << p.width << ",\n";
    out << "  \"height\": " << p.height << ",\n";
    out << "  \"frames\": " << p.offline_frames << ",\n";
    out << "  \"warmup\": " << args.warmupFrames << ",\n";
    out << "  \"max_depth\": " << p.max_depth << ",\n";
    out << "  \"photon_paths\": " << p.num_photon_paths << ",\n";
    out << "  \"seed\": " << p.seed << ",\n";
    // Splatting culls implicitly: a photon that reaches no visible point adds nothing.
    // ... and the adaptive LoD hash-grid SPPM with a bitset of its own.
    if (gAccel == ACCEL_SPPM_LEPSKI)
        out << "  \"culling\": " << (lepskiCullingOn() ? "\"bitset\"" : "false") << ",\n";
    else if (gAccel == ACCEL_RTPM) // ... and RTPM with the paper's mask
        out << "  \"culling\": " << (rtpmCullingOn() ? "\"mask\"" : "false") << ",\n";
    else
        out << "  \"culling\": " << (isSplatting(gAccel) ? "\"implicit\"" : (gCulling ? "true" : "false")) << ",\n";
    out << "  \"mask_bits\": " << p.cull_mask_bits << ",\n";
    out << "  \"radius_mode\": \"" << (p.radius_mode == RADIUS_FOOTPRINT ? "footprint" : "global") << "\",\n";
    out << "  \"k_pix\": " << p.k_pix << ",\n";
    out << "  \"s0\": " << p.level_s0 << ",\n";
    if (gAccel == ACCEL_SPPM_LEPSKI)
        lepskiWriteJson(out);
    if (gAccel == ACCEL_RTPM)
        rtpmWriteJson(out);
    out << "  \"per_frame\": [\n";
    for (size_t i = 0; i < timings.size(); ++i) {
        const FrameTimings& t = timings[i];
        out << "    {\"frame\": " << i
            << ", \"radius\": " << t.radius
            << ", \"photons\": " << t.photons
            << ", \"deposits\": " << t.deposits
            << ", \"culled\": " << t.culled
            << ", \"r_max\": " << t.r_max
            << ", \"r_mean\": " << t.r_mean
            << ", \"r_px_mean\": " << t.r_px_mean
            << ", \"num_levels\": " << t.num_levels
            << ", \"level_min\": " << t.level_min
            << ", \"level_max\": " << t.level_max
            << ", \"entries\": " << t.entries
            << ", \"bytes\": " << t.bytes;
        if (p.stats)
            out << ", \"tested_mean\": " << t.tested.mean << ", \"tested_p95\": " << t.tested.p95
                << ", \"tested_max\": " << t.tested.max
                << ", \"accepted_mean\": " << t.accepted.mean << ", \"accepted_p95\": " << t.accepted.p95
                << ", \"accepted_max\": " << t.accepted.max
                << ", \"cells_mean\": " << t.cells_visited.mean << ", \"cells_p95\": " << t.cells_visited.p95
                << ", \"cells_max\": " << t.cells_visited.max;
        out
            << ", \"cells\": " << t.cells
            << ", \"buckets\": " << t.buckets
            << ", \"photon_trace_ms\": " << t.photon_trace
            << ", \"build_ms\": " << t.build
            << ", \"camera_ms\": " << t.camera
            << ", \"mark_ms\": " << t.mark
            << ", \"query_ms\": " << t.query
            << ", \"finalize_ms\": " << t.finalize
            << ", \"fused_ms\": " << t.fused
            << ", \"vp_entries\": " << t.vp_entries
            << ", \"vp_unique\": " << t.vp_unique
            << ", \"reached\": " << t.reached
            << ", \"pairs\": " << t.pairs
            << ", \"accum_ms\": " << t.accum << "}"
            << (i + 1 < timings.size() ? ",\n" : "\n");
    }
    out << "  ]\n";
    out << "}\n";
}

// How far the structure's answer is from the brute-force one.
struct GatherVerification {
    double max_abs = 0.0; // largest absolute difference in any channel
    double max_rel = 0.0; // largest relative difference where there is signal
    int differing_pixels = 0; // pixels past the tolerance below
    bool ok = true;
};

static GatherVerification compareRadiance(const std::vector<float3>& ref,
    const std::vector<float3>& test);

// Runs the query pass twice over the same photon map and the same gather
// points — once through the selected structure (uniform grid or hash grid),
// once over every photon — and compares the two images.  This is the correctness check the acceleration structures are held
// to: the structure is only allowed to change which photons get *looked at*,
// never which ones are inside the radius.
//
// The two are not expected to be bit-identical.  They add the same photons in a
// different order, and float addition is not associative, so a few ulps of
// drift per pixel is the floor.  A structure that misses photons fails by
// orders of magnitude more than that, which is what the tolerance is set for.
//
// Leaves query_radiance holding the structure's result, so the caller
// accumulates the image it would have rendered anyway.
static GatherVerification verifyQueryPass(RendererState& state, Params& p,
    CUstream& stream, std::vector<float3>& brute, std::vector<float3>& grid)
{
    const size_t pixels = size_t(p.width) * size_t(p.height);
    brute.resize(pixels);
    grid.resize(pixels);

    const int builtAccel = p.accel;

    p.accel = ACCEL_BRUTE_FORCE;
    launchQueryPass(state, p, stream);
    CUDA_CHECK(cudaMemcpy(brute.data(), p.query_radiance,
        pixels * sizeof(float3), cudaMemcpyDeviceToHost));

    p.accel = builtAccel;
    launchQueryPass(state, p, stream);
    CUDA_CHECK(cudaMemcpy(grid.data(), p.query_radiance,
        pixels * sizeof(float3), cudaMemcpyDeviceToHost));

    return compareRadiance(brute, grid);
}

// --verify-gather for splatting: the frame again as an exact gather over the
// same gather points and the same photon paths — the photons stored this time
// — through the matching structure (hash for splat, mlhash for mlsplat), and
// the two images compared.  Both find the same (photon, visible point) pairs
// and add the same products, so only the order of the sums tells them apart.
// Leaves query_radiance holding the splatting result.
static GatherVerification verifySplatFrame(RendererState& state, Params& p, CUstream& stream,
    std::vector<float3>& splat, std::vector<float3>& gather)
{
    const size_t pixels = size_t(p.width) * size_t(p.height);
    splat.resize(pixels);
    gather.resize(pixels);
    CUDA_CHECK(cudaMemcpy(splat.data(), p.query_radiance, pixels * sizeof(float3),
        cudaMemcpyDeviceToHost));

    const AccelType splatAccel = gAccel;
    gAccel = (splatAccel == ACCEL_MLSPLAT) ? ACCEL_MLHASH : ACCEL_HASH_GRID;
    p.culling = 0;
    launchPhotonTracing(state, p, stream, nullptr);
    launchQueryPass(state, p, stream);
    CUDA_CHECK(cudaMemcpy(gather.data(), p.query_radiance, pixels * sizeof(float3),
        cudaMemcpyDeviceToHost));
    gAccel = splatAccel;
    p.accel = splatAccel;

    CUDA_CHECK(cudaMemcpy(p.query_radiance, splat.data(), pixels * sizeof(float3),
        cudaMemcpyHostToDevice));
    return compareRadiance(gather, splat);
}

// How far `test` is from `ref`, pixel by pixel.
static GatherVerification compareRadiance(const std::vector<float3>& ref,
    const std::vector<float3>& test)
{
    const size_t pixels = std::min(ref.size(), test.size());
    GatherVerification v;
    for (size_t i = 0; i < pixels; ++i) {
        const float a[3] = { ref[i].x, ref[i].y, ref[i].z };
        const float b[3] = { test[i].x, test[i].y, test[i].z };
        bool bad = false;
        for (int c = 0; c < 3; ++c) {
            const double diff = std::fabs((double)a[c] - (double)b[c]);
            const double scale = std::max(std::fabs((double)a[c]), std::fabs((double)b[c]));
            v.max_abs = std::max(v.max_abs, diff);
            // Relative error is meaningless on a pixel that is essentially
            // black; the absolute figure covers those.
            if (scale > 1e-4) {
                const double rel = diff / scale;
                v.max_rel = std::max(v.max_rel, rel);
                if (rel > 1e-3)
                    bad = true;
            } else if (diff > 1e-5) {
                bad = true;
            }
        }
        if (bad)
            ++v.differing_pixels;
    }
    v.ok = (v.differing_pixels == 0);
    return v;
}

// Per-pixel images of the last frame's lookups, all three channels the same
// value: the radius in pixel footprints, r / (t * pixel_angle), which in
// footprint mode is radius_scale * k_pix wherever the clamp did not engage,
// and the cell level.  -1 where the pixel has no lookup.  Gather points are
// per-pixel, not photon data, and this is one readback at the end of a run.
static void writeGatherDebugImages(const std::string& prefix, const Params& p,
    int W, int H)
{
    std::vector<GatherPoint> gps(size_t(W) * size_t(H));
    CUDA_CHECK(cudaMemcpy(gps.data(), p.gather_points, gps.size() * sizeof(GatherPoint),
        cudaMemcpyDeviceToHost));
    std::vector<float3> rpx(gps.size()), level(gps.size());
    for (size_t i = 0; i < gps.size(); ++i) {
        const GatherPoint& g = gps[i];
        const float w = g.t * p.pixel_angle;
        const float v = (g.valid && w > 0.f) ? g.r / w : -1.f;
        const float l = g.valid ? float(g.level) : -1.f;
        rpx[i] = make_float3(v, v, v);
        level[i] = make_float3(l, l, l);
    }
    saveEXR(prefix + ".rpx.exr", rpx, W, H, true);
    saveEXR(prefix + ".level.exr", level, W, H, true);
    std::cout << "[Debug] " << prefix << ".rpx.exr and .level.exr, last frame: k_pix "
              << p.k_pix << ", radius scale " << p.radius_scale << "\n";
}

// The adaptive LoD hash-grid SPPM's host side (--accel sppm_lepski).
#include "sppm_lepski_host.h"
// RTPM's host side (--accel rtpm).
#include "rtpm_host.h"

static bool renderOffline(RendererState& state, Params& p, const RunArgs& args,
    const std::string& outputFile)
{
    const int W = static_cast<int>(p.width);
    const int H = static_cast<int>(p.height);

    uchar4* dFrame = nullptr;
    float3* dAccum = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dFrame), W * H * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAccum), W * H * sizeof(float3)));
    CUDA_CHECK(cudaMemset(dFrame, 0, W * H * sizeof(uchar4)));
    CUDA_CHECK(cudaMemset(dAccum, 0, W * H * sizeof(float3)));

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUdeviceptr dLightvisBuf = 0;
    if (p.render_mode == 2)
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dLightvisBuf), W * H * sizeof(float3)));

    if (p.render_mode == 1) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dGatherPoints),
            size_t(W) * H * sizeof(GatherPoint)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dQueryRadiance),
            size_t(W) * H * sizeof(float3)));
        p.gather_points = reinterpret_cast<GatherPoint*>(state.dGatherPoints);
        p.query_radiance = reinterpret_cast<float3*>(state.dQueryRadiance);

        // --stats: one counter of each kind per pixel, written by the query
        // pass and summarised on the device after it is timed.
        // With splatting they are counted per deposit instead, at most one
        // per bounce of every path.
        if (args.stats) {
            const size_t perPixel = size_t(W) * H;
            const size_t perDeposit = std::min<size_t>(size_t(p.num_photon_paths) * p.max_depth, kMaxPhotonSlots);
            const size_t slots = isSplatting(gAccel) ? perDeposit : perPixel;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.stat_tested), slots * sizeof(int)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.stat_accepted), slots * sizeof(int)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.stat_cells), perPixel * sizeof(int)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.stat_slot), sizeof(int)));
            p.stat_capacity = (int)slots;
            p.stats = 1;
        }

        // Splatting's accumulators: K copies of a float3 per visible point.
        if (isSplatting(gAccel)) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.splat_acc),
                size_t(p.splat_stripes) * W * H * 3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.splat_counts), 3 * sizeof(unsigned long long)));
        }
    }

    p.frame_buffer = dFrame;
    p.accum_buffer = reinterpret_cast<float3*>(dAccum);
    // Frame i is the same frame in every run — the same photons, gather points
    // and radius, all seeded by (index, frame, seed) — so starting at
    // --first-frame N renders exactly frames N.. of a longer run.  The image
    // is the average of the frames rendered.
    p.frame_index = args.firstFrame;

    // Print the schedule up front.  The last frame's radius is not the blur you
    // get: the frames are averaged with equal weight, so what the image behaves
    // like is sqrt(mean r_i^2), which settles around 1.22*r_N.  Seeing all three
    // is what stops r_1 from being chosen by feel.
    if (p.render_mode == 1 && p.use_ppm) {
        double sum2 = 0.0;
        for (int f = 0; f < p.offline_frames; ++f) {
            const double ri = ppmRadius(p.ppm_radius_initial, p.ppm_alpha, f);
            sum2 += ri * ri;
        }
        const double rEff = std::sqrt(sum2 / p.offline_frames);
        const float rN = ppmRadius(p.ppm_radius_initial, p.ppm_alpha, p.offline_frames - 1);
        std::cout << "[PPM] alpha = " << p.ppm_alpha
                  << "  frames = " << p.offline_frames
                  << "\n      r1   = " << p.ppm_radius_initial
                  << "  (" << (p.ppm_radius_initial / p.scene_radius) << " of scene radius)"
                  << "\n      rN   = " << rN << "  (" << (rN / p.ppm_radius_initial) << " of r1)"
                  << "\n      reff = " << rEff << "  <- the blur the averaged image actually has\n";
    }

    std::vector<FrameTimings> timings;
    timings.reserve(p.offline_frames);

    // Error against time (--checkpoints, --dump-frames).  Both happen between
    // frames, after everything of the frame has been timed, so neither the
    // file writes nor the --stats summaries are in the recorded time, which
    // is the sum of the frames' GPU pass timers.
    std::vector<int> checkpoints = args.checkpoints;
    if (args.checkpointsPow2)
        for (int n = 1; n <= p.offline_frames; n *= 2)
            checkpoints.push_back(n);
    std::sort(checkpoints.begin(), checkpoints.end());
    checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
    const bool isExr = outputFile.size() > 4 && outputFile.compare(outputFile.size() - 4, 4, ".exr") == 0;
    const std::string stem = isExr ? outputFile.substr(0, outputFile.size() - 4) : outputFile;
    const int cpDigits = (int)std::to_string(std::max(p.offline_frames, 1)).size();
    struct CheckpointRecord {
        int frames;
        int frameIndex;
        double gpuMs;
        std::string file;
    };
    std::vector<CheckpointRecord> cpRecords;
    double gpuMs = 0.0;
    if (!checkpoints.empty() && p.render_mode == 2)
        std::cerr << "[Checkpoints] not in photon-tracing-only mode; ignored\n";
    if (!args.dumpFrames.empty() && p.render_mode != 1)
        std::cerr << "[Dump] single frames only in photon mapping mode; ignored\n";

    auto writeCheckpoints = [&]() {
        std::ofstream out(stem + ".checkpoints.json");
        out.precision(6);
        out << std::fixed << "{\n  \"output\": " << jsonString(outputFile)
            << ",\n  \"mode\": \"" << (p.render_mode == 0 ? "pt" : "pm") << "\""
            << ",\n  \"config\": "
            << jsonString(p.render_mode == 0 ? std::string("pt") : configName(gAccel, gCulling, p.radius_mode))
            << ",\n  \"seed\": " << p.seed << ",\n  \"first_frame\": " << args.firstFrame
            << ",\n  \"frames\": " << p.offline_frames << ",\n  \"checkpoints\": [\n";
        for (size_t i = 0; i < cpRecords.size(); ++i)
            out << "    {\"frames\": " << cpRecords[i].frames << ", \"frame_index\": "
                << cpRecords[i].frameIndex << ", \"gpu_ms\": " << cpRecords[i].gpuMs
                << ", \"file\": " << jsonString(cpRecords[i].file) << "}"
                << (i + 1 < cpRecords.size() ? ",\n" : "\n");
        out << "  ]\n}\n";
    };

    // --verify-gather replaces the timed query pass with two untimed ones, so a
    // run either measures or checks — never both.
    const bool verifyGather = args.verifyGather && p.render_mode == 1
        && gAccel != ACCEL_BRUTE_FORCE && !estimatesOnly(gAccel);
    if (args.verifyGather && gAccel == ACCEL_BRUTE_FORCE)
        std::cerr << "[Verify] --accel brute builds no structure; nothing to compare against\n";
    if (args.verifyGather && estimatesOnly(gAccel))
        std::cerr << "[Verify] the " << accelLabel(gAccel) << " keeps one photon per "
                     "bucket, so it estimates the brute-force answer rather than "
                     "reproducing it; there is nothing to compare pixel by pixel\n";
    std::vector<float3> verifyBrute, verifyGrid;
    int verifyFailures = 0;
    FusedHistory fusedHist;
    if (gFused) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.fused_max_r), sizeof(unsigned int)));
        p.fused_store_points = verifyGather ? 1 : 0;
    }

    // --verify-culling traces every frame three times over the same gather
    // points — the full map twice, then the culled map — and runs the same
    // lookup over each.  The two full maps differ only in the order in which
    // the photons were deposited (slots come from an atomic), so their
    // difference is the float-summation floor, and the culled map is held to
    // that: exact culling changes which photons are stored, never which ones
    // lie within r, and a dropped photon would show up as a difference of
    // order 1/(photons per lookup), orders of magnitude above the floor.
    // Untimed, like --verify-gather.
    const bool verifyCulling = args.verifyCulling && p.render_mode == 1
        && !estimatesOnly(gAccel) && !isSplatting(gAccel) && !gFused;
    if (args.verifyCulling && estimatesOnly(gAccel))
        std::cerr << "[Verify] the " << accelLabel(gAccel) << " keeps one photon per "
                     "bucket, and culling changes which photons compete for a "
                     "bucket; compare averages over many frames instead\n";
    if (args.verifyCulling && (isSplatting(gAccel) || gFused))
        std::cerr << "[Verify] --verify-culling is for the gathers with a culled map;"
                     " splatting culls implicitly and --fused runs without culling\n";
    std::vector<float3> cullFull, cullFull2, cullTest;
    int cullFailures = 0;
    double cullWorst = 0.0, cullFloorWorst = 0.0;

    for (int f = 0; f < p.offline_frames; ++f) {

        if (p.render_mode == 0) {
            // Path Tracing launch
            gpuMs += launchPathTracing(state, p, stream);
        } else if (p.render_mode == 1) {
            // Probabilistic PPM: shrink the radius for this frame *before*
            // anything uses it, because the grid is binned at gather_radius.
            updatePPMRadius(p);

            FrameTimings ft;
            // Splatting culls implicitly — a deposit that reaches no visible
            // point adds nothing — so it has no mask.
            p.culling = (!isSplatting(gAccel) && (gCulling || verifyCulling)) ? 1 : 0;

            // Pass 1 — the camera walk, parking every pixel at its first
            // diffuse vertex.  It goes first so that the photon pass and the
            // build can be told what this frame's lookups are going to need
            // (with culling: which cells they search).  Nothing in it reads
            // the photon map, and every pass draws from its own random stream,
            // so moving it ahead of the photons changes no random number and
            // no pixel.
            // --fused: after the first frame, the structure first and then the
            // camera walk and lookup in one pass.
            // The adaptive LoD hash-grid SPPM is a frame of its own: four
            // stages, its own buffers, and it writes the image itself.
            const bool lepskiFrame = gAccel == ACCEL_SPPM_LEPSKI;
            const bool fusedFrame = gFused && fusedHist.valid;
            if (lepskiFrame) {
                GatherVerification v;
                launchLepskiFrame(state, p, stream, &ft, f == 0, verifyGather ? &v : nullptr);
                if (verifyGather) {
                    if (!v.ok)
                        ++verifyFailures;
                    std::cout << "\r[Verify] frame " << f << (v.ok ? "  OK  " : "  FAIL")
                              << "  sppm_lepski, grid over the culled map vs brute force over"
                                 " the full one: max abs diff = " << v.max_abs
                              << "  max rel = " << v.max_rel
                              << "  pixels off = " << v.differing_pixels << "\n";
                }
            } else if (fusedFrame)
                launchFusedFrame(state, p, stream, &ft, fusedHist);
            else
                ft.camera = launchCameraPass(state, p, stream, &ft);

            if (lepskiFrame) {
                // nothing left to launch
            } else if (fusedFrame) {
                if (verifyGather) {
                    // The fused frame stored its gather points; the lookup
                    // again over them, brute force, must agree.
                    const size_t pixels = size_t(W) * size_t(H);
                    verifyGrid.resize(pixels);
                    CUDA_CHECK(cudaMemcpy(verifyGrid.data(), p.query_radiance,
                        pixels * sizeof(float3), cudaMemcpyDeviceToHost));
                    const int built = p.accel;
                    p.accel = ACCEL_BRUTE_FORCE;
                    launchQueryPass(state, p, stream);
                    verifyBrute.resize(pixels);
                    CUDA_CHECK(cudaMemcpy(verifyBrute.data(), p.query_radiance,
                        pixels * sizeof(float3), cudaMemcpyDeviceToHost));
                    p.accel = built;
                    CUDA_CHECK(cudaMemcpy(p.query_radiance, verifyGrid.data(),
                        pixels * sizeof(float3), cudaMemcpyHostToDevice));
                    const GatherVerification v = compareRadiance(verifyBrute, verifyGrid);
                    if (!v.ok)
                        ++verifyFailures;
                    std::cout << "\r[Verify] frame " << f << (v.ok ? "  OK  " : "  FAIL")
                              << "  fused-" << accelName(gAccel) << " vs brute: max abs diff = "
                              << v.max_abs << "  max rel = " << v.max_rel
                              << "  pixels off = " << v.differing_pixels << "\n";
                }
            } else if (gAccel == ACCEL_RTPM) {
                // Family E: at the gather points the camera pass just left,
                // RTPM's own passes — the culling mask, the photon
                // generation, the photon structures, the collection — which
                // leave the frame's radiance where a query pass would.
                GatherVerification v;
                launchRtpmPasses(state, p, stream, &ft, verifyGather ? &v : nullptr);
                if (verifyGather) {
                    if (!v.ok)
                        ++verifyFailures;
                    std::cout << "\r[Verify] frame " << f << (v.ok ? "  OK  " : "  FAIL")
                              << "  rtpm, structures over the culled maps vs brute force over"
                                 " the full ones: max abs diff = " << v.max_abs
                              << "  max rel = " << v.max_rel
                              << "  pixels off = " << v.differing_pixels << "\n";
                }
            } else if (isSplatting(gAccel)) {
                // Family C: the visible points into their grid, the photons
                // splatted into them as they are deposited, the flux into
                // radiance.
                launchSplatPasses(state, p, stream, &ft);
                if (verifyGather) {
                    const GatherVerification v = verifySplatFrame(state, p, stream,
                        verifyGrid, verifyBrute);
                    if (!v.ok)
                        ++verifyFailures;
                    std::cout << "\r[Verify] frame " << f
                              << (v.ok ? "  OK  " : "  FAIL")
                              << "  " << accelName(gAccel) << " vs "
                              << (gAccel == ACCEL_MLSPLAT ? "mlhash" : "hash")
                              << ": max abs diff = " << v.max_abs
                              << "  max rel = " << v.max_rel
                              << "  pixels off = " << v.differing_pixels << "\n";
                }
            } else {
                // With culling, the cells those lookups will search are marked
                // next, from the gather points just written.
                if (p.culling)
                    ft.mark = launchMarkPass(state, p, stream);

                if (verifyCulling) {
                    const size_t pixels = size_t(W) * size_t(H);
                    cullFull.resize(pixels);
                    cullFull2.resize(pixels);
                    p.culling = 0;
                    for (std::vector<float3>* ref : { &cullFull, &cullFull2 }) {
                        launchPhotonTracing(state, p, stream);
                        launchQueryPass(state, p, stream);
                        CUDA_CHECK(cudaMemcpy(ref->data(), p.query_radiance,
                            pixels * sizeof(float3), cudaMemcpyDeviceToHost));
                    }
                    p.culling = 1;
                }

                // Pass 2 — Photon tracing, then the grid build.  A fresh map every
                // frame: reusing one map would make every frame share the same
                // density-estimate error, so the blotches would stay put no matter
                // how long it ran.  It is also what makes the frames independent
                // estimates, which is the assumption the PPM radius schedule is
                // derived under.
                launchPhotonTracing(state, p, stream, &ft);

                // Pass 3 — the photon lookup the camera pass parked.
                if (verifyGather) {
                    const GatherVerification v = verifyQueryPass(state, p, stream,
                        verifyBrute, verifyGrid);
                    if (!v.ok)
                        ++verifyFailures;
                    std::cout << "\r[Verify] frame " << f
                              << (v.ok ? "  OK  " : "  FAIL")
                              << "  " << accelName(gAccel) << " vs brute: max abs diff = " << v.max_abs
                              << "  max rel = " << v.max_rel
                              << "  pixels off = " << v.differing_pixels << "\n";
                } else {
                    ft.query = launchQueryPass(state, p, stream);
                }

                if (verifyCulling) {
                    cullTest.resize(cullFull.size());
                    CUDA_CHECK(cudaMemcpy(cullTest.data(), p.query_radiance,
                        cullTest.size() * sizeof(float3), cudaMemcpyDeviceToHost));
                    const GatherVerification v = compareRadiance(cullFull, cullTest);
                    const GatherVerification floor = compareRadiance(cullFull, cullFull2);
                    // At the floor: within 1e-5, or no further than twice what two
                    // full maps of the same frame are from each other.
                    const bool ok = v.ok
                        && v.max_rel <= std::max(1e-5, 2.0 * floor.max_rel);
                    if (!ok)
                        ++cullFailures;
                    cullWorst = std::max(cullWorst, v.max_rel);
                    cullFloorWorst = std::max(cullFloorWorst, floor.max_rel);
                    std::cout << "\r[Verify] frame " << f << (ok ? "  OK  " : "  FAIL")
                              << "  " << accelName(gAccel) << " culled vs full: max abs = "
                              << v.max_abs << "  max rel = " << v.max_rel
                              << "  | full vs full: max rel = " << floor.max_rel
                              << "  | culled " << 100.0 * ft.culled << "% of "
                              << ft.deposits << " deposits, levels " << p.num_levels << "\n";
                }
            }

            // The first frame of a fused run is a normal one: it gives the next
            // its radius and levels.
            if (gFused && !fusedFrame)
                rememberForFused(p, fusedHist);

            if (p.stats && isSplatting(gAccel)) {
                // per deposit: visible-point entries tested, pairs made
                int deposits = 0;
                CUDA_CHECK(cudaMemcpy(&deposits, p.stat_slot, sizeof(int), cudaMemcpyDeviceToHost));
                deposits = std::min(deposits, p.stat_capacity);
                summarizeQueryStats(p.stat_tested, deposits, state.grid, stream, ft.tested);
                summarizeQueryStats(p.stat_accepted, deposits, state.grid, stream, ft.accepted);
            } else if (p.stats) {
                summarizeQueryStats(p.stat_tested, W * H, state.grid, stream, ft.tested);
                summarizeQueryStats(p.stat_accepted, W * H, state.grid, stream, ft.accepted);
                summarizeQueryStats(p.stat_cells, W * H, state.grid, stream, ft.cells_visited);
            }

            if (!lepskiFrame)
                ft.accum = launchAccumulatePass(p, stream);
            timings.push_back(ft);
            gpuMs += ft.total();

            // This frame's own radiance, before the next frame overwrites it.
            if (std::binary_search(args.dumpFrames.begin(), args.dumpFrames.end(), f + 1)) {
                std::vector<float3> img(size_t(W) * H);
                CUDA_CHECK(cudaMemcpy(img.data(), p.query_radiance, img.size() * sizeof(float3),
                    cudaMemcpyDeviceToHost));
                saveEXR(stem + ".frame" + std::to_string(p.frame_index) + ".exr", img, W, H,
                    args.floatExr);
            }
        } else if (p.render_mode == 2) {
            // Nothing marks cells without a camera pass, so nothing is culled.
            p.culling = 0;
            launchPhotonTracing(state, p, stream);
            launchLightVis(state, p, stream, dLightvisBuf);
        }

        // The image accumulated so far, and the GPU time it took.
        if (p.render_mode != 2 && std::binary_search(checkpoints.begin(), checkpoints.end(), f + 1)) {
            std::vector<float3> img(size_t(W) * H);
            CUDA_CHECK(cudaMemcpy(img.data(), dAccum, img.size() * sizeof(float3),
                cudaMemcpyDeviceToHost));
            const float inv = 1.f / float(f + 1);
            for (auto& c : img) {
                c.x = std::isfinite(c.x) ? c.x * inv : 0.f;
                c.y = std::isfinite(c.y) ? c.y * inv : 0.f;
                c.z = std::isfinite(c.z) ? c.z * inv : 0.f;
            }
            std::string n = std::to_string(f + 1);
            n.insert(0, size_t(std::max(0, cpDigits - (int)n.size())), '0');
            const std::string file = stem + ".cp" + n + ".exr";
            saveEXR(file, img, W, H, args.floatExr);
            cpRecords.push_back({ f + 1, p.frame_index, gpuMs, file });
            writeCheckpoints();
        }

        p.frame_index++;
        std::cout << "\rOffline frame " << (f + 1) << "/" << p.offline_frames
                  << "  r = " << p.gather_radius << std::flush;
    }

    std::cout << "\n";

    reportTimings(timings, args, p);
    if (gAccel == ACCEL_SPPM_LEPSKI && p.render_mode == 1)
        lepskiReport(p);
    if (gAccel == ACCEL_RTPM && p.render_mode == 1)
        rtpmReport();
    writeTimingJson(args.timingJson, timings, args, p);
    // (the adaptive LoD hash-grid SPPM keeps gather points of its own; its
    // debug images are the views of --lepski-view)
    if (!args.debugExr.empty() && p.render_mode == 1 && gAccel != ACCEL_SPPM_LEPSKI)
        writeGatherDebugImages(args.debugExr, p, W, H);
    if (!args.debugExr.empty() && p.stats) {
        // Photons accepted by each lookup of the last frame (-1: no lookup).
        std::vector<int> accepted(size_t(W) * H);
        CUDA_CHECK(cudaMemcpy(accepted.data(), p.stat_accepted, accepted.size() * sizeof(int),
            cudaMemcpyDeviceToHost));
        std::vector<float3> img(accepted.size());
        for (size_t i = 0; i < accepted.size(); ++i)
            img[i] = make_float3(float(accepted[i]), float(accepted[i]), float(accepted[i]));
        saveEXR(args.debugExr + ".accepted.exr", img, W, H, true);
    }
    if (p.stats) {
        CUDA_CHECK(cudaFree(p.stat_tested));
        CUDA_CHECK(cudaFree(p.stat_accepted));
        CUDA_CHECK(cudaFree(p.stat_cells));
        CUDA_CHECK(cudaFree(p.stat_slot));
        p.stat_tested = p.stat_accepted = p.stat_cells = p.stat_slot = nullptr;
        p.stats = 0;
    }
    if (p.splat_acc) {
        CUDA_CHECK(cudaFree(p.splat_acc));
        CUDA_CHECK(cudaFree(p.splat_counts));
        p.splat_acc = nullptr;
        p.splat_counts = nullptr;
    }
    if (p.fused_max_r) {
        CUDA_CHECK(cudaFree(p.fused_max_r));
        p.fused_max_r = nullptr;
    }
    if (verifyGather) {
        std::cout << "[Verify] " << (verifyFailures == 0 ? "PASS" : "FAIL")
                  << " — " << (p.offline_frames - verifyFailures) << "/"
                  << p.offline_frames << " frames matched "
                  << (isSplatting(gAccel) ? (gAccel == ACCEL_MLSPLAT ? "the mlhash gather" : "the hash gather")
                                          : "brute force") << "\n";
    }
    if (verifyCulling) {
        std::cout << "[Verify] culling " << (cullFailures == 0 ? "PASS" : "FAIL")
                  << " — " << (p.offline_frames - cullFailures) << "/"
                  << p.offline_frames << " frames matched the full map"
                  << "  (worst max rel " << cullWorst << ", full-vs-full floor "
                  << cullFloorWorst << ")\n";
    }

    std::vector<float3> hostAccum(W * H);
    const float invF = 1.0f / float(p.offline_frames);

    if (p.render_mode == 2) {
        // The visualisation splats into its own buffer rather than accum_buffer.
        CUDA_CHECK(cudaMemcpy(hostAccum.data(), reinterpret_cast<void*>(dLightvisBuf),
                              W * H * sizeof(float3), cudaMemcpyDeviceToHost));
        for (auto& c : hostAccum) {
            c.x *= invF * p.lightvis_exposure;
            c.y *= invF * p.lightvis_exposure;
            c.z *= invF * p.lightvis_exposure;
        }
    } else {
        // Read accum_buffer and average
        CUDA_CHECK(cudaMemcpy(hostAccum.data(), dAccum,
                              W * H * sizeof(float3), cudaMemcpyDeviceToHost));
        for (auto& c : hostAccum) {
            c.x *= invF;
            c.y *= invF;
            c.z *= invF;
        }
    }

    for (auto& c : hostAccum) {
        if (!std::isfinite(c.x)) c.x = 0.f;
        if (!std::isfinite(c.y)) c.y = 0.f;
        if (!std::isfinite(c.z)) c.z = 0.f;
    }

    bool ok = saveEXR(outputFile, hostAccum, W, H, args.floatExr);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dFrame));
    CUDA_CHECK(cudaFree(dAccum));
    if (dLightvisBuf)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dLightvisBuf)));
    if (state.dGatherPoints) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.dGatherPoints)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.dQueryRadiance)));
        state.dGatherPoints = state.dQueryRadiance = 0;
    }
    freeGridScratch(state.grid);

    // A failed comparison is a failed run: the image is there, but it was not
    // produced by a lookup anyone should trust.
    return ok && verifyFailures == 0 && cullFailures == 0;
}

bool renderRealtime(const CameraFileState& cam, RendererState& state, Params& p) {
    int W = p.width, H = p.height;

    GLFWwindow* window = initGL();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // optional later:
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    const char* glsl_version = "#version 330";
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

       // Window / PBO setup
    GLuint pbo, displayTex;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, W * H * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsResource* cudaPBO;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard));
    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    initQuad();

    CUdeviceptr dAccum;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAccum), W * H * sizeof(float3)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
    p.accum_buffer = reinterpret_cast<float3*>(dAccum);

    CUdeviceptr dLightvisBuf = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dLightvisBuf),
                        W * H * sizeof(float3)));

    // The split gather needs these in every mode here, because the render mode
    // is a dropdown away at any moment.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dGatherPoints),
        size_t(W) * H * sizeof(GatherPoint)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dQueryRadiance),
        size_t(W) * H * sizeof(float3)));
    p.gather_points = reinterpret_cast<GatherPoint*>(state.dGatherPoints);
    p.query_radiance = reinterpret_cast<float3*>(state.dQueryRadiance);
    // ... and splatting's accumulators, the structure being a dropdown too.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.splat_acc),
        size_t(p.splat_stripes) * W * H * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p.splat_counts), 3 * sizeof(unsigned long long)));

    p.frame_index = 0;

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bool photons_valid = false;
    bool retracePhotons = true; // a fresh map per frame is what makes it converge
    int lastDeposits = 0;
    FrameTimings lastFrame; // statistics of the last photon pass, for the GUI

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Renderer (F1 to toggle GUI mode)");

        bool resetAccum = false;
        bool resetPhotons = false;
        bool skyChanged = false;

        ImGui::Text("F1 - Toggle GUI mode");
        ImGui::Text("ESC - Exit app");
        ImGui::Text("Frame: %u", p.frame_index);

        // Render mode
        const char* renderModes[] = { "Path Tracing", "Photon Mapping", "Photon Tracing Only" };
        int renderMode = p.render_mode;
        if (ImGui::Combo("Render mode", &renderMode, renderModes, IM_ARRAYSIZE(renderModes))) {
            p.render_mode = renderMode;
            if (p.render_mode == 1)
                p.samples_per_pixel = 1;
            resetAccum = true;
        }

        if (ImGui::CollapsingHeader("Sampling")) {
            // Photon mapping pins this to 1 — see Params::samples_per_pixel.
            // Saying so beats a slider that silently does nothing.
            if (p.render_mode == 1) {
                ImGui::Text("Samples / pixel: 1");
                ImGui::SameLine();
                ImGui::TextDisabled("(fixed: frames, not samples, are what converge a photon map)");
            } else if (ImGui::SliderInt("Samples / pixel", &p.samples_per_pixel, 1, 16)) {
                resetAccum = true;
            }
            if (ImGui::SliderInt("Max depth", &p.max_depth, 1, 16)) {
                resetAccum = true;
                resetPhotons = true;
            }
        }

        if (ImGui::CollapsingHeader("Lighting")) {
            if (ImGui::SliderFloat("Sky intensity", &p.sky_intensity, 0.f, 4.f, "%.2f")) {
                resetAccum = true;
                resetPhotons = true;
                skyChanged = true;
            }
            bool skyPhotons = p.emit_sky_photons != 0;
            if (ImGui::Checkbox("Sky emits photons", &skyPhotons)) {
                p.emit_sky_photons = skyPhotons ? 1 : 0;
                resetAccum = true;
                resetPhotons = true;
                skyChanged = true;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(photon mapping only)");
            // Say plainly when the control cannot do anything, rather than
            // leaving it looking broken: it is inert in path-tracing mode, and
            // inert again when the sky is off or cannot reach the scene.
            if (p.render_mode == 0)
                ImGui::TextDisabled("  no effect in Path Tracing mode");
            else if (p.sky_intensity <= 0.f)
                ImGui::TextDisabled("  no effect while Sky intensity is 0");
            else if (p.emit_sky_photons && p.sky_select_prob <= 0.f)
                ImGui::TextDisabled("  sky carries no flux in this scene");
            ImGui::Text("Photon paths from sky: %.0f%%  (0%% = area lights only)",
                100.0 * p.sky_select_prob);
        }

        if (p.render_mode != 0 && ImGui::CollapsingHeader("Photon map", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::SliderInt("Paths / pass", &p.num_photon_paths, 50'000, 4'000'000)) {
                resetAccum = true;
                resetPhotons = true;
            }
            ImGui::Text("Stored: %d / %d", lastDeposits, p.photon_capacity);

            float rMin = 0.001f * p.scene_radius;
            float rMax = 0.200f * p.scene_radius;
            const char* rLabel = p.use_ppm ? "Initial radius (r1)" : "Gather radius";
            if (ImGui::SliderFloat(rLabel, &p.ppm_radius_initial, rMin, rMax, "%.4f")) {
                p.gather_radius = p.ppm_radius_initial;
                resetAccum = true;
                resetPhotons = true; // the grid is binned at this radius
            }

            bool ppm = p.use_ppm != 0;
            if (ImGui::Checkbox("Progressive radius (PPM)", &ppm)) {
                p.use_ppm = ppm ? 1 : 0;
                if (!p.use_ppm)
                    p.gather_radius = p.ppm_radius_initial;
                else
                    p.adaptive_radius = 0; // the two do not compose
                resetAccum = true;
                resetPhotons = true;
            }
            if (p.use_ppm) {
                if (ImGui::SliderFloat("alpha", &p.ppm_alpha, 0.30f, 0.99f, "%.3f")) {
                    resetAccum = true;
                    resetPhotons = true;
                }
                ImGui::SameLine();
                ImGui::TextDisabled("(2/3 = MSE-optimal)");
                // r shrinks as i^(-(1-alpha)/2), which is slow enough that it is
                // worth showing rather than assuming.
                ImGui::Text("r = %.5f   (%.2f%% of r1)", p.gather_radius,
                    100.f * p.gather_radius / std::max(p.ppm_radius_initial, 1e-9f));
            }

            // Per-pixel radius: k_pix pixel footprints at the end of the camera
            // path, shrunk by the same schedule.  Global keeps one radius per
            // frame, the slider above.
            const char* radiusModes[] = { "Global", "Footprint (per pixel)" };
            int radiusMode = p.radius_mode;
            if (ImGui::Combo("Radius", &radiusMode, radiusModes, IM_ARRAYSIZE(radiusModes))) {
                p.radius_mode = radiusMode;
                if (p.radius_mode == RADIUS_FOOTPRINT)
                    p.adaptive_radius = 0; // the two do not compose
                resetAccum = true;
                resetPhotons = true; // the grid is binned at the largest radius
            }
            if (p.radius_mode == RADIUS_FOOTPRINT
                && ImGui::SliderFloat("k_pix", &p.k_pix, 0.5f, 32.f, "%.2f pixels")) {
                resetAccum = true;
                resetPhotons = true;
            }

            bool adaptive = p.adaptive_radius != 0;
            if (!p.use_ppm && p.radius_mode == RADIUS_GLOBAL
                && ImGui::Checkbox("Adaptive radius", &adaptive)) {
                p.adaptive_radius = adaptive ? 1 : 0;
                resetAccum = true;
            }
            if (!p.use_ppm && p.radius_mode == RADIUS_GLOBAL && adaptive) {
                if (ImGui::SliderInt("Target photons", &p.target_photons, 8, 512))
                    resetAccum = true;
            }

            if (ImGui::SliderFloat("Plane tolerance", &p.photon_plane_tol, 0.01f, 1.0f, "%.3f"))
                resetAccum = true;
            if (ImGui::SliderFloat("Normal tolerance", &p.photon_normal_tol, -1.0f, 0.99f, "%.3f"))
                resetAccum = true;

            bool storeDirect = p.store_direct_photons != 0;
            if (ImGui::Checkbox("Direct light from photons", &storeDirect)) {
                p.store_direct_photons = storeDirect ? 1 : 0;
                resetAccum = true;
                resetPhotons = true;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(off = shadow rays, much quieter)");

            if (ImGui::SliderFloat("Power scale", &p.photon_power_scale, 0.1f, 10.f, "%.2f")) {
                resetAccum = true;
                resetPhotons = true;
            }

            // Turning it off freezes the map the next frame traces, so that
            // one must not be culled: a map culled for one frame's lookups is
            // missing photons the next frame's lookups need.
            if (ImGui::Checkbox("Retrace every frame", &retracePhotons) && gCulling)
                resetPhotons = true;
            ImGui::SameLine();
            ImGui::TextDisabled(p.radius_mode == RADIUS_FOOTPRINT
                    ? "(always on with a per-pixel radius)"
                    : "(off = noise stops converging)");

            bool culling = gCulling;
            if (ImGui::Checkbox("Photon culling", &culling)) {
                gCulling = culling;
                resetAccum = true;
                resetPhotons = true;
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(store only photons a lookup can reach)");
            if (gCulling && !retracePhotons && p.radius_mode == RADIUS_GLOBAL)
                ImGui::TextDisabled("  off while Retrace every frame is off");
            else if (gCulling && p.render_mode != 1)
                ImGui::TextDisabled("  photon mapping only");
            else if (gCulling)
                ImGui::Text("Culled: %.1f%% of %lld deposits", 100.0 * lastFrame.culled,
                    lastFrame.deposits);

            // Switching between the exact structures must not change the
            // image: the same photons pass the same tests, only the set of
            // photons *looked at* differs — splatting included, which finds
            // the very same pairs from the photon's side.  Brute force is the
            // reference that tests every photon.  The two stochastic hash
            // grids are the exception by design — one photon per bucket, so
            // they add noise (and the non-rectified one a bias); see
            // insertStochastic.  Listed in AccelType order: the index is the
            // enum value.
            const char* accelItems[] = { "Brute force", "Uniform grid", "Hash grid",
                "Stochastic hash grid", "Rectified stochastic hash grid",
                "Multi-level hash grid", "Multi-level rectified hash grid",
                "Splatting (single level)", "Splatting (multi-level)", "Stochastic splatting",
                "Adaptive LoD hash-grid SPPM (Lepski)", "RTPM (photon BVH, Kern et al. 2023)" };
            int accelIndex = (int)gAccel;
            if (ImGui::Combo("Acceleration structure", &accelIndex, accelItems,
                    IM_ARRAYSIZE(accelItems))) {
                gAccel = (AccelType)accelIndex;
                // Splatting fixes the radii before the photons: no adaptive one.
                if (isSplatting(gAccel))
                    p.adaptive_radius = 0;
                // The adaptive LoD hash-grid SPPM's buffers go when another
                // structure is chosen, and come back if it is chosen again.
                if (gAccel != ACCEL_SPPM_LEPSKI)
                    lepskiFree(gLepski.buffers);
                // ... and so do RTPM's.  It has no adaptive radius either:
                // its boxes are built before any photon is counted.
                if (gAccel != ACCEL_RTPM)
                    rtpmRelease();
                else
                    p.adaptive_radius = 0;
                resetAccum = true;
                resetPhotons = true; // the structure is built by the photon pass
            }
            if (isSplatting(gAccel))
                ImGui::TextDisabled("  culling implicit: only photons reaching a visible point count");
            // The adaptive LoD hash-grid SPPM's own controls (sppm_lepski_host.h).
            if (gAccel == ACCEL_SPPM_LEPSKI)
                lepskiGui(lastFrame);
            // RTPM's own controls (rtpm_host.h); each changes the estimate.
            if (gAccel == ACCEL_RTPM && rtpmGui(lastFrame)) {
                resetAccum = true;
                resetPhotons = true;
            }
        }

        if (p.render_mode == 2 && ImGui::CollapsingHeader("Photon visualisation", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::SliderFloat("Exposure", &p.lightvis_exposure, 0.01f, 20.f, "%.2f", ImGuiSliderFlags_Logarithmic))
                resetAccum = true;
            if (ImGui::SliderInt("Splat radius (px)", &p.lightvis_splat_px, 0, 4))
                resetAccum = true;
        }

        // Camera position
        bool cameraEdited = false;
        if (ImGui::CollapsingHeader("Camera")) {
            float camPos[3] = { p.cam_eye.x, p.cam_eye.y, p.cam_eye.z };
            if (ImGui::InputFloat3("Camera position", camPos, "%.3f")) {
                p.cam_eye = make_float3(camPos[0], camPos[1], camPos[2]);
                cameraEdited = true;
            }
            float yaw = gYaw;
            if (ImGui::InputFloat("Yaw", &yaw, 0.5f, 5.0f, "%.3f")) {
                gYaw = yaw;
                cameraEdited = true;
            }
            float pitch = gPitch;
            if (ImGui::SliderFloat("Pitch", &pitch, -89.0f, 89.0f, "%.3f")) {
                gPitch = pitch;
                cameraEdited = true;
            }
        }

        ImGui::End();

        if (cameraEdited) {
            rebuildCameraVectors(p);
            p.frame_index = 0;
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
            gCameraChanged = false;
        }

        if (skyChanged)
            p.sky_select_prob = computeSkySelectProb(p, state.hostLights);

        if (resetPhotons)
            photons_valid = false;

        if (resetAccum) {
            p.frame_index = 0;
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
        }

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;

        if (!gGuiMode && !ImGui::GetIO().WantCaptureKeyboard) {
            handleKeys(window, p, 4.0f);
        }

        static bool cKeyWasDown = false;
        bool cKeyDown = glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS;
        if (!gGuiMode && cKeyDown && !cKeyWasDown) {
            p.cam_eye = cam.eye;
            gYaw = cam.yaw;
            gPitch = cam.pitch;
            gCameraChanged = true;
        }
        cKeyWasDown = cKeyDown;

        if (gCameraChanged) {
            rebuildCameraVectors(p);
            p.frame_index = 0;
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
            gCameraChanged = false;
        }

        static bool vKeyWasDown = false;
        bool vKeyDown = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
        if (!gGuiMode && vKeyDown && !vKeyWasDown) {
            std::cout
                << "\n=== Camera State ===\n"
                << "p.cam_eye = make_float3("
                << p.cam_eye.x << "f, "
                << p.cam_eye.y << "f, "
                << p.cam_eye.z << "f);\n"
                << "gYaw = " << gYaw << "f;\n"
                << "gPitch = " << gPitch << "f;\n"
                << "====================\n";
        }
        vKeyWasDown = vKeyDown;

        static bool f1WasDown = false;
        bool f1Down = glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS;
        if (f1Down && !f1WasDown) {
            gGuiMode = !gGuiMode;

            if (gGuiMode) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            } else {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                gFirstMouse = true;
            }
            std::cout << gGuiMode << '\n';
        }
        f1WasDown = f1Down;

        uchar4* devPtr;
        size_t sz;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPBO, stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&devPtr), &sz, cudaPBO));
        p.frame_buffer = devPtr;

        if (p.render_mode == 0) {
            // Path Tracing launch
            launchPathTracing(state, p, stream);
        } else {
            // Photons are re-traced every frame by default: the accumulation
            // buffer can only average away photon noise if each frame sees an
            // *independent* photon map.  Holding one map fixed (as this used
            // to) freezes its density-estimate error into the image, which is
            // what left the permanent blotches.
            //
            // A per-pixel radius needs it too: the radii move with the pixel
            // jitter every frame, and a frozen single-level grid's cells, sized
            // for the largest radius of the frame it was traced in, need not
            // cover the next frame's.
            //
            // Splatting keeps no map at all: every frame splats a fresh photon
            // pass into that frame's visible points.
            //
            // The adaptive LoD hash-grid SPPM is an iteration of its own; a
            // frame index of 0 — the camera moved, or a setting changed —
            // restarts its running estimate.
            if (p.render_mode == 1 && gAccel == ACCEL_SPPM_LEPSKI) {
                launchLepskiFrame(state, p, stream, &lastFrame, p.frame_index == 0);
                lastDeposits = lastFrame.photons;
                photons_valid = false;
            } else if (p.render_mode == 1 && gAccel == ACCEL_RTPM) {
                // RTPM traces its own photon maps every frame, at the gather
                // points of the renderer's camera pass.
                updatePPMRadius(p);
                p.culling = 0;
                launchCameraPass(state, p, stream);
                launchRtpmPasses(state, p, stream, &lastFrame);
                launchAccumulatePass(p, stream);
                lastDeposits = lastFrame.photons;
                photons_valid = false;
            } else if (p.render_mode == 1 && isSplatting(gAccel)) {
                updatePPMRadius(p);
                p.culling = 0;
                launchCameraPass(state, p, stream);
                launchSplatPasses(state, p, stream, &lastFrame);
                launchAccumulatePass(p, stream);
                photons_valid = false;
            } else {
                const bool retrace = retracePhotons || p.radius_mode == RADIUS_FOOTPRINT;
                const bool tracePhotons = retrace || !photons_valid;

                // Before anything else — the grid is binned at gather_radius.
                // With "Retrace every frame" off the map (and so the radius) is
                // frozen, which is consistent: a schedule that shrank the radius
                // over a map that never changes would just re-filter the same
                // photons and converge to nothing.
                if (tracePhotons)
                    updatePPMRadius(p);

                // Culling keeps only what this frame's lookups can reach, so it
                // needs a camera pass to mark the cells and a map that is not
                // reused by the frames after this one.
                p.culling = (gCulling && retrace && p.render_mode == 1) ? 1 : 0;

                // Pass 1 — the camera walk (photon mapping only).  Ahead of the
                // photons, as in renderOffline, so that the photon pass can be
                // told what this frame's lookups are going to need.
                if (p.render_mode == 1)
                    launchCameraPass(state, p, stream);
                if (p.culling)
                    launchMarkPass(state, p, stream);

                // Pass 2 — Photon tracing and the grid build.
                if (tracePhotons) {
                    lastDeposits = launchPhotonTracing(state, p, stream, &lastFrame);
                    photons_valid = true;
                }

                // Pass 3 — the photon lookup, or splatting the map straight onto
                // the film
                if (p.render_mode == 1) {
                    launchQueryPass(state, p, stream);
                    launchAccumulatePass(p, stream);
                } else {
                    launchLightVis(state, p, stream, dLightvisBuf);
                }
            }
        }

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPBO, stream));
        p.frame_index++;
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        drawQuad(displayTex);
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }

    // Cleanups
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    cudaStreamDestroy(stream);
    cudaFree(reinterpret_cast<void*>(dAccum));
    cudaFree(reinterpret_cast<void*>(dLightvisBuf));
    cudaFree(reinterpret_cast<void*>(state.dGatherPoints));
    cudaFree(reinterpret_cast<void*>(state.dQueryRadiance));
    state.dGatherPoints = state.dQueryRadiance = 0;
    cudaFree(p.splat_acc);
    cudaFree(p.splat_counts);
    p.splat_acc = nullptr;
    p.splat_counts = nullptr;
    freeGridScratch(state.grid);
    cudaGraphicsUnregisterResource(cudaPBO);

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &displayTex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return true; // Return true if successful
}

// main
int main(int argc, char** argv) {

    // program arguments
    RunArgs args{};
    args = parseArgs(argc, argv);
    // Checked before anything is loaded, so a typo fails in milliseconds
    // rather than after the scene and the pipeline have been built.
    // The reservoir unit test needs nothing but the GPU.
    if (args.selftestReservoir)
        return runReservoirSelfTest(10, 100000) ? 0 : 1;
    gRectBuckets = args.rectBuckets;
    gMlrectInTrace = args.mlrectInTrace;

    if (!parseAccel(args.accel, gAccel)) {
        std::cerr << "Unknown --accel value: " << args.accel
                  << " (expected brute, uniform, hash, stochastic, rect, mlhash, mlrect,"
                     " splat, mlsplat, stochsplat, sppm_lepski or rtpm)\n";
        return 2;
    }
    if (gAccel == ACCEL_SPPM_LEPSKI && args.verifyCulling) {
        std::cerr << "--accel sppm_lepski culls with a bitset of its own; --verify-gather"
                     " checks it together with the grid\n";
        return 2;
    }
    if (gAccel == ACCEL_RTPM && args.verifyCulling) {
        std::cerr << "--accel rtpm culls with a mask of its own (--rtpm-culling); --verify-gather"
                     " checks it together with the photon structures\n";
        return 2;
    }
    gLepski.options = args.lepski;
    gRtpm.options = args.rtpm;
    gFused = args.fused;
    if (gFused && (!args.offline || !args.photon || fusedIndex(gAccel) < 0 || args.culling)) {
        std::cerr << "--fused is for offline photon mapping in family A: --accel uniform, hash,"
                     " rect, mlhash or mlrect, without culling\n";
        return 2;
    }
    CameraFileState cam{};
    cam = loadCameraFile(args.cameraFile);

    RendererState state;
    Scene scene;
    Params p = {};

    // --res only applies offline: the realtime path renders into a fixed-size
    // GL window, so a mismatch there would just corrupt the display.
    int W = 1280, H = 720;
    if (args.offline && args.width > 0 && args.height > 0) {
        W = args.width;
        H = args.height;
    }

    p.width = W;
    p.height = H;

    // Initial camera state
    p.cam_eye = cam.eye;
    gYaw = cam.yaw;
    gPitch = cam.pitch;
    rebuildCameraVectors(p);

    createContext(state, args.validation);
    gEnv = queryEnvironment(args.validation);
    std::cout << "Context created (" << gEnv.buildType << " build, OptiX validation "
              << (args.validation ? "on" : "off") << ").\n";

    loadScene(args.sceneFile, scene, state);
    std::cout << "Materials: " << scene.materials.size()
              << ", Triangles: " << scene.triangles.size() << "\n";

    // Build emissive light list
    buildLightList(scene, state);

    // Needed before the photon defaults below: they are scene-relative.
    computeSceneBounds(scene, p);

    // Photons are re-traced every frame now, so a pass is sized for one frame
    // rather than for the whole render; the frames average together.
    p.num_photon_paths = 400'000;
    p.photon_capacity = 4'000'000;
    CUDA_CHECK(cudaMalloc(&p.photon_map, size_t(p.photon_capacity) * sizeof(Photon)));
    CUDA_CHECK(cudaMalloc(&p.photon_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(p.photon_count, 0, sizeof(int)));

    p.offline_frames = 8; // number of frames to render in offline mode
    p.samples_per_pixel = 4;
    p.max_depth         = 8;
    p.seed              = 0; // --seed overrides; 0 keeps every run reproducible

    // Gather radius as a fraction of the scene, not an absolute 1.0 — a radius
    // that happens to be hucdge for the scene is exactly what smears the photon
    // estimate into big soft stains.
    p.gather_radius = 0.015f * p.scene_radius;

    // Probabilistic PPM.  alpha = 2/3 is the MSE-optimal exponent, not a taste
    // knob — see the derivation on Params::use_ppm.
    p.use_ppm = (args.ppmAlpha >= 0.f) ? (args.ppmAlpha > 0.f ? 1 : 0) : 1;
    p.ppm_alpha = (args.ppmAlpha > 0.f) ? args.ppmAlpha : 2.f / 3.f;
    p.ppm_radius_initial = p.gather_radius;

    // The per-pixel adaptive shrink and the PPM schedule are two answers to the
    // same question and they do not compose.  Worse, the adaptive one picks its
    // radius from the photon count of the very map it then integrates, so the
    // radius is correlated with the flux — a bias the schedule's analysis does
    // not cover and cannot drive to zero.  PPM wins; adaptive is the fallback.
    p.adaptive_radius = p.use_ppm ? 0 : 1;
    p.target_photons = 64;
    p.photon_plane_tol = 0.20f; // |offset along n| <= 20% of the radius
    p.photon_normal_tol = 0.70f; // ~45 degrees of normal agreement
    p.store_direct_photons = 0; // direct light comes from shadow rays instead
    p.photon_power_scale = 1.0f; // physically correct; not a brightness fudge

    // The sky was already lighting the path-traced image through the miss
    // shader; now the photon pass emits it too, so both modes agree.
    // Sky off by default.  With it on, the two modes reach it by different
    // routes — the path tracer through the miss shader on every escaping
    // continuation ray, photon mapping through an analytic term plus emitted
    // sky photons — and unless emit_sky_photons is also on they are not even
    // integrating the same light.  Turning it off removes that asymmetry
    // entirely; --sky turns it back on when you want to exercise it.
    p.sky_intensity = 0.0f;
    p.emit_sky_photons = 0; // without this photon mapping gets no indirect sky

    p.lightvis_exposure = 1.0f;
    p.lightvis_splat_px = 1;

    p.render_mode = 0; // start with path tracing by default
    if (args.photon) {
        p.render_mode = 1; // start with photon mapping
    }
    if (args.lightvis) {
        p.render_mode = 2; // photon tracing only (visualise the map)
    }

    // One camera sample per frame in photon mapping — see the note on
    // Params::samples_per_pixel.  The camera pass ignores the field anyway;
    // setting it here keeps what gets printed and saved honest.
    if (p.render_mode == 1)
        p.samples_per_pixel = 1;


    if (args.frames > 0)
        p.offline_frames = args.frames;

    // Resolve r_1 — after the frame count, because --final-radius is defined in
    // terms of it.  An oversized r_1 is the expensive mistake here: the frames
    // are averaged with equal weight, so the early wide-radius frames stay in
    // the result forever.  Mean bias goes as r_1^2 * N^(alpha-1), so holding the
    // blur fixed while doubling r_1 costs 2^(2/(1-alpha)) = 64x the frames at
    // alpha = 2/3.  Naming the final radius instead makes that impossible to get
    // wrong by accident.
    if (args.finalRadius > 0.f) {
        const float rN = args.finalRadius * p.scene_radius;
        const float shrink = p.use_ppm
            ? ppmRadius(1.f, p.ppm_alpha, p.offline_frames - 1) : 1.f;
        p.ppm_radius_initial = rN / shrink;
    } else if (args.radius > 0.f) {
        p.ppm_radius_initial = args.radius * p.scene_radius;
    }
    p.gather_radius = p.ppm_radius_initial;

    // Cell levels, shared by culling and the multi-level grids.  Counted from
    // the scene box minimum; level 0 is 2^-18 of the box diagonal unless --s0
    // says otherwise, so the 32 levels run from 2^-18 to 2^13 diagonals.
    {
        const float dx = p.scene_aabb_max.x - p.scene_aabb_min.x;
        const float dy = p.scene_aabb_max.y - p.scene_aabb_min.y;
        const float dz = p.scene_aabb_max.z - p.scene_aabb_min.z;
        const float diag = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1e-6f);
        p.level_origin = p.scene_aabb_min;
        p.level_s0 = (args.s0 > 0.f) ? args.s0 : std::ldexp(diag, -18);

        // Per-pixel radius.  The clamp is a safety net for degenerate paths,
        // not part of the method.
        p.radius_mode = (args.radiusMode == "footprint") ? RADIUS_FOOTPRINT : RADIUS_GLOBAL;
        p.k_pix = args.kPix;
        p.radius_scale = 1.f;
        p.radius_min = 1e-6f * diag;
        p.radius_max = 0.25f * diag;

        // The top level must still be at least as wide as any radius a
        // lookup can ask for, or levelOf has no level to give it.
        const float rMax = std::max(p.ppm_radius_initial, p.radius_max);
        if (levelCellSize(p.level_s0, MAX_LEVELS - 1) < rMax) {
            std::cerr << "--s0 " << p.level_s0 << " is too small: the top level's cells ("
                      << levelCellSize(p.level_s0, MAX_LEVELS - 1)
                      << ") are narrower than a radius of " << rMax << "\n";
            return 2;
        }
    }
    p.cull_mask_bits = (unsigned int)args.maskBits;
    p.table_load = args.tableLoad;
    p.splat_cell_factor = args.splatCellFactor;
    p.splat_stripes = args.splatStripes;
    p.inline_positions = args.inlinePositions ? 1 : 0;
    p.hier_mask = args.hierMask ? 1 : 0;
    gCulling = args.culling;

    // The adaptive radius re-gathers with a radius picked from the photon
    // count; a per-pixel radius is already a per-pixel choice, and the two do
    // not compose.  Splatting fixes every radius before the photons are
    // traced, so it cannot pick one from what they deliver.
    if (p.radius_mode == RADIUS_FOOTPRINT)
        p.adaptive_radius = 0;
    if (isSplatting(gAccel)) {
        if (p.adaptive_radius)
            std::cout << "[Splat] the adaptive radius needs the photons before the radius;"
                         " off for splatting\n";
        p.adaptive_radius = 0;
        if (gCulling)
            std::cout << "[Splat] --culling is ignored: splatting culls implicitly\n";
        gCulling = false;
    }
    // The adaptive LoD hash-grid SPPM takes its radii from the pixel
    // footprint and keeps them fixed — it picks a bandwidth per pixel instead
    // of shrinking one — so it has neither the schedule nor the adaptive
    // radius, and it culls with a bitset of its own.
    if (gAccel == ACCEL_SPPM_LEPSKI) {
        p.radius_mode = RADIUS_FOOTPRINT;
        p.use_ppm = 0;
        p.adaptive_radius = 0;
        if (gCulling)
            std::cout << "[sppm_lepski] --culling is ignored: the method culls with its own"
                         " active-cell bitset\n";
        gCulling = false;
    }
    // RTPM builds its photon boxes before a single photon is counted, so it
    // cannot pick a radius from what a gather finds; and its culling is the
    // paper's own mask (--rtpm-culling), not the renderer's.
    if (gAccel == ACCEL_RTPM) {
        p.adaptive_radius = 0;
        if (gCulling)
            std::cout << "[rtpm] --culling is ignored: RTPM culls with its own mask"
                         " (--rtpm-culling on)\n";
        gCulling = false;
    }

    if (args.photonPaths > 0)
        p.num_photon_paths = args.photonPaths;
    if (args.maxDepth > 0)
        p.max_depth = args.maxDepth;
    if ((gAccel == ACCEL_SPPM_LEPSKI || gAccel == ACCEL_RTPM)
        && p.max_depth > PACKED_PHOTON_MAX_DEPTH + 1) {
        std::cerr << "--accel " << accelName(gAccel) << " keeps a photon's bounce index in six"
                     " bits: --depth up to " << (PACKED_PHOTON_MAX_DEPTH + 1) << "\n";
        return 2;
    }
    p.seed = args.seed;
    if (args.skyIntensity >= 0.f)
        p.sky_intensity = args.skyIntensity;
    if (args.skyPhotons >= 0)
        p.emit_sky_photons = args.skyPhotons;

    uploadSceneBuffers(scene, state);

    buildAccel(state);
    createModule(state);
    createProgramGroups(state);
    createPipeline(state);
    createSBT(state);

    p.handle = state.gasHandle;
    p.triangles = reinterpret_cast<Triangle*>(state.dTriangles);
    p.materials = reinterpret_cast<Material*>(state.dMaterials);


    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dParams), sizeof(Params)));

    // NEE light list
    p.lights = reinterpret_cast<EmissiveTriangle*>(state.dLights);
    p.num_lights = (int)state.hostLights.size();
    p.total_light_area = 0.f;
    for (const auto& lt : state.hostLights)
        p.total_light_area += lt.area;

    // How to split photon paths between the emissive triangles and the sky.
    p.sky_select_prob = computeSkySelectProb(p, state.hostLights);
    std::cout << "Photon path split: " << int(100.f * p.sky_select_prob)
              << "% sky / " << int(100.f * (1.f - p.sky_select_prob)) << "% area lights\n";


    if(args.offline) {
        if (!renderOffline(state, p, args, args.outputFile)) {
            std::cerr << "Offline rendering failed.\n";
            return 1;
        }
    } else {
        if (!renderRealtime(cam, state, p)) {
            std::cerr << "Realtime rendering failed.\n";
            return 1;
        }
    }

    // Cleanups
    lepskiDestroy();
    rtpmDestroy();
    cudaFree(p.photon_map);
    cudaFree(p.photon_count);
    cudaFree(reinterpret_cast<void*>(state.dParams));
    if (state.dLights)
        cudaFree(reinterpret_cast<void*>(state.dLights));
    cudaFree(reinterpret_cast<void*>(state.drg_photon));
    cudaFree(reinterpret_cast<void*>(state.drg_camera));
    cudaFree(reinterpret_cast<void*>(state.drg_splat));
    cudaFree(reinterpret_cast<void*>(state.drg_stochsplat));
    for (int i = 0; i < 5; ++i)
        cudaFree(reinterpret_cast<void*>(state.drg_fused[i]));
    cudaFree(reinterpret_cast<void*>(state.drg_lightvis));
    cudaFree(state.dUsedLevels);
    cudaFree(state.dDepositCounts);
    cudaFree(state.dCullMask);
    cudaFree(state.dCullParent);
    for (auto tex : state.texObjects)
        cudaDestroyTextureObject(tex);
    for (auto arr : state.texArrays)
        cudaFreeArray(arr);
 
    return 0;
}