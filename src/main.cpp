#include "params.h"
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
#include <fstream>
#include <iostream>
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
static constexpr size_t kMaxPhotonSlots = 32'000'000;   // 32M * 40B = 1.28 GB


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
    OptixProgramGroup gatherRaygenGroup = nullptr;

    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixTraversableHandle gasHandle = 0;

    CUdeviceptr dGasOutputBuffer = 0;
    CUdeviceptr dVertices = 0;
    CUdeviceptr dTriangles = 0;
    CUdeviceptr dMaterials = 0;
    CUdeviceptr dLights = 0; // EmissiveTriangle array
    CUdeviceptr dParams = 0;

    CUdeviceptr drg_photon = 0;
    CUdeviceptr drg_gather = 0;

    CUdeviceptr dGridCellStart = 0;
    CUdeviceptr dGridCellCount = 0;
    CUdeviceptr dGridPhotonIds = 0;

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
};

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

// OptiX context
void createContext(RendererState& state)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions opt = {};
    opt.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
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
// 5. gather raygen

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

    // Gather raygen
    OptixProgramGroupDesc gatherRd = {};
    gatherRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    gatherRd.raygen.module = state.ptxModule;
    gatherRd.raygen.entryFunctionName = "__raygen__gather";
    OPTIX_CHECK(optixProgramGroupCreate(state.context, &gatherRd, 1, &pgo,
        log, &ls, &state.gatherRaygenGroup));

    // Light visualization raygen
    OptixProgramGroupDesc lightvisRd = {};
    lightvisRd.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    lightvisRd.raygen.module = state.ptxModule;
    lightvisRd.raygen.entryFunctionName = "__raygen__lightvis";
    OptixProgramGroupOptions opts = {};
    OPTIX_CHECK(optixProgramGroupCreate(
        state.context, &lightvisRd, 1, &pgo, log, &ls, &state.lightvisRaygenGroup));
}

// Pipeline
void createPipeline(RendererState& state)
{
    // Every program group whose SBT record can be launched must be linked into
    // the pipeline — __raygen__lightvis was missing here, so "Photon Tracing
    // Only" was launching a record from an unlinked group.
    OptixProgramGroup groups[] = {
        state.raygenGroup,
        state.missGroup,
        state.shadowMissGroup,
        state.hitGroup,
        state.photonRaygenGroup,
        state.gatherRaygenGroup,
        state.lightvisRaygenGroup
    };

    OptixPipelineLinkOptions lo = {};
    lo.maxTraceDepth = 8;

    char log[2048];
    size_t ls = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        state.context,
        &state.pipelineCompileOptions,
        &lo,
        groups, (uint32_t)std::size(groups),
        log, &ls,
        &state.pipeline));
    if (ls > 1)
        std::cerr << "[OptiX Pipeline] " << log << "\n";

    OptixStackSizes ss = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygenGroup, &ss, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.missGroup, &ss, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.shadowMissGroup, &ss, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.hitGroup, &ss, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.photonRaygenGroup, &ss, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.gatherRaygenGroup, &ss, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.lightvisRaygenGroup, &ss, state.pipeline));

    uint32_t fromTraversal, fromState, continuation;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &ss,
        lo.maxTraceDepth, // maxTraceDepth
        0, 0, // CC depth, DC depth
        &fromTraversal, &fromState, &continuation));

    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        fromTraversal,
        fromState,
        continuation,
        2 // IAS + GAS depth
        ));
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

    // Gather raygen SBT record
    SbtRecord<RayGenData> rgGather = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.gatherRaygenGroup, &rgGather));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.drg_gather), sizeof(rgGather)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.drg_gather),
        &rgGather, sizeof(rgGather), cudaMemcpyHostToDevice));

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
}

void buildPhotonGrid(Params& params,
    CUdeviceptr& dGridCellStart,
    CUdeviceptr& dGridCellCount,
    CUdeviceptr& dGridPhotonIds)
{
    // get photon count from GPU
    int storedCount = 0;
    CUDA_CHECK(cudaMemcpy(&storedCount, params.photon_count,
        sizeof(int), cudaMemcpyDeviceToHost));
    storedCount = std::min(storedCount, params.photon_capacity);

    if (storedCount == 0) {
        params.use_grid = 0;
        return;
    }

    std::vector<Photon> hostPhotons(storedCount);
    CUDA_CHECK(cudaMemcpy(hostPhotons.data(), params.photon_map,
        storedCount * sizeof(Photon),
        cudaMemcpyDeviceToHost));

    // AABB bounds of all photons
    float3 bmin = hostPhotons[0].pos;
    float3 bmax = hostPhotons[0].pos;
    for (int i = 1; i < storedCount; ++i) {
        const float3& p = hostPhotons[i].pos;
        bmin.x = std::min(bmin.x, p.x);
        bmax.x = std::max(bmax.x, p.x);
        bmin.y = std::min(bmin.y, p.y);
        bmax.y = std::max(bmax.y, p.y);
        bmin.z = std::min(bmin.z, p.z);
        bmax.z = std::max(bmax.z, p.z);
    }
    // one cell extra in each direction to avoid boundary issues
    float r = params.gather_radius;
    bmin.x -= r;
    bmin.y -= r;
    bmin.z -= r;
    bmax.x += r;
    bmax.y += r;
    bmax.z += r;

    // The grid is binned at `cs`, normally the gather radius itself.  The axis
    // count is capped, and the cap has to be absorbed by *growing the cell*
    // rather than by clamping dims alone: the host below clamps an out-of-range
    // photon into the edge cell, while the device derives its index from
    // grid.cell_size and skips anything outside dims, so if the two disagree the
    // gather silently loses photons.  Growing cs instead keeps them consistent
    // and stays correct, because the device's 3x3x3 walk covers the search
    // sphere for any cs >= r (it is only slower, more photons per cell to test).
    //
    // This matters now that the PPM schedule shrinks r every frame: cell count
    // grows as r^-3, so a factor-3 radius reduction is 27x the cells.
    const int kMaxCellsPerAxis = 1024;
    const float extX = bmax.x - bmin.x, extY = bmax.y - bmin.y, extZ = bmax.z - bmin.z;
    float cs = r;
    cs = std::max(cs, extX / kMaxCellsPerAxis);
    cs = std::max(cs, extY / kMaxCellsPerAxis);
    cs = std::max(cs, extZ / kMaxCellsPerAxis);

    // grid dimensions (number of cells in each direction)
    int3 dims;
    dims.x = std::min(kMaxCellsPerAxis, std::max(1, (int)std::ceil(extX / cs)));
    dims.y = std::min(kMaxCellsPerAxis, std::max(1, (int)std::ceil(extY / cs)));
    dims.z = std::min(kMaxCellsPerAxis, std::max(1, (int)std::ceil(extZ / cs)));

    int totalCells = dims.x * dims.y * dims.z;
    std::cout << "[PhotonGrid] dims = " << dims.x << "x" << dims.y << "x" << dims.z
              << "  cells = " << totalCells
              << "  photons = " << storedCount << "\n";

    // Count photons in each cell
    auto cellIndex = [&](float3 pos) -> int {
        int ix = (int)std::floor((pos.x - bmin.x) / cs);
        int iy = (int)std::floor((pos.y - bmin.y) / cs);
        int iz = (int)std::floor((pos.z - bmin.z) / cs);
        ix = std::max(0, std::min(ix, dims.x - 1));
        iy = std::max(0, std::min(iy, dims.y - 1));
        iz = std::max(0, std::min(iz, dims.z - 1));
        return iz * dims.y * dims.x + iy * dims.x + ix;
    };

    std::vector<int> cellCount(totalCells, 0);
    for (int i = 0; i < storedCount; ++i)
        cellCount[cellIndex(hostPhotons[i].pos)]++;

    // Prefix sum -> cell_start
    std::vector<int> cellStart(totalCells, 0);
    for (int c = 1; c < totalCells; ++c)
        cellStart[c] = cellStart[c - 1] + cellCount[c - 1];

    // Filling grid_photon_ids
    std::vector<int> gridPhotonIds(storedCount);
    std::vector<int> insertCursor(cellStart); // копия для записи

    for (int i = 0; i < storedCount; ++i) {
        int c = cellIndex(hostPhotons[i].pos);
        gridPhotonIds[insertCursor[c]++] = i;
    }

    // Upload to GPU.  This runs once per frame now, so the buffers are kept
    // and only grown when they no longer fit.
    // One capacity per buffer — sharing a counter between two pointers would let
    // the second one skip a realloc it actually needed.
    static size_t startCapacity = 0;
    static size_t countCapacity = 0;
    static size_t idCapacity = 0;

    auto ensure = [](CUdeviceptr& ptr, size_t& capacity, size_t needed) {
        if (ptr && capacity >= needed)
            return;
        if (ptr)
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
        capacity = needed + needed / 4; // a little headroom
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), capacity * sizeof(int)));
    };

    ensure(dGridCellStart, startCapacity, (size_t)totalCells);
    ensure(dGridCellCount, countCapacity, (size_t)totalCells);
    ensure(dGridPhotonIds, idCapacity, (size_t)storedCount);

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dGridCellStart),
        cellStart.data(), totalCells * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dGridCellCount),
        cellCount.data(), totalCells * sizeof(int),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dGridPhotonIds),
        gridPhotonIds.data(), storedCount * sizeof(int),
        cudaMemcpyHostToDevice));

    // Update params
    params.grid.aabb_min = bmin;
    params.grid.aabb_max = bmax;
    params.grid.dims = dims;
    params.grid.cell_size = cs;
    params.grid.cell_start = reinterpret_cast<int*>(dGridCellStart);
    params.grid.cell_count = reinterpret_cast<int*>(dGridCellCount);
    params.grid.grid_photon_ids = reinterpret_cast<int*>(dGridPhotonIds);
    params.use_grid = 1;
}

static bool saveEXR(const std::string& filename,
                    const std::vector<float3>& hdr,
                    int width, int height) {
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
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // или FLOAT
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

void launchPathTracing(RendererState& state, Params& p, CUstream& stream) {
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                                &p, sizeof(Params),
                                cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                            state.dParams, sizeof(Params),
                            &state.sbt, p.width, p.height, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static bool tracePhotonsOnce(RendererState& state, Params& p, CUstream stream, int& deposits)
{
    CUDA_CHECK(cudaMemsetAsync(p.photon_count, 0, sizeof(int), stream));

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
int launchPhotonTracing(RendererState& state, Params& p, CUstream& stream)
{
    int deposits = 0;
    if (!tracePhotonsOnce(state, p, stream, deposits)) {
        size_t need = size_t(deposits * 1.2) + 1024;
        std::cout << "[Photons] saturated (" << deposits << " > " << p.photon_capacity
                  << "), growing to " << need << " and retracing\n";

        CUDA_CHECK(cudaFree(p.photon_map));
        p.photon_capacity = int(std::min<size_t>(need, kMaxPhotonSlots));
        CUDA_CHECK(cudaMalloc(&p.photon_map, size_t(p.photon_capacity) * sizeof(Photon)));

        if (!tracePhotonsOnce(state, p, stream, deposits))
            std::cerr << "[Photons] STILL saturated — lower num_photon_paths\n";
    }
    buildPhotonGrid(p, state.dGridCellStart, state.dGridCellCount, state.dGridPhotonIds);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.dParams), &p, sizeof(Params),
                          cudaMemcpyHostToDevice));
    return std::min(deposits, p.photon_capacity);
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

void launchPhotonGathering(RendererState& state, Params& p, CUstream& stream) {
    OptixShaderBindingTable sbtGather = state.sbt;
    sbtGather.raygenRecord = state.drg_gather;

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                              &p, sizeof(Params),
                              cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                            state.dParams, sizeof(Params),
                            &sbtGather, p.width, p.height, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));
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

static bool renderOffline(RendererState& state, Params& p, const std::string& outputFile)
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

    p.frame_buffer = dFrame;
    p.accum_buffer = reinterpret_cast<float3*>(dAccum);
    p.frame_index  = 0;

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

    for (int f = 0; f < p.offline_frames; ++f) {

        if (p.render_mode == 0) {
            // Path Tracing launch
            launchPathTracing(state, p, stream);
        } else if (p.render_mode == 1) {
            // Probabilistic PPM: shrink the radius for this frame *before* the
            // map is traced, because buildPhotonGrid bins at gather_radius.
            updatePPMRadius(p);

            // Pass 1 — Photon tracing.  A fresh map every frame: reusing one
            // map would make every frame share the same density-estimate
            // error, so the blotches would stay put no matter how long it ran.
            // It is also what makes the frames independent estimates, which is
            // the assumption the PPM radius schedule is derived under.
            launchPhotonTracing(state, p, stream);

            // Pass 2 — Gathering
            launchPhotonGathering(state, p, stream);
        } else if (p.render_mode == 2) {
            launchPhotonTracing(state, p, stream);
            launchLightVis(state, p, stream, dLightvisBuf);
        }

        p.frame_index++;
        std::cout << "\rOffline frame " << (f + 1) << "/" << p.offline_frames
                  << "  r = " << p.gather_radius << std::flush;
    }

    std::cout << "\n";

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

    bool ok = saveEXR(outputFile, hostAccum, W, H);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dFrame));
    CUDA_CHECK(cudaFree(dAccum));
    if (dLightvisBuf)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dLightvisBuf)));

    return ok;
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

    p.frame_index = 0;

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bool photons_valid = false;
    bool retracePhotons = true; // a fresh map per frame is what makes it converge
    int lastDeposits = 0;

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
            resetAccum = true;
        }

        if (ImGui::CollapsingHeader("Sampling")) {
            if (ImGui::SliderInt("Samples / pixel", &p.samples_per_pixel, 1, 16))
                resetAccum = true;
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

            bool adaptive = p.adaptive_radius != 0;
            if (!p.use_ppm && ImGui::Checkbox("Adaptive radius", &adaptive)) {
                p.adaptive_radius = adaptive ? 1 : 0;
                resetAccum = true;
            }
            if (!p.use_ppm && adaptive) {
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

            ImGui::Checkbox("Retrace every frame", &retracePhotons);
            ImGui::SameLine();
            ImGui::TextDisabled("(off = noise stops converging)");
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
            // Pass 1 — Photon tracing.  Re-traced every frame by default: the
            // accumulation buffer can only average away photon noise if each
            // frame sees an *independent* photon map.  Holding one map fixed
            // (as this used to) freezes its density-estimate error into the
            // image, which is what left the permanent blotches.
            if (retracePhotons || !photons_valid) {
                // Before the trace — the grid is binned at gather_radius.  With
                // "Retrace every frame" off the map (and so the radius) is
                // frozen, which is consistent: a schedule that shrank the radius
                // over a map that never changes would just re-filter the same
                // photons and converge to nothing.
                updatePPMRadius(p);
                lastDeposits = launchPhotonTracing(state, p, stream);
                photons_valid = true;
            }

            // Pass 2 — Gathering, or splatting the map straight onto the film
            if (p.render_mode == 1)
                launchPhotonGathering(state, p, stream);
            else
                launchLightVis(state, p, stream, dLightvisBuf);
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
    CameraFileState cam{};
    cam = loadCameraFile(args.cameraFile);

    RendererState state;
    Scene scene;
    Params p = {};

    int W = 1280, H = 720;

    p.width = W;
    p.height = H;

    // Initial camera state
    p.cam_eye = cam.eye;
    gYaw = cam.yaw;
    gPitch = cam.pitch;
    rebuildCameraVectors(p);

    createContext(state);
    std::cout << "Context created.\n";

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
    p.gather_radius = 0.03f * p.scene_radius;

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

    if (args.photonPaths > 0)
        p.num_photon_paths = args.photonPaths;
    if (args.maxDepth > 0)
        p.max_depth = args.maxDepth;
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
        if (!renderOffline(state, p, args.outputFile)) {
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
    cudaFree(p.photon_map);
    cudaFree(p.photon_count);
    cudaFree(reinterpret_cast<void*>(state.dParams));
    if (state.dLights)
        cudaFree(reinterpret_cast<void*>(state.dLights));
    cudaFree(reinterpret_cast<void*>(state.drg_photon));
    cudaFree(reinterpret_cast<void*>(state.drg_gather));
    for (auto tex : state.texObjects)
        cudaDestroyTextureObject(tex);
    for (auto arr : state.texArrays)
        cudaFreeArray(arr);
 
    return 0;
}