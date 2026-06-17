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

// Camera state
static float gYaw = -90.f;
static float gPitch = 0.f;
static bool gCameraChanged = false;
static double gLastX = 0.0, gLastY = 0.0;
static bool gFirstMouse = true;

// Mouse callback to control camera orientation. Updates gYaw and gPitch based on mouse movement, and sets gCameraChanged to true when the camera is updated
static void mouseCallback(GLFWwindow*, double xpos, double ypos)
{
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
    float yr = gYaw * M_PI / 180.f;
    float pr = gPitch * M_PI / 180.f;

    // forward в glTF (Y-up, right-handed)
    float3 fwd = make_float3(cosf(pr) * cosf(yr), sinf(pr), cosf(pr) * sinf(yr));
    float len = sqrtf(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z);
    fwd = make_float3(fwd.x / len, fwd.y / len, fwd.z / len);

    float3 worldUp = make_float3(0.f, 1.f, 0.f);

    // right = fwd × worldUp, then normalize
    float3 right = make_float3(
        fwd.y * worldUp.z - fwd.z * worldUp.y,
        fwd.z * worldUp.x - fwd.x * worldUp.z,
        fwd.x * worldUp.y - fwd.y * worldUp.x);
    float rlen = sqrtf(right.x * right.x + right.y * right.y + right.z * right.z);
    right = make_float3(right.x / rlen, right.y / rlen, right.z / rlen);

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

    uint32_t numVertices = 0;
    uint32_t numMaterials = 0;

    std::vector<Material> hostMaterials;
    std::vector<uint32_t> sbtOffsets;

    std::vector<EmissiveTriangle> hostLights;

    // Texture tracking
    std::vector<cudaArray_t> texArrays;
    std::vector<cudaTextureObject_t> texObjects;
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

        EmissiveTriangle lt;
        lt.v0 = tri.v0;
        lt.v1 = tri.v1;
        lt.v2 = tri.v2;
        lt.emission = em;
        lt.area = area;
        lt.tri_idx = i;
        state.hostLights.push_back(lt);
    }
    std::cout << "Light list: " << state.hostLights.size() << " emissive triangles.\n";
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
}

// Pipeline
void createPipeline(RendererState& state)
{
    OptixProgramGroup groups[] = {
        state.raygenGroup,
        state.missGroup,
        state.shadowMissGroup,
        state.hitGroup,
        state.photonRaygenGroup,
        state.gatherRaygenGroup
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

// main
int main(int argc, char** argv)
{
    std::string sceneFile = (argc > 1) ? argv[1] : "scene.glb";
    RendererState state;
    Scene scene;

    GLFWwindow* window = initGL();
    createContext(state);
    std::cout << "Context created.\n";

    loadScene(sceneFile, scene, state);
    std::cout << "Materials: " << scene.materials.size()
              << ", Triangles: " << scene.triangles.size() << "\n";

    Params p = {};

    // Build emissive light list
    buildLightList(scene, state);

    const int NUM_PHOTONS = 500000;
    Photon* dPhotonMap;
    CUDA_CHECK(cudaMalloc(&dPhotonMap, NUM_PHOTONS * sizeof(Photon)));
    int* dPhotonCount;
    CUDA_CHECK(cudaMalloc(&dPhotonCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(dPhotonCount, 0, sizeof(int)));

    p.photon_map = dPhotonMap;
    p.num_photons = NUM_PHOTONS;
    p.photon_count = dPhotonCount;
    p.gather_radius = 3.0f; // tune for the scene
    p.render_mode = 1;

    uploadSceneBuffers(scene, state);

    buildAccel(state);
    createModule(state);
    createProgramGroups(state);
    createPipeline(state);
    createSBT(state);

    // Window / PBO setup
    int W = 1280, H = 720;
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
    p.width = W;
    p.height = H;
    p.handle = state.gasHandle;
    p.triangles = reinterpret_cast<Triangle*>(state.dTriangles);
    p.materials = reinterpret_cast<Material*>(state.dMaterials);

    // NEE light list
    p.lights = reinterpret_cast<EmissiveTriangle*>(state.dLights);
    p.num_lights = (int)state.hostLights.size();
    p.total_light_area = 0.f;
    for (const auto& lt : state.hostLights)
        p.total_light_area += lt.area;

    p.cam_eye = make_float3(-0.278f, -0.8f, 0.273f);
    gYaw = -90.f;
    gPitch = 0.f;
    rebuildCameraVectors(p);
    p.samples_per_pixel = 4;
    p.max_depth = 8;
    p.frame_index = 0;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.dParams), sizeof(Params)));

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bool photons_valid = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;

        handleKeys(window, p, 4.0f);
        if (gCameraChanged) {
            rebuildCameraVectors(p);
            p.frame_index = 0;
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
            gCameraChanged = false;
        }

        static bool pKeyWasDown = false;
        bool pKeyDown = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
        if (pKeyDown && !pKeyWasDown) {
            p.render_mode = 1 - p.render_mode;
            p.frame_index = 0;
            CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dAccum), 0, W * H * sizeof(float3)));
            photons_valid = false;
        }
        pKeyWasDown = pKeyDown;

        uchar4* devPtr;
        size_t sz;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPBO, stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&devPtr), &sz, cudaPBO));
        p.frame_buffer = devPtr;

        if (p.render_mode == 0) {
            // Path Tracing launch
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                &p, sizeof(p), cudaMemcpyHostToDevice, stream));
            OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                state.dParams, sizeof(Params),
                &state.sbt, // raygen = __raygen__rg
                W, H, 1));
            CUDA_CHECK(cudaStreamSynchronize(stream));

        } else {
            // Photon Mapping launch

            // Pass 1 — Photon Tracing (1D launch: NUM_PHOTONS threads, each tracing one photon)
            // Reset photon count every time
            if (!photons_valid) {
                CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(p.photon_count),
                    0, sizeof(int), stream));

                OptixShaderBindingTable sbtPhoton = state.sbt;
                sbtPhoton.raygenRecord = state.drg_photon;

                CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                    &p, sizeof(p), cudaMemcpyHostToDevice, stream));
                OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                    state.dParams, sizeof(Params),
                    &sbtPhoton,
                    p.num_photons, 1, 1)); // 1D!
                CUDA_CHECK(cudaStreamSynchronize(stream));
                photons_valid = true;
            }

            // Pass 2 — Gathering (2D launch: W×H threads, each gathering photons for one pixel)
            OptixShaderBindingTable sbtGather = state.sbt;
            sbtGather.raygenRecord = state.drg_gather;

            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.dParams),
                &p, sizeof(p), cudaMemcpyHostToDevice, stream));
            OPTIX_CHECK(optixLaunch(state.pipeline, stream,
                state.dParams, sizeof(Params),
                &sbtGather,
                W, H, 1)); // 2D as PT
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPBO, stream));
        p.frame_index++;

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayTex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        drawQuad(displayTex);
        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(reinterpret_cast<void*>(dAccum));
    cudaFree(reinterpret_cast<void*>(state.dParams));
    if (state.dLights)
        cudaFree(reinterpret_cast<void*>(state.dLights));
    cudaGraphicsUnregisterResource(cudaPBO);
    cudaFree(dPhotonMap);
    cudaFree(dPhotonCount);
    cudaFree(reinterpret_cast<void*>(state.drg_photon));
    cudaFree(reinterpret_cast<void*>(state.drg_gather));

    for (auto tex : state.texObjects)
        cudaDestroyTextureObject(tex);
    for (auto arr : state.texArrays)
        cudaFreeArray(arr);

    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &displayTex);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}