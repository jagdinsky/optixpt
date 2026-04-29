cmake_minimum_required(VERSION 3.18)
project(renderer LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(OptiX_INCLUDE $ENV{HOME}/optix/include)

# Compile device.cu → PTX
add_library(device_ptx OBJECT src/device.cu)
set_target_properties(device_ptx PROPERTIES CUDA_PTX_COMPILATION ON)
target_include_directories(device_ptx PRIVATE
    ${OptiX_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
)

# Main executable
add_executable(renderer src/main.cpp)
target_include_directories(renderer PRIVATE
    ${OptiX_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(renderer
    CUDA::cuda_driver
    CUDA::cudart
)

# Copy device.ptx next to the executable after build
add_custom_command(TARGET renderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_OBJECTS:device_ptx>
    $<TARGET_FILE_DIR:renderer>/device.ptx
)