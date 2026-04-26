cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc

make -j$(nproc)

./renderer