set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.3/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++")

cmake_minimum_required(VERSION 3.18)
project(2opt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

add_executable(2opt main.cu
        barcode.h
        synthesis_schedule.h
        synthesis_distance_device.cuh
        layout.h
        random_layout.h
        layout_cost.h
        synthesis_distance_host.h
        barcode_index.h
        local_search_host.h
        direction.h
        input_layout.h
)

set_target_properties(2opt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
