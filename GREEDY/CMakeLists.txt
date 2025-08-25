cmake_minimum_required(VERSION 4.0)
project(GREEDY CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(CUDAToolkit REQUIRED)

include_directories(src)

add_executable(GREEDY
        src/algorithm.cu
        src/barcodes.cu
        src/index_pool.cu
        src/layout.cu
        src/main.cu
)



# === Tunables (overridable with -D ...) ===
set(ROW_COUNT 1024 CACHE STRING "Number of layout rows")
set(COL_COUNT 768 CACHE STRING "Number of layout cols")
set(NUMBER_OF_SCHEDULES "1024*768" CACHE STRING "Number of schedules provided")
set(SCHEDULE_LENGTH "34*4" CACHE STRING "Schedule length")

# Generate a config header with these values
configure_file(
        ${CMAKE_SOURCE_DIR}/src/config.cuh.in
        ${CMAKE_BINARY_DIR}/generated/config.cuh
        @ONLY
)
target_include_directories(GREEDY PRIVATE ${CMAKE_BINARY_DIR}/generated)