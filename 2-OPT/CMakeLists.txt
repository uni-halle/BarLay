cmake_minimum_required(VERSION 3.18)
project(2opt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 60;80)

# Finden der OpenMP-Bibliothek
find_package(OpenMP REQUIRED)

add_executable(2opt src/main.cu
        src/barcode.h
        src/synthesis_schedule.h
        src/synthesis_distance_device.cuh
        src/layout.h
        src/random_layout.h
        src/layout_cost.h
        src/synthesis_distance_host.h
        src/barcode_index.h
        src/local_search_host.h
        src/direction.h
        src/input_layout.h
        src/read_barcodes.h
        src/read_barcodes.h
)

# add hidden executable that uses synthesis schedules as input
add_executable(2opt_syn_sched EXCLUDE_FROM_ALL
        src/main_syn_sched.cu
        src/barcode.h
        src/synthesis_schedule.h
        src/synthesis_distance_device.cuh
        src/layout.h
        src/random_layout.h
        src/layout_cost.h
        src/synthesis_distance_host.h
        src/barcode_index.h
        src/local_search_host.h
        src/direction.h
        src/input_layout.h
        src/read_barcodes.h
        src/read_barcodes.h
)


set_target_properties(2opt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(convert_barcodes EXCLUDE_FROM_ALL
        src/barcode.h
        src/convert_barcodes_to_synthesis_schedules.cu
)