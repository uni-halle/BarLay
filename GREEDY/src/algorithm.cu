#ifndef _ALGORITHM_
#define _ALGORITHM_

#include "params.hu"
#include "types.hu"
#include "neighborhood.cu"

#include "barcodes.cu"
#include "layout.cu"

#include <stdio.h>
#include <time.h>
#include <iostream>

#define THREAD_BLOCKS 1024
#define THREADS_PER_THREAD_BLOCK 256

namespace index_pool {
    struct Pool {
        uint32_t size;
        uint32_t indices[NUMBER_OF_BARCODES];
    };

    __global__ void prepare(struct Pool *pool) {
        pool->size = NUMBER_OF_BARCODES;
        for (size_t i = 0; i < NUMBER_OF_BARCODES; i++)
            pool->indices[i] = i;
    }

    __device__ uint32_t poll(struct Pool *pool, uint32_t index) {
        // save polled index
        uint32_t polled = pool->indices[index];

        // replace polled index with index in last position
        pool->indices[index] = pool->indices[pool->size - 1];

        // reduce pool size
        pool->size--;

        return polled;
    }
}

void checkForError(const char *function, int line)
{
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
        cudaDeviceReset();

        fprintf(
            stderr,
            "cuda error (%s) caught in %s on line %d",
            cudaGetErrorString(err),
            function,
            line
        );
        std::cerr << std::flush;

        exit(-1);
    }
}

namespace algorithm {
    struct BarcodeQuality {
        uint16_t quality;
        uint32_t indexPoolHandle;
    };

    using ResultBlock = struct BarcodeQuality[THREAD_BLOCKS];

    __global__ void grid_reduction(
        uint32_t x,
        uint32_t y,
        layout::Layout *layout,
        barcodes::ScheduleSet *schedules,
        index_pool::Pool *pool,
        ResultBlock *result_block
    ) {
        uint32_t blockOffset = THREADS_PER_THREAD_BLOCK * blockIdx.x;
        uint32_t threadId = threadIdx.x;

        if (blockOffset >= pool->size) {
            if (threadId == 0)
                (*result_block)[blockIdx.x].quality = USHRT_MAX;

            return;
        }

        __shared__ neighborhood::Neighborhood neighbors;
        __shared__ struct BarcodeQuality qualities[THREADS_PER_THREAD_BLOCK];

        qualities[threadId].indexPoolHandle = blockOffset + threadId;
        qualities[threadId].quality = USHRT_MAX;

        #define MASK_OFFSET_X MASK_X / 2;
        #define MASK_OFFSET_Y MASK_Y / 2;
        
        if (threadId == 0)
            neighborhood::load(x, y, layout, schedules, &neighbors);

        __syncthreads();

        // calculate quality of each barcode
        while (blockOffset + threadId < pool->size) {
            barcodes::SynthSchedule candidate = (*schedules)[pool->indices[blockOffset + threadId]];
            
            uint16_t quality = neighborhood::nquality(x, y, &candidate, schedules, &neighbors);

            if (quality < qualities[threadId].quality) {
                qualities[threadId].quality = quality;
                qualities[threadId].indexPoolHandle = blockOffset + threadId;
            }

            blockOffset += THREAD_BLOCKS*THREADS_PER_THREAD_BLOCK;
        }

        __syncthreads();

        // run parallel reduction to find best for this block
        for (uint16_t s = THREADS_PER_THREAD_BLOCK / 2; s > 0; s >>= 1) {
            if (threadId < s && qualities[threadId + s].quality < qualities[threadId].quality)
                qualities[threadId] = qualities[threadId + s];

            __syncthreads();
        }

        // write block result to global memory
        if (threadId == 0)
            (*result_block)[blockIdx.x] = qualities[0];
    }

    __global__ void block_reduction(
        size_t x,
        size_t y,
        layout::Layout *layout,
        barcodes::ScheduleSet *schedules,
        index_pool::Pool *pool,
        ResultBlock *result_block
    ) {
        uint32_t threadId = threadIdx.x;

        __shared__ struct BarcodeQuality qualities[THREAD_BLOCKS];

        // load block results into shared memory
        qualities[threadId] = (*result_block)[threadId];
        
        __syncthreads();

        // run parallel reduction to find best for this block
        for (uint16_t s = THREAD_BLOCKS / 2; s > 0; s >>= 1) {
            if (threadId < s && qualities[threadId + s].quality < qualities[threadId].quality)
                qualities[threadId] = qualities[threadId + s];

            __syncthreads();
        }

        // position best barcode at (x,y)
        if (threadId == 0) {
            layout->positions[x][y].i_barcode = index_pool::poll(pool, qualities[0].indexPoolHandle);
        }
    }

    void optimize(const barcodes::ScheduleSet *schedules, layout::Layout *result) {
        barcodes::ScheduleSet *device_schedules;
        cudaMalloc((void **)&device_schedules, sizeof(barcodes::ScheduleSet));
        cudaMemcpy((void *)device_schedules, (void *)schedules, sizeof(barcodes::ScheduleSet), cudaMemcpyHostToDevice);

        layout::Layout *device_layout;
        cudaMalloc((void **)&device_layout, sizeof(layout::Layout));

        ResultBlock *device_result_block;
        cudaMalloc((void **)&device_result_block, sizeof(ResultBlock));

        index_pool::Pool *device_index_pool;
        cudaMalloc((void **)&device_index_pool, sizeof(index_pool::Pool));
        index_pool::prepare<<<1,1>>>(device_index_pool);

        int finished = 0;
        int percent = -1;
        for (size_t x = 0; x < ROW_COUNT; x++) {
            for (size_t y = 0; y < COL_COUNT; y++) {
                grid_reduction<<<THREAD_BLOCKS, THREADS_PER_THREAD_BLOCK>>>(x, y, device_layout, device_schedules, device_index_pool, device_result_block);

                checkForError(__func__, __LINE__);

                block_reduction<<<1, THREAD_BLOCKS>>>(x, y, device_layout, device_schedules, device_index_pool, device_result_block);

                checkForError(__func__, __LINE__);
                
                finished++;
                int newPercent = (100 * finished) / (ROW_COUNT * COL_COUNT);
                if (percent < newPercent )
                {
                    cudaDeviceSynchronize();
                    percent = newPercent;
                    
                    fprintf(stderr, "\roptimizing %*d %% [", 3, percent);

                    for (size_t i = 0; i < percent; i++)
                        fprintf(stderr, "#");
                    for (size_t i = 0; i < 100 - percent; i++)
                        fprintf(stderr, " ");
                    

                    fprintf(stderr, "]");

                    std::cerr << std::flush;
                }
            }
        }

        fprintf(stderr, "\nfinished optimization\n");

        cudaMemcpy(result, device_layout, sizeof(layout::Layout), cudaMemcpyDeviceToHost);

        cudaFree(device_schedules);
        cudaFree(device_layout);
        cudaFree(device_result_block);
        cudaFree(device_index_pool);
    }
}

#endif