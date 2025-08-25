#include "config.cuh"
#include "neighborhood.cuh"
#include "algorithm.cuh"
#include "index_pool.cuh"

#include <cstdio>
#include <iostream>

void checkForError(const char *function, int line)
{
    cudaDeviceSynchronize();

    const cudaError_t err = cudaGetLastError();
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
    __global__ void grid_reduction(
        const uint16_t x,
        const uint16_t y,
        const layout::Layout * __restrict__ layout,
        const barcodes::ScheduleSet * __restrict__ schedules,
        const index_pool::Pool * __restrict__ pool,
        ResultBlock * __restrict__ result_block
    ) {
        uint32_t blockOffset = THREADS_PER_THREAD_BLOCK * blockIdx.x;
        const uint32_t threadId = threadIdx.x;

        if (blockOffset >= pool->size) {
            if (threadId == 0)
                (*result_block)[blockIdx.x].quality = USHRT_MAX;

            return;
        }

        __shared__ neighborhood::Neighborhood neighbors;
        __shared__ BarcodeQuality qualities[THREADS_PER_THREAD_BLOCK];

        qualities[threadId].indexPoolHandle = blockOffset + threadId;
        qualities[threadId].quality = USHRT_MAX;
        
        if (threadId == 0)
            neighborhood::load(x, y, layout, schedules, &neighbors);

        __syncthreads();

        // calculate quality of each barcode
        while (blockOffset + threadId < pool->size) {
            barcodes::Schedule candidate = (*schedules)[pool->indices[blockOffset + threadId]];

            const uint16_t quality = neighborhood::nquality(x, y, &candidate, &neighbors);

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
        const uint16_t x,
        const uint16_t y,
        layout::Layout * __restrict__ layout,
        index_pool::Pool * __restrict__ pool,
        const ResultBlock * __restrict__ result_block
    ) {
        const uint32_t threadId = threadIdx.x;

        __shared__ BarcodeQuality qualities[THREAD_BLOCKS];

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
            layout->positions[x][y].i_schedule = index_pool::poll(pool, qualities[0].indexPoolHandle);
        }
    }

    __global__ void process_row(
        const uint16_t x,
        layout::Layout * __restrict__ layout,
        barcodes::ScheduleSet * __restrict__ schedules,
        index_pool::Pool * __restrict__ pool,
        ResultBlock * __restrict__ result_block
    ) {
        for (uint16_t y = 0; y < COL_COUNT; y++) {
            grid_reduction<<<THREAD_BLOCKS, THREADS_PER_THREAD_BLOCK>>>(x, y, layout, schedules, pool, result_block);
            block_reduction<<<1, THREAD_BLOCKS>>>(x, y, layout, pool, result_block);
        }
    }

    void optimize(const barcodes::ScheduleSet * __restrict__ schedules, layout::Layout * __restrict__ result) {
        barcodes::ScheduleSet *device_schedules;
        cudaMalloc(reinterpret_cast<void **>(&device_schedules), sizeof(barcodes::ScheduleSet));
        cudaMemcpy(device_schedules, schedules, sizeof(barcodes::ScheduleSet), cudaMemcpyHostToDevice);

        layout::Layout *device_layout;
        cudaMalloc(reinterpret_cast<void **>(&device_layout), sizeof(layout::Layout));

        ResultBlock *device_result_block;
        cudaMalloc(reinterpret_cast<void **>(&device_result_block), sizeof(ResultBlock));

        index_pool::Pool *device_index_pool;
        cudaMalloc(reinterpret_cast<void **>(&device_index_pool), sizeof(index_pool::Pool));
        index_pool::prepare<<<1,1>>>(device_index_pool);

        int finished = 0;
        int percent = -1;
        for (uint16_t x = 0; x < ROW_COUNT; x++) {
            process_row<<<1, 1>>>(x, device_layout, device_schedules, device_index_pool, device_result_block);
            checkForError(__func__, __LINE__);

            finished += COL_COUNT;
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

        fprintf(stderr, "\nfinished optimization\n");

        cudaMemcpy(result, device_layout, sizeof(layout::Layout), cudaMemcpyDeviceToHost);

        cudaFree(device_schedules);
        cudaFree(device_layout);
        cudaFree(device_result_block);
        cudaFree(device_index_pool);
    }
}