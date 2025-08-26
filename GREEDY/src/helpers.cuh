#ifndef _D_SYNTH_
#define _D_SYNTH_

#include "barcodes.cuh"

#include <cstdint>

namespace helpers {

    __device__ __forceinline__ uint16_t d_synth(
        const barcodes::Schedule * __restrict__ s1,
        const barcodes::Schedule * __restrict__ s2
        )
    {
        uint16_t distance = 0;

        for (size_t i_chunk = 0; i_chunk < SYNTH_SCHEDULE_CHUNKS; i_chunk++)
            distance += __popc(s1->chunks[i_chunk] ^ s2->chunks[i_chunk]);

        return distance;
    }

    __device__ __host__ __forceinline__ bool index_in_range(const uint16_t index, const uint16_t lo, const uint16_t hi) {
        return static_cast<uint16_t>(index - lo) <= static_cast<uint16_t>(hi - lo);
    }
}

#endif