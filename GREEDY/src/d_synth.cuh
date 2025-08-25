#ifndef _D_SYNTH_
#define _D_SYNTH_

#include "barcodes.cuh"

#include <cstdint>

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

#endif