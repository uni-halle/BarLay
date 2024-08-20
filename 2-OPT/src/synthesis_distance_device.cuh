//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_SYNTHESIS_DISTANCE_DEVICE_CUH
#define INC_2OPT_SYNTHESIS_DISTANCE_DEVICE_CUH

#include "synthesis_schedule.h"

__device__ inline unsigned synthesis_distance_device(
        const synthesis_schedule &s1,
        const synthesis_schedule &s2) {
    unsigned dist = 0;
    for (int i = 0; i < synthesis_schedule::MAX_CHUNKS; i++)
        dist += __popc(s1.chunks[i] ^ s2.chunks[i]);
    return dist;
}


#endif //INC_2OPT_SYNTHESIS_DISTANCE_DEVICE_CUH
