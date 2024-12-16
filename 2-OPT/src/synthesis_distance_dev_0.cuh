//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_SYNTHESIS_DISTANCE_DEV_0_CUH
#define INC_2OPT_SYNTHESIS_DISTANCE_DEV_0_CUH

#include "synthesis_schedule.h"

__device__ inline unsigned synthesis_distance_dev_0(
        const synthesis_schedule &s1,
        const synthesis_schedule &s2) {
    unsigned dist = 0;
    for (int i = 0; i < synthesis_schedule::MAX_CHUNKS; i++) {

        for(int j=0; j<32; j++) {
            int s1_ij = (s1.chunks[i] >> j) & 1;
            int s2_ij = (s2.chunks[i] >> j) & 1;
            if( s1_ij != s2_ij)
                dist++;
        }
    }
    return dist;
}


#endif //INC_2OPT_SYNTHESIS_DISTANCE_DEV_0_CUH
