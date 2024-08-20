//
// Created by steffen on 08.02.24.
//

#ifndef INC_2OPT_SYNTHESIS_DISTANCE_HOST_H
#define INC_2OPT_SYNTHESIS_DISTANCE_HOST_H

#include "synthesis_schedule.h"
#include <bit>
#include <cstdint>

int synthesis_distance_host(
        const synthesis_schedule &s1,
        const synthesis_schedule &s2) {

    int dist = 0;

    for (int i = 0; i < synthesis_schedule::MAX_CHUNKS; i++) {
        dist += std::__popcount(s1.chunks[i] ^ s2.chunks[i]);
    }

    return dist;
}

#endif //INC_2OPT_SYNTHESIS_DISTANCE_HOST_H
