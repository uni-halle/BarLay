#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "missing neighborhood name, please call this script as follows:"
    echo "define_new_neighborhood.sh <NEIGHBORHOOD_NAME>"
    exit 1
fi

NEIGHBORHOOD_FILE_NAME=`dirname $0`/../src/neighborhood/$1.cu

if test -f "$NEIGHBORHOOD_FILE_NAME"; then
    echo "$NEIGHBORHOOD_FILE_NAME already exists. Please choose another neighborhood name."
    exit 1
fi

cat << EOF > $NEIGHBORHOOD_FILE_NAME
#include "../params.hu"
#include "../types.hu"
#include "../d_synth.cu"

namespace neighborhood {

    // DEFINE DATA TYPE FOR NEIGHBORHOOD HERE
    using Neighborhood = ?;

    __device__ __forceinline__ void load(
        uint32_t x,
        uint32_t y,
        layout::Layout *layout,
        barcodes::ScheduleSet *schedules,
        Neighborhood* neighbors
    ) {
        // EDIT HERE
    }

    __device__ __forceinline__ uint16_t nquality(
        uint32_t x,
        uint32_t y,
        barcodes::Schedule *candidate,
        barcodes::ScheduleSet *schedules,
        Neighborhood* neighbors
    ) {
        uint16_t quality = 0;

        // EDIT HERE
        
       return quality;
    }
}
EOF

echo "new neighborhood file with boilerplate created for $1"