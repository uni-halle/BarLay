#!/bin/bash

NEIGHBORHOOD=$5
if [ "$#" -ne 5 ]; then
    if [ "$#" -ne 4 ]; then
        echo "incorrect number of arguments ($#), please call this script as follows:"
        echo "build_greedy.sh <NUMBER_OF_BARCODES> <BARCODE_LENGTH> <ROW_COUNT> <COL_COUNT> [ <NEIGHBORHOOD_NAME> ]"
        exit 1
    fi
    echo "defaulting to 'n8'-neighborhood"
    NEIGHBORHOOD="n8"
fi

cat << EOF > `dirname $0`/../src/params.hu
#ifndef _PARAMS_
#define _PARAMS_

#define NUMBER_OF_BARCODES $1
#define BARCODE_LENGTH $2
#define ROW_COUNT $3
#define COL_COUNT $4

#define SYNTH_SCHEDULE_CHUNKS (BARCODE_LENGTH) + 7 / 8

#endif
EOF

cat << EOF > `dirname $0`/../src/neighborhood.cu
#ifndef _NEIGHBORHOOD_
#define _NEIGHBORHOOD_

#include "neighborhood/$NEIGHBORHOOD.cu"

#endif
EOF

nvcc -Xptxas -O3 --debug --ptxas-options=-v -o ./bin/barlay ./src/main.cu