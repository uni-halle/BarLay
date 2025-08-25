#ifndef _D_SYNTH_
#define _D_SYNTH_

#include "barcodes.cuh"

__device__ int d_synth(const barcodes::Schedule *s1, const barcodes::Schedule *s2);

#endif