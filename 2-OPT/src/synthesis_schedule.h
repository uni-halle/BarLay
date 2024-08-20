//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_SYNTHESIS_SCHEDULE_H
#define INC_2OPT_SYNTHESIS_SCHEDULE_H

#include <cstdint>
#include <vector>
#include "barcode.h"
#include <cstring>

struct synthesis_schedule {

    /**********************************************************************************************
     * A synthesis cycle consists of several synthesis batches.
     *
     * For example, the barcode ATGCTT needs 4 synthesis batches
     *
     *   ACGT | ACGT | ACGT | ACGT
     *   1001 | 0010 | 0101 | 0001
     *
     * and has thus the synthesis cycle 1001001001010001.
     *
     * We group 8 batches into one chunk of 32 Bit length.
     *********************************************************************************************/

    const static int MAX_CHUNKS = 4;
    uint32_t chunks[MAX_CHUNKS];

public:
    synthesis_schedule() = default;

    /**
     * Convert a string of zeros and ones to a synthesis schedule.
     * @param str String of zeros and ones.
     */
    synthesis_schedule(const std::string& str) {

        assert(str.size() <= MAX_CHUNKS * 32);
        memset(chunks, 0, MAX_CHUNKS * sizeof(uint32_t));

        // for each chunk
        for (int j = 0; j < MAX_CHUNKS; j++) {

            // construct the j-th chunk
            uint32_t j_th_chunk = 0;

            // for each character associated to the j-th chunk
            for (int i = 0; i < 32 && j * 32 + i < str.size(); i++) {
                assert(j*32+i < str.size());
                const char c = str[j * 32 + i];
                assert(c == '0' || c == '1');
                const int k = c - '0';
                assert(k == 0 || k == 1);
                j_th_chunk <<= 1;
                j_th_chunk += k;
            }

            chunks[j] = j_th_chunk;
        }
    }

    /**
     * Compute the synthesis schedule associated to some barcode.
     * @param b Barcode.
     */
    synthesis_schedule(const barcode& b) {

        memset(chunks, 0, MAX_CHUNKS * sizeof(uint32_t));

        int barcode_index = 0;
        for (int j = 0; barcode_index < barcode::BARCODE_LENGTH; j++) {

            uint32_t current_chunk = 0;

            // for the 8 batches of the j-th chunk
            for (int i = 0; i < 8; i++) {

                // construct the i-th batch of the j-th chunk
                char synthesis_order[] = {'A', 'C', 'G', 'T'};
                for (char c : synthesis_order) {
                    current_chunk <<= 1;
                    if (barcode_index < barcode::BARCODE_LENGTH && b[barcode_index] == c) {
                        current_chunk++; // set right-most bit to one
                        barcode_index++;
                    }
                }
            }

            assert(j < MAX_CHUNKS);
            chunks[j] = current_chunk;
        }
    }
};

#endif //INC_2OPT_SYNTHESIS_SCHEDULE_H