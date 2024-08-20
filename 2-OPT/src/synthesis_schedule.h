//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_SYNTHESIS_SCHEDULE_H
#define INC_2OPT_SYNTHESIS_SCHEDULE_H

#include <cstdint>
#include <vector>
#include "barcode.h"
#include <cstring>
#include <sstream>

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
    unsigned length = 0;

public:
    synthesis_schedule() = default;

    /**
     * Convert a string of zeros and ones to a synthesis schedule.
     * @param str String of zeros and ones.
     */
    synthesis_schedule(const std::string& str) {

        assert(str.size() <= MAX_CHUNKS * 32);
        memset(chunks, 0, MAX_CHUNKS * sizeof(uint32_t));
        length = str.size();

        // for each chunk
        for (int j = 0; j < MAX_CHUNKS; j++) {

            // construct the j-th chunk
            uint32_t j_th_chunk = 0;

            // for each character associated to the j-th chunk
            int i = 0;
            for (; i < 32 && j * 32 + i < str.size(); i++) {
                const char c = str[j * 32 + i];
                const int k = c - '0';
                assert(c == '0' || c == '1');
                assert(k == 0 || k == 1);
                j_th_chunk <<= 1;
                j_th_chunk += k;
            }

            // fill last chunk with zeros
            j_th_chunk <<= (32 - i);

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
            int i = 0;
            for (; i < 8 && barcode_index < barcode::BARCODE_LENGTH; i++) {

                // construct the i-th batch of the j-th chunk
                char synthesis_order[] = {'A', 'C', 'G', 'T'};
                for (char c : synthesis_order) {
                    current_chunk <<= 1;
                    length++;
                    if (barcode_index < barcode::BARCODE_LENGTH && b[barcode_index] == c) {
                        current_chunk++; // set right-most bit to one
                        barcode_index++;
                    }
                }
            }

            // fill last chunk with zeros
            current_chunk <<= (8-i)*4;

            assert(j < MAX_CHUNKS);
            chunks[j] = current_chunk;
        }
    }

    /**
     * Test whether the synthesis schedule contains a '1' in synthesis round i.
     * @param round Round counter (starting with 0).
     * @return
     */
    bool is_active(unsigned round) const {
        assert(round < length);
        unsigned k = round / 32;
        unsigned r = round - 32 * k; // r = round % 32
        assert(k < MAX_CHUNKS);
        assert(r < 32);
        assert(32*k+r == round);
        bool res = (chunks[k] >> (31-r)) & 1;
        return res;
    }
};

inline std::ostream& operator<<(std::ostream& os, const synthesis_schedule& s) {
    std::stringstream ss;
    for(unsigned j=0; j<s.length; j++)
        ss << s.is_active(j);
    return os << ss.str();
}


#endif //INC_2OPT_SYNTHESIS_SCHEDULE_H
