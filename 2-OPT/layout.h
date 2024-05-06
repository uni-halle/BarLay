//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_LAYOUT_H
#define INC_2OPT_LAYOUT_H

#include <vector>
#include "barcode_index.h"

namespace barcode_layout {

    typedef unsigned position;

    struct layout {

        static const unsigned row_count = 768;
        static const unsigned col_count = 1024;

        // the barcode with index order[xy] is stored at position (x,y)
        std::vector<barcode_index> order;

        // for each barcode index i, position_of_barcode[i] gives i's position on the layout
        std::vector<position> position_of_barcode;

        layout(unsigned num_barcodes) : order(num_barcodes),
                                        position_of_barcode(num_barcodes, (position) -1) {}

        barcode_index get_barcode_index(position ij) const {
            assert(ij < order.size());
            return order[ij];
        }

        /**
         * Test whether some barcode is currently assigned to some layout position.
         * @param ind
         * @return
         */
        bool is_assigned(barcode_index ind) const {
            return position_of_barcode[ind] < layout::row_count * layout::col_count;
        }

        void swap_barcodes_at_position(position ij, position xy) {
            assert(ij < order.size());
            assert(xy < order.size());
            barcode_index b_ij = order[ij];
            barcode_index b_xy = order[xy];
            order[ij] = b_xy;
            order[xy] = b_ij;
            position_of_barcode[b_ij] = xy;
            position_of_barcode[b_xy] = ij;
        }

        void swap_barcodes(barcode_index b_1, barcode_index b_2) {
            position p_1 = position_of_barcode[b_1];
            position p_2 = position_of_barcode[b_2];
            order[p_1] = b_2;
            order[p_2] = b_1;
            position_of_barcode[b_1] = p_2;
            position_of_barcode[b_2] = p_1;
        }

        /**
         * Return barcode indices (assigned or unassigned) in row-major-order.
         * @return
         */
        const std::vector<barcode_index> &get_barcode_order() const {
            return order;
        }

        position get_position(barcode_index i) const {
            return position_of_barcode[i];
        }
    };


    __host__ __device__ position flat(unsigned i, unsigned j) {
        return i * layout::col_count + j;
    }
}

#endif //INC_2OPT_LAYOUT_H
