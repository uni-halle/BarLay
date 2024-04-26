//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_RANDOM_LAYOUT_H
#define INC_2OPT_RANDOM_LAYOUT_H

#include "layout.h"
#include "barcode.h"
#include "input_layout.h"
#include <vector>
#include <algorithm>
#include <random>

namespace barcode_layout {

    struct random_layout : public input_layout {

        explicit random_layout(const std::vector<barcode> &barcodes)
                : input_layout(barcodes) {

            // randomly permute the indices on the array
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(order.begin(), order.end(), g);

            for (position k = 0; k < barcodes.size(); k++)
                position_of_barcode[order[k]] = k;
        }

    };

}

#endif //INC_2OPT_RANDOM_LAYOUT_H
