//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_INPUT_LAYOUT_H
#define INC_2OPT_INPUT_LAYOUT_H

#include "layout.h"
#include "barcode.h"
#include <vector>

namespace barcode_layout {

    struct input_layout : public layout {

        explicit input_layout(const std::vector<barcode> &barcodes)
                : layout(barcodes.size()) {

            for (position ij = 0; ij < barcodes.size(); ij++)
                order[ij] = ij;

            for (barcode_index k = 0; k < barcodes.size(); k++)
                position_of_barcode[k] = k;
        }

    };

}

#endif //INC_2OPT_RANDOM_LAYOUT_H
