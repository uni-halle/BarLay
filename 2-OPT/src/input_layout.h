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

        input_layout(size_t barcode_count)
                : layout(barcode_count) {

            for (position ij = 0; ij < barcode_count; ij++)
                order[ij] = ij;

            for (barcode_index k = 0; k < barcode_count; k++)
                position_of_barcode[k] = k;
        }

    };

}

#endif //INC_2OPT_INPUT_LAYOUT_H
