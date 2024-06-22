#ifndef _LAYOUT_
#define _LAYOUT_

#include "params.hu"
#include "types.hu"

#include <stdint.h>
#include <stdio.h>

namespace layout {
    void json(FILE *target, const barcodes::Set *barcode_set, const struct Layout *layout) {
        fputs("[\n", target);
        for (size_t y = 0; y < COL_COUNT; y++) {
            fputs("\t[", target);

            for (size_t x = 0; x < ROW_COUNT; x++) {
                const struct Position *position = &layout->positions[x][y];
                const barcodes::Barcode *barcode = &(*barcode_set)[position->i_barcode];

                fputs("\"", target);

                for (size_t i = 0; i < BARCODE_LENGTH; i++)
                    fputc(barcode->nucleotides[i], target);
                    
                fputs("\"", target);

                if(x < ROW_COUNT-1) fputs(",", target);
            }

            fputs("]", target);

            if(y < COL_COUNT-1) fputs(",", target);

            fputs("\n", target);
        }

        fputs("]\n", target);
    }
    void print(FILE *target, const barcodes::Set *barcode_set, const struct Layout *layout) {
        for (size_t x = 0; x < ROW_COUNT; x++) {
            for (size_t y = 0; y < COL_COUNT; y++) {
                const struct Position *position = &layout->positions[x][y];
                const barcodes::Barcode *barcode = &(*barcode_set)[position->i_barcode];

                for (size_t i = 0; i < BARCODE_LENGTH; i++)
                    fputc(barcode->nucleotides[i], target);
                
                fputs("\n", target);
            }
        }
    }
}

#endif