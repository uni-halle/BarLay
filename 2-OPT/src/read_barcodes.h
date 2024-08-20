//
// Created by agkya on 20.08.24.
//

#ifndef READ_BARCODES_H
#define READ_BARCODES_H

#include <fstream>
#include "barcode.h"
#include <vector>

namespace barcode_layout {

    inline std::vector<barcode> read_barcodes(const std::string& barcode_file) {

        std::vector<barcode> barcodes;
        std::ifstream infile(barcode_file);
        std::string line;
        while (std::getline(infile, line))
            barcodes.emplace_back(line);

        return barcodes;
    }

}

#endif //READ_BARCODES_H
