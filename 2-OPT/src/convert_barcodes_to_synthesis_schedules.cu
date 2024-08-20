//
// Created by agkya on 20.08.24.
//

#include <iostream>
#include "read_barcodes.h"
#include "synthesis_schedule.h"

using namespace barcode_layout;

int main(int argc, char** argv) {

    if(argc != 2) {
        std::cout << "Parameters: <barcode_file>" << std::endl;
        return -1;
    }

    // read barcodes
    auto barcodes = read_barcodes(argv[1]);

    // for each barcode
    for(const barcode& b : barcodes) {
        std::cout << synthesis_schedule(b) << std::endl;
    }

    return 0;
}
