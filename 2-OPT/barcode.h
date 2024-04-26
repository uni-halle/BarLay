//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_BARCODE_H
#define INC_2OPT_BARCODE_H

#include <string>
#include <cassert>

struct barcode {

    static const int BARCODE_LENGTH = 34;
    char nucleotides[BARCODE_LENGTH];

public:

    barcode(const std::string &seq) {
        assert(seq.size() == BARCODE_LENGTH);
        seq.copy(nucleotides, BARCODE_LENGTH);
    }

    char operator[](int i) const {
        return nucleotides[i];
    }

};

#endif //INC_2OPT_BARCODE_H
