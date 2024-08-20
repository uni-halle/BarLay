//
// Created by steffen on 15.02.24.
//

#ifndef INC_2OPT_DIRECTION_H
#define INC_2OPT_DIRECTION_H

namespace barcode_layout {

    enum direction {
        NORTH_WEST = 0,
        WEST = 1,
        SOUTH_WEST = 2,
        NORTH = NORTH_WEST + 3,
        CENTER = WEST + 3,
        SOUTH = SOUTH_WEST + 3,
        NORTH_EAST = NORTH + 3,
        SOUTH_EAST = SOUTH + 3,
        EAST = CENTER + 3
    };

}

#endif //INC_2OPT_DIRECTION_H
