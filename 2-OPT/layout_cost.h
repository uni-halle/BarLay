//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_LAYOUT_COST_H
#define INC_2OPT_LAYOUT_COST_H

#include "layout.h"
#include "synthesis_schedule.h"
#include "synthesis_distance_host.h"
#include <vector>

namespace barcode_layout {

    unsigned layout_cost(const std::vector<synthesis_schedule> &schedules, const layout &l) {

        unsigned cost = 0;

        // todo: improve performance

        for (unsigned i = 0; i < layout::row_count; i++) {
            for (unsigned j = 0; j < layout::col_count; j++) {

                position ij = flat(i, j);
                barcode_index b_ij = l.get_barcode_index(ij);
                auto s_ij = schedules[b_ij];

                if (i > 0)  // NORTH
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i - 1, j))]);
                if (j > 0) // WEST
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i, j - 1))]);
                if (i > 0 && j > 0) // NORTH WEST
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i - 1, j - 1))]);
                if (i > 0 && j + 1 < layout::col_count) // NORTH EAST
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i - 1, j + 1))]);
                if (i + 1 < layout::row_count) // SOUTH
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i + 1, j))]);
                if (i + 1 < layout::row_count && j > 0) // SOUTH WEST
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i + 1, j - 1))]);
                if (i + 1 < layout::row_count && j + 1 < layout::col_count) // SOUTH EAST
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i + 1, j + 1))]);
                if (j + 1 < layout::col_count) // EAST
                    cost += synthesis_distance_host(s_ij, schedules[l.get_barcode_index(flat(i, j + 1))]);
            }
        }

#if false

        // SOUTH direction
        for (uint32_t i = 0; i < layout::row_count - 1; i++) {
            for (uint32_t j = 0; j < layout::col_count; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t south = l.get_index_at_position(i + 1, j);
                cost += synthesis_distance_host(schedules[center], schedules[south]);
            }
        }

        // NORTH direction
        for (uint32_t i = 1; i < layout::row_count; i++) {
            for (uint32_t j = 0; j < layout::col_count; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t north = l.get_index_at_position(i - 1, j);
                cost += synthesis_distance_host(schedules[center], schedules[north]);
            }
        }

        // WEST direction
        for (uint32_t i = 0; i < layout::row_count; i++) {
            for (uint32_t j = 1; j < layout::col_count; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t west = l.get_index_at_position(i, j - 1);
                cost += synthesis_distance_host(schedules[center], schedules[west]);
            }
        }

        // EAST direction
        for (uint32_t i = 0; i < layout::row_count; i++) {
            for (uint32_t j = 0; j < layout::col_count - 1; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t east = l.get_index_at_position(i, j + 1);
                cost += synthesis_distance_host(schedules[center], schedules[east]);
            }
        }

        // SOUTH EAST direction
        for (uint32_t i = 0; i < layout::row_count - 1; i++) {
            for (uint32_t j = 0; j < layout::col_count - 1; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t s_east = l.get_index_at_position(i + 1, j + 1);
                cost += synthesis_distance_host(schedules[center], schedules[s_east]);
            }
        }

        // SOUTH WEST direction
        for (uint32_t i = 0; i < layout::row_count - 1; i++) {
            for (uint32_t j = 1; j < layout::col_count; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t s_west = l.get_index_at_position(i + 1, j - 1);
                cost += synthesis_distance_host(schedules[center], schedules[s_west]);
            }
        }

        // NORTH EAST direction
        for (uint32_t i = 1; i < layout::row_count; i++) {
            for (uint32_t j = 0; j < layout::col_count - 1; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t n_east = l.get_index_at_position(i - 1, j + 1);
                cost += synthesis_distance_host(schedules[center], schedules[n_east]);
            }
        }

        // NORTH WEST direction
        for (uint32_t i = 1; i < layout::row_count; i++) {
            for (uint32_t j = 1; j < layout::col_count; j++) {
                uint32_t center = l.get_index_at_position(i, j);
                uint32_t n_west = l.get_index_at_position(i - 1, j - 1);
                cost += synthesis_distance_host(schedules[center], schedules[n_west]);
            }
        }

#endif
        return cost;
    }
}

#endif //INC_2OPT_LAYOUT_COST_H
