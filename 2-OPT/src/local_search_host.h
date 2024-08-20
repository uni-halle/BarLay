//
// Created by steffen on 15.02.24.
//

#ifndef INC_2OPT_LOCAL_SEARCH_HOST_H
#define INC_2OPT_LOCAL_SEARCH_HOST_H

#include "layout.h"
#include "synthesis_schedule.h"
#include "direction.h"
#include <vector>
#include <algorithm>
#include <chrono>

namespace barcode_layout {

    class local_search_host {

        layout current_layout;
        unsigned current_cost;

    public:

        local_search_host(const std::vector<synthesis_schedule> &schedules,
                          layout initial_layout)
                : current_layout(std::move(initial_layout)),
                  current_cost(layout_cost(schedules, current_layout)) {

            unsigned barcode_count = schedules.size();

            for (int iteration = 1;; iteration++) {

                auto start = std::chrono::high_resolution_clock::now();

                /**********************************************************************************
                 * For each layout position (i,j) find a barcode b such that exchanging both
                 * barcodes reduces the layout cost.
                 *********************************************************************************/

                std::vector<barcode_index> swap_partner(layout::row_count * layout::col_count);
                std::vector<int> cost_improvement(layout::row_count * layout::col_count);

                for (int i = 0; i < layout::row_count; i++) {
                    for (int j = 0; j < layout::col_count; j++) {

                        position ij = i * layout::col_count + j;

                        /*******************************************************************************
                         * Load the barcodes and synthesis schedules of the barcode at position (i,j)
                         * and its surrounding 8 neighbors.
                         ******************************************************************************/

                        barcode_index N1[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
                        N1[CENTER] = current_layout.get_barcode_index(ij);
                        if (i > 0)
                            N1[NORTH] = current_layout.get_barcode_index(ij-layout::col_count);
                        if (i + 1 < layout::row_count)
                            N1[SOUTH] = current_layout.get_barcode_index(ij + layout::col_count);
                        if (j > 0)
                            N1[WEST] = current_layout.get_barcode_index(ij - 1 );
                        if (j + 1 < layout::col_count)
                            N1[EAST] = current_layout.get_barcode_index(ij + 1);
                        if (i > 0 && j > 0)
                            N1[NORTH_WEST] = current_layout.get_barcode_index(ij - layout::col_count - 1);
                        if (i > 0 && j + 1 < layout::col_count)
                            N1[NORTH_EAST] = current_layout.get_barcode_index(ij - layout::col_count + 1);
                        if (i + 1 < layout::row_count && j > 0)
                            N1[SOUTH_WEST] = current_layout.get_barcode_index(ij + layout::col_count - 1);
                        if (i + 1 < layout::row_count && j + 1 < layout::col_count)
                            N1[SOUTH_EAST] = current_layout.get_barcode_index(ij + layout::col_count + 1);

                        synthesis_schedule S1[9];
                        for (int k = 0; k < 9; k++) {
                            if (N1[k] != -1)
                                S1[k] = schedules[N1[k]];
                        }

                        /**********************************************************************************
                         * For each layout position (i,j) find a barcode b such that exchanging both
                         * barcodes reduces the layout cost.
                         *********************************************************************************/

                        position best_xy = i * layout::col_count + j;
                        int best_improvement = 0;

                        for (int x = 0; x < layout::row_count; x++) {
                            for (int y = 0; y < layout::col_count; y++) {

                                unsigned xy = x * layout::col_count + y;

                                /*******************************************************************************
                                 * Load the barcodes and synthesis schedules of the barcode at position (x,y)
                                 * and its surrounding 8 neighbors.
                                 ******************************************************************************/

                                barcode_index N2[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
                                N2[CENTER] = current_layout.get_barcode_index(xy);
                                if (x > 0)
                                    N2[NORTH] = current_layout.get_barcode_index(xy - layout::col_count);
                                if (x + 1 < layout::row_count)
                                    N2[SOUTH] = current_layout.get_barcode_index(xy + layout::col_count);
                                if (y > 0)
                                    N2[WEST] = current_layout.get_barcode_index(xy - 1);
                                if (y + 1 < layout::col_count)
                                    N2[EAST] = current_layout.get_barcode_index(xy + 1);
                                if (x > 0 && y > 0)
                                    N2[NORTH_WEST] = current_layout.get_barcode_index(xy - layout::col_count - 1);
                                if (x > 0 && y + 1 < layout::col_count)
                                    N2[NORTH_EAST] = current_layout.get_barcode_index(xy - layout::col_count + 1);
                                if (x + 1 < layout::row_count && y > 0)
                                    N2[SOUTH_WEST] = current_layout.get_barcode_index(xy + layout::col_count - 1);
                                if (x + 1 < layout::row_count && y + 1 < layout::col_count)
                                    N2[SOUTH_EAST] = current_layout.get_barcode_index(xy + layout::col_count + 1);

                                synthesis_schedule S2[9];
                                for (int k = 0; k < 9; k++) {
                                    if (N2[k] != -1)
                                        S2[k] = schedules[N2[k]];
                                }

                                /*******************************************************************
                                 * Calculate the effect on the layout cost if we exchange the barcodes
                                 * at positions (i,j) and (x,y).
                                 *******************************************************************/

                                int local_cost_pre_swap = 0;
                                for (auto dir: {NORTH, NORTH_WEST, NORTH_EAST, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST,
                                                EAST}) {
                                    if (N1[dir] != -1)
                                        local_cost_pre_swap += synthesis_distance_host(S1[CENTER], S1[dir]);
                                    if (N2[dir] != -1)
                                        local_cost_pre_swap += synthesis_distance_host(S2[CENTER], S2[dir]);
                                }

                                int local_cost_post_swap = 0;
                                for (auto dir: {NORTH, NORTH_WEST, NORTH_EAST, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST,
                                                EAST}) {

                                    if (N1[dir] != -1) {
                                        if (N1[dir] == N2[CENTER])
                                            local_cost_post_swap += synthesis_distance_host(S1[CENTER], S1[dir]);
                                        else
                                            local_cost_post_swap += synthesis_distance_host(S2[CENTER], S1[dir]);
                                    }

                                    if (N2[dir] != -1) {
                                        if (N2[dir] == N1[CENTER])
                                            local_cost_post_swap += synthesis_distance_host(S2[CENTER], S2[dir]);
                                        else
                                            local_cost_post_swap += synthesis_distance_host(S1[CENTER], S2[dir]);
                                    }
                                }

                                if (local_cost_pre_swap > local_cost_post_swap) {
                                    int improvement = 2 * (local_cost_pre_swap - local_cost_post_swap);
                                    if (improvement > best_improvement) {
                                        best_xy = xy;
                                        best_improvement = improvement;
                                    }
                                }
                            }
                        }

                        swap_partner[ij] = current_layout.get_barcode_index(best_xy);
                        cost_improvement[ij] = best_improvement;

                        /**************************************************************************
                         * For each unassigned barcode, we calculate the effect of placing it
                         * at position (i,j).
                         *************************************************************************/

                        for (unsigned k = layout::row_count * layout::col_count; k < barcode_count; k++) {

                            barcode_index index = current_layout.get_barcode_index(k);
                            synthesis_schedule sched = schedules[index];

                            unsigned local_cost_pre_swap_ij = 0;
                            for (auto dir: {NORTH, NORTH_WEST, NORTH_EAST, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST, EAST}) {
                                if (N1[dir] != -1)
                                    local_cost_pre_swap_ij += synthesis_distance_host(S1[CENTER], S1[dir]);
                            }

                            unsigned local_cost_post_swap_ij = 0;
                            for (auto dir: {NORTH, NORTH_WEST, NORTH_EAST, WEST, SOUTH_WEST, SOUTH, SOUTH_EAST, EAST}) {
                                if (N1[dir] != -1)
                                    local_cost_post_swap_ij += synthesis_distance_host(sched, S1[dir]);
                            }

                            int improvement = 2 * (local_cost_pre_swap_ij - local_cost_post_swap_ij);
                            //printf("swapping positions %i with %i improves layout cost by %i.\n", ij, xy, improvement);

                            if (improvement > cost_improvement[ij]) {
                                swap_partner[ij] = index;
                                cost_improvement[ij] = improvement;
                            }
                        }

                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

                        std::cout << "ij=" << ij << ", time=" << duration.count() << " ms"
                                  << std::endl;
                    }
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                    std::cout << "i=" << i << ", time=" << duration.count() << " ms"
                              << std::endl;
                }

                // sort the swaps by their improvement
                std::vector<unsigned> swap_order;
                for (unsigned ij = 0; ij < layout::row_count * layout::col_count; ij++)
                    swap_order.push_back(ij);
                std::sort(swap_order.begin(), swap_order.end(),
                          [&](unsigned id1, unsigned id2) {
                              return cost_improvement[id1] > cost_improvement[id2];
                          });

                /*std::cout << "swap order:" << std::endl;
                for (unsigned ij: swap_order) {
                    std::cout << ij << ": " << swap_partner[ij] << ", improvement="
                              << cost_improvement[ij] << std::endl;
                }*/

                position ij = swap_order[0];
                barcode_index b_ij = current_layout.get_barcode_index(ij);
                barcode_index b_xy = swap_partner[ij];
                if (cost_improvement[ij] <= 0)
                    break;

                // swap barcode at position ij with barcode at position swap_partner[ij]
                current_layout.swap_barcodes(b_ij, b_xy);

                // assert that the cost improvement was calculated correctly
                unsigned new_cost = current_cost - cost_improvement[ij];
                assert(new_cost == layout_cost(schedules, current_layout));

                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

                std::cout << "iteration=" << iteration << ": swapped barcodes " << b_ij << " and " << b_xy
                          << ", improvement=" << cost_improvement[ij]
                          << ", time=" << duration.count() << " ms"
                          << std::endl;

                current_cost = new_cost;
            }
        }

        operator layout() {
            return current_layout;
        }
    };

}

#endif //INC_2OPT_LOCAL_SEARCH_HOST_H
