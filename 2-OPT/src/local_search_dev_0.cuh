//
// Created by steffen on 07.02.24.
//

#ifndef INC_2OPT_LOCAL_SEARCH_0_CUH
#define INC_2OPT_LOCAL_SEARCH_0_CUH

#include <utility>

#include "synthesis_schedule.h"
#include "synthesis_distance_dev_0.cuh"
#include "layout.h"
#include "layout_cost.h"
#include "direction.h"
#include <chrono>
#include <unordered_set>
#include <bitset>

namespace barcode_layout {

    /**
     *
     *
     * @param schedules
     * @param barcode_order
     * @param barcode_position
     * @param barcode_count
     * @param swap_partner
     * @param cost_improvement
     */
    __global__ void find_best_swap_partner_v0(
            const position *positions,
            const unsigned position_count,
            const synthesis_schedule *schedules,
            const unsigned barcode_count,
            position *swap_partner,     // output variable
            unsigned *cost_improvement       // output variable
    ) {
        unsigned index = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned stride = blockDim.x * gridDim.x;

        // for each position (i,j) that needs to find a new best swap partner
        for (; index < position_count; index += stride) {

            position ij = positions[index];
            unsigned i = ij / layout::col_count;
            unsigned j = ij % layout::col_count;

            /*******************************************************************************
             * Calculate the local cost of layout position (i,j).
             ******************************************************************************/

            unsigned local_cost_pre_swap_ij = 0;

            position ij_N = flat(i-1,j);
            position ij_S = flat(i+1,j);
            position ij_W = flat(i,j-1);
            position ij_E = flat(i,j+1);
            position ij_NW = flat(i-1,j-1);
            position ij_NE = flat(i-1,j+1);
            position ij_SW = flat(i+1,j-1);
            position ij_SE = flat(i+1,j+1);

            if (i > 0)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_N]);
            if (i + 1 < layout::row_count)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_S]);
            if (j > 0)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_W]);
            if (j + 1 < layout::col_count)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_E]);
            if (i > 0 && j > 0)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_NW]);
            if (i > 0 && j + 1 < layout::col_count)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_NE]);
            if (i + 1 < layout::row_count && j > 0)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_SW]);
            if (i + 1 < layout::row_count && j + 1 < layout::col_count)
                local_cost_pre_swap_ij += synthesis_distance_dev_0(schedules[ij], schedules[ij_SE]);

            /**************************************************************************
             * Thread (i,j) simulates the exchange of the two barcodes at positions
             * (i,j) with all layout positions (x,y).
             *
             * It determines the layout position (x,y) that corresponds to the best swap.
             *************************************************************************/

            position best_xy = ij;
            unsigned best_improvement = 0;

            assert(layout::row_count > 0);
            assert(layout::col_count > 0);

            for (unsigned x = 0; x < layout::row_count; x++) {
                for (unsigned y = 0; y < layout::col_count; y++) {

                    position xy = flat(x, y);
                    position xy_N = flat(x-1,y);
                    position xy_S = flat(x+1,y);
                    position xy_W = flat(x,y-1);
                    position xy_E = flat(x,y+1);
                    position xy_NW = flat(x-1,y-1);
                    position xy_NE = flat(x-1,y+1);
                    position xy_SW = flat(x+1,y-1);
                    position xy_SE = flat(x+1,y+1);

                    /**************************************************************************
                     * Calculate the local cost of layout position (x,y).
                     *************************************************************************/

                    unsigned local_cost_pre_swap_xy = 0;

                    if (x > 0)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_N]);
                    if (x + 1 < layout::row_count)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_S]);
                    if (y > 0)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_W]);
                    if (y + 1 < layout::col_count)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_E]);
                    if (x > 0 && y > 0)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_NW]);
                    if (x > 0 && y + 1 < layout::col_count)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_NE]);
                    if (x + 1 < layout::row_count && y > 0)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_SW]);
                    if (x + 1 < layout::row_count && y + 1 < layout::col_count)
                        local_cost_pre_swap_xy += synthesis_distance_dev_0(schedules[xy], schedules[xy_SE]);

                    unsigned local_cost_pre_swap = local_cost_pre_swap_ij + local_cost_pre_swap_xy;

                    /*****************************************************************************
                     * Simulate the effect of swapping the barcodes at positions (i,j) and (x,y).
                     *
                     * First calculate the local cost of layout position (i,j) after the swap.
                     ****************************************************************************/

                    unsigned local_cost_post_swap = 0;

                    if (i > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_N]);
                    if (i + 1 < layout::row_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_S]);
                    if (j > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_W]);
                    if (j + 1 < layout::col_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_E]);
                    if (i > 0 && j > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_NW]);
                    if (i > 0 && j + 1 < layout::col_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_NE]);
                    if (i + 1 < layout::row_count && j > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_SW]);
                    if (i + 1 < layout::row_count && j + 1 < layout::col_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[xy], schedules[ij_SE]);

                    /*****************************************************************************
                     * Now calculate the local cost of layout position (x,y) after the swap.
                     ****************************************************************************/

                    if (x > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_N]);
                    if (x + 1 < layout::row_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_S]);
                    if (y > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_W]);
                    if (y + 1 < layout::col_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_E]);
                    if (x > 0 && y > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_NW]);
                    if (x > 0 && y + 1 < layout::col_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_NE]);
                    if (x + 1 < layout::row_count && y > 0)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_SW]);
                    if (x + 1 < layout::row_count && y + 1 < layout::col_count)
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy_SE]);

                    /********************************************************************
                     * Test if (x,y) is in the direct neighborhood of (i,j).
                     *
                     * If so, one of the above synthesis distances must be zero, as
                     * we calculated the synthesis distance of two identical schedules.
                     *
                     * In this case, we must add the synthesis distance between the
                     * barcodes at positions (i,j) and (x,y) to both local costs.
                     *******************************************************************/

                    if (std::abs((int) i - (int) x) <= 1 && std::abs((int) j - (int) y) <= 1) {
                        local_cost_pre_swap -= synthesis_distance_dev_0(schedules[ij], schedules[xy]);
                        local_cost_post_swap += synthesis_distance_dev_0(schedules[ij], schedules[xy]);
                    }

                    if (local_cost_post_swap < local_cost_pre_swap) {

                        unsigned improvement = 2 * (local_cost_pre_swap - local_cost_post_swap);

                        //printf("swapping positions %i with %i improves layout cost by %i.\n", ij, xy, improvement);

                        if (improvement > best_improvement) {
                            best_xy = xy;
                            best_improvement = improvement;
                        }
                    }
                }
            }

            /**************************************************************************
             * For each unassigned barcode, we calculate the effect of replacing the
             * barcode at position (i,j).
             *************************************************************************/

            for (position xy = layout::row_count * layout::col_count; xy < barcode_count; xy++) {

                synthesis_schedule s_xy = schedules[xy];

                unsigned local_cost_post_swap_ij = 0;

                if (i > 0)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_N]);
                if (i + 1 < layout::row_count)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_S]);
                if (j > 0)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_W]);
                if (j + 1 < layout::col_count)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_E]);
                if (i > 0 && j > 0)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_NW]);
                if (i > 0 && j + 1 < layout::col_count)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_NE]);
                if (i + 1 < layout::row_count && j > 0)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_SW]);
                if (i + 1 < layout::row_count && j + 1 < layout::col_count)
                    local_cost_post_swap_ij += synthesis_distance_dev_0(s_xy, schedules[ij_SE]);

                //printf("local_cost_pre_swap_ij=%i, local_cost_post_swap_ij=%i\n", local_cost_pre_swap_ij,

                if (local_cost_post_swap_ij < local_cost_pre_swap_ij) {

                    unsigned improvement = 2 * (local_cost_pre_swap_ij - local_cost_post_swap_ij);

                    // printf("swapping positions %i with %i improves layout cost by %i. pre_swap=%i, post_swap=%i\n",
                    //       ij, xy, improvement, local_cost_pre_swap_ij, local_cost_post_swap_ij);

                    if (improvement > best_improvement) {
                        best_xy = xy;
                        best_improvement = improvement;
                    }
                }
            }

            /**************************************************************************
             * Store the barcodes best swap partner in global memory.
             *************************************************************************/

            swap_partner[index] = best_xy;
            cost_improvement[index] = best_improvement;

            //printf("%i: best swap partner is %i and improves layout cost by %i.\n", ij,
            //       best_xy, best_improvement);
        }
    }


    class local_search_dev_0 {

        layout current_layout;
        unsigned current_cost;

        position *positions_host, *positions_dev;
        unsigned position_count;

        position *best_xy_dev, *best_xy_host;
        unsigned *cost_improvement_dev, *cost_improvement_host;
        synthesis_schedule *schedules_dev, *schedules_host;

    public:

        static bool verbose;

        local_search_dev_0(const std::vector<synthesis_schedule> &schedules,
                     layout initial_layout)
                : current_layout(std::move(initial_layout)),
                  current_cost(layout_cost(schedules, current_layout)) {

            size_t barcode_count = schedules.size();

            /* Order the synthesis schedules as given by the current layout */
            size_t size = barcode_count * sizeof(synthesis_schedule);
            schedules_host = (synthesis_schedule *) malloc(size);
            for (position ij = 0; ij < barcode_count; ij++)
                schedules_host[ij] = schedules[current_layout.get_barcode_index(ij)];
            cudaMalloc(&schedules_dev, size);

            /****************************************************************************
             * For each layout position (i,j) and each (assigned or unassigned) barcode b,
             * we determine the effect on the layout cost when we exchange both barcodes.
             *
             * For each layout position (i,j), we determine a barcode b that maximizes
             * the improvement of the layout cost.
             ***************************************************************************/

            positions_host = (position *) malloc(layout::row_count * layout::col_count * sizeof(position));
            cudaMalloc(&positions_dev, layout::row_count * layout::col_count * sizeof(position));

            cudaMalloc(&best_xy_dev, layout::row_count * layout::col_count * sizeof(position));
            cudaMalloc(&cost_improvement_dev, layout::row_count * layout::col_count * sizeof(unsigned));
            best_xy_host = (position *) malloc(
                    layout::row_count * layout::col_count * sizeof(position));
            cost_improvement_host = (unsigned *) malloc(layout::row_count * layout::col_count * sizeof(unsigned));

            if(verbose) {
                // print headline
                std::cout << "iteration,cost,ms,swaps,positions" << std::endl;
                std::cout << "0," << current_cost << ",0,0,0" << std::endl;
            }

            // initially, we need to find best swap partners for *all* positions
            for (position ij = 0; ij < layout::row_count * layout::col_count; ij++)
                positions_host[ij] = ij;
            position_count = layout::row_count * layout::col_count;

            bool refresh = true;

            unsigned total_swap_cnt = 0;
            for (int iteration = 1;; iteration++) {

                auto start = std::chrono::high_resolution_clock::now();

                /* Copy the positions to the device for which we need to find best swap partners */
                cudaMemcpy(positions_dev, positions_host,
                           position_count * sizeof(position),
                           cudaMemcpyHostToDevice);

                /* Copy the schedule order to the device */
                cudaMemcpy(schedules_dev, schedules_host,
                           barcode_count * sizeof(synthesis_schedule),
                           cudaMemcpyHostToDevice);

                /* Find best swap partners for each position (i,j) */
                find_best_swap_partner_v0
                <<<1024, 256>>>(
                        positions_dev,
                        position_count,
                        schedules_dev,
                        barcode_count,
                        best_xy_dev,
                        cost_improvement_dev);
                auto err = cudaPeekAtLastError();
                if (err != cudaError_t::cudaSuccess) {
                    printf("GPU kernel assert: %s\n", cudaGetErrorString(err));
                    break;
                }
                cudaDeviceSynchronize();

                cudaMemcpy(best_xy_host, best_xy_dev,
                           position_count * sizeof(position),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(cost_improvement_host, cost_improvement_dev,
                           position_count * sizeof(unsigned),
                           cudaMemcpyDeviceToHost);

                /****************************************************************************
                 * We greedily select a subset of swaps that can be performed independently.
                 *
                 * For this purpose, we mark positions in whose neighborhood we performed
                 * some swap. These positions are locked for further locks.
                 ***************************************************************************/

                // sort the swaps by their improvement
                std::vector<unsigned> swap_order;
                for (unsigned index = 0; index < position_count; index++)
                    swap_order.push_back(index);
                std::sort(swap_order.begin(), swap_order.end(),
                          [&](unsigned p, unsigned q) {
                              if (cost_improvement_host[p] > cost_improvement_host[q])
                                  return true;
                              else if (cost_improvement_host[p] == cost_improvement_host[q])
                                  return p < q;
                              else
                                  return false;
                          });

                /*std::cout << "swap order:" << std::endl;
                for (position ij: swap_order) {
                    std::cout << ij << ": " << best_xy_host[ij] << ", improvement="
                              << cost_improvement_host[ij] << std::endl;
                }*/

                /******************************************************************************
                 * If no swap can be performed any more, we re-calculate best swap partners
                 * for *all* positions. If no swap can improve the current layout after that,
                 * we found a local optimum.
                 ******************************************************************************/

                if (cost_improvement_host[swap_order[0]] <= 0) {

                    if(verbose) {
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                        std::cout << iteration << "," << current_cost << "," << duration.count()
                                  << "," << 0 << "," << position_count << std::endl;
                    }

                    if (refresh)
                        break; // we found a local optimum

                    // mark all positions as needed for recalculation
                    position_count = layout::row_count * layout::col_count;
                    for (position ij = 0; ij < layout::row_count * layout::col_count; ij++)
                        positions_host[ij] = ij;

                    refresh = true;

                    continue;
                } else {
                    refresh = false;
                }

                unsigned new_cost = current_cost;

                std::vector<bool> locked(barcode_count, false);

                unsigned swap_cnt = 0;

                for (unsigned index: swap_order) {

                    if (cost_improvement_host[index] <= 0)
                        break;

                    position ij = positions_host[index];
                    unsigned i = ij / layout::col_count;
                    unsigned j = ij % layout::col_count;

                    position xy = best_xy_host[index];
                    unsigned x = xy / layout::col_count;
                    unsigned y = xy % layout::col_count;

                    // if some barcode in the neighborhood of (i,j) has already been affected by some previous swap
                    if (locked[ij])
                        continue; // do not perform this swap

                    // if the swap partner at position (x,y) has been affected by some previous swap
                    if (locked[xy])
                        continue; // do not perform this swap

                    /****************************************************************************
                     * Exchange the barcodes b_ij and b_xy.
                     **************************************************************************/

                    synthesis_schedule sched = schedules_host[ij];
                    schedules_host[ij] = schedules_host[xy];
                    schedules_host[xy] = sched;
                    current_layout.swap_barcodes_at_position(ij, xy);

                    /****************************************************************************
                     * Mark the neighbored positions of (i,j) and (x,y) as locked.
                     **************************************************************************/

                    locked[ij] = true;
                    locked[xy] = true;

                    if (i > 0)
                        locked[flat(i - 1, j)] = true;
                    if (i + 1 < layout::row_count)
                        locked[flat(i + 1, j)] = true;
                    if (j > 0)
                        locked[flat(i, j - 1)] = true;
                    if (j + 1 < layout::col_count)
                        locked[flat(i, j + 1)] = true;
                    if (i > 0 && j > 0)
                        locked[flat(i - 1, j - 1)] = true;
                    if (i > 0 && j + 1 < layout::col_count)
                        locked[flat(i - 1, j + 1)] = true;
                    if (i + 1 < layout::row_count && j > 0)
                        locked[flat(i + 1, j - 1)] = true;
                    if (i + 1 < layout::row_count && j + 1 < layout::col_count)
                        locked[flat(i + 1, j + 1)] = true;

                    // if (x,y) is on the layout
                    if (xy < layout::row_count * layout::col_count) {
                        // we need to lock the adjacent layout positions
                        if (x > 0)
                            locked[flat(x - 1, y)] = true;
                        if (x + 1 < layout::row_count)
                            locked[flat(x + 1, y)] = true;
                        if (y > 0)
                            locked[flat(x, y - 1)] = true;
                        if (y + 1 < layout::col_count)
                            locked[flat(x, y + 1)] = true;
                        if (x > 0 && y > 0)
                            locked[flat(x - 1, y - 1)] = true;
                        if (x > 0 && y + 1 < layout::col_count)
                            locked[flat(x - 1, y + 1)] = true;
                        if (x + 1 < layout::row_count && y > 0)
                            locked[flat(x + 1, y - 1)] = true;
                        if (x + 1 < layout::row_count && y + 1 < layout::col_count)
                            locked[flat(x + 1, y + 1)] = true;
                    }

                    /****************************************************************************
                     * Update counting variables.
                     **************************************************************************/

                    swap_cnt++;
                    total_swap_cnt++;

                    // assert that the cost improvement was calculated correctly
                    new_cost -= cost_improvement_host[index];

                    /*std::cout << "performed swap #" << swap_cnt << " at positions ij=" << ij << " (" << i << "," << j
                              << ")" << " and xy=" << xy << " ("
                              << x << "," << y << ")" << std::endl;
                    std::cout << "cost improvement = " << cost_improvement_host[ij] << std::endl;
                    std::cout << "new_cost         = " << new_cost << std::endl;
                    std::cout << "layout_cost      = " << layout_cost(schedules, current_layout) << std::endl;
                    std::cout << "diff             = " << layout_cost(schedules, current_layout) - new_cost
                              << std::endl;
                    if (new_cost != layout_cost(schedules, current_layout)) {
                        std::cerr << "ERROR!" << std::endl;

                        return;
                    }*/
                    assert(new_cost == layout_cost(schedules, current_layout));
                }

                /************************************************************************
                 * Output the progress
                 ***********************************************************************/

                current_cost = new_cost;
                if(verbose) {
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                    std::cout << iteration << "," << new_cost << "," << duration.count()
                              << "," << swap_cnt << "," << position_count << std::endl;
                }

                /************************************************************************
                 * Determine the positions (i,j) for which we need to find a new best
                 * swap partners. These are exactly the positions that have been locked.
                 ***********************************************************************/

                position_count = 0;
                for (position ij = 0; ij < layout::row_count * layout::col_count; ij++) {
                    if (locked[ij]) {
                        positions_host[position_count] = ij;
                        position_count++;
                    }
                }
            }

        }

        ~local_search_dev_0() {
            cudaFree(schedules_dev);
            cudaFree(best_xy_dev);
            free(best_xy_host);
            free(schedules_host);
            free(cost_improvement_host);
            cudaFree(cost_improvement_dev);
            cudaFree(positions_dev);
            free(positions_host);
        }

        operator layout() {
            return current_layout;
        }
    };

    bool local_search_dev_0::verbose = false;
}

#endif //INC_2OPT_LOCAL_SEARCH_0_CUH
