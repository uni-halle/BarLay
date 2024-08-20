#include <iostream>
#include "synthesis_schedule.h"
#include "layout.h"
#include "random_layout.h"
#include "input_layout.h"
#include "local_search.cuh"
#include "layout_cost.h"
#include "local_search_host.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace barcode_layout;

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "Usage: 2opt <file> [<options> ...]" << std::endl;
        std::cout << " file: barcode file" << std::endl;
        std::cout << " options are:" << std::endl;
        std::cout << "    -v: verbose mode (additional output)" << std::endl;
        return -1;
    }

    /* Parse command line arguments */
    if (argc > 2 && std::string(argv[2]) == "-v") {
        local_search::verbose = true;
    }

    /* Read barcodes from file */
    std::vector<barcode> barcodes;
    std::ifstream infile(argv[1]);
    std::string line;
    while (std::getline(infile, line))
        barcodes.emplace_back(line);

    if(barcodes.size() < layout::col_count*layout::row_count) {
        std::cerr << "Error! Not enough barcodes! At least " << layout::col_count*layout::row_count << " needed!" << std::endl;
        return -2;
    }

    //std::cout << "Excess: " << barcodes.size() - layout::row_count*layout::col_count << std::endl;

    /* Calculate synthesis schedules */
    std::vector<synthesis_schedule> schedules;
    for (const auto &barcode: barcodes) // for each barcode
        schedules.emplace_back(barcode);

    /* Create an initial layout */
    layout initial_layout = input_layout(barcodes.size());
    unsigned initial_cost = layout_cost(schedules, initial_layout);

    /* Print the initial layout cost */
    if (local_search::verbose)
        std::cout << "initial layout cost = " << layout_cost(schedules, initial_layout) << std::endl;

    /* Start the local search */
    layout final_layout = local_search(schedules, initial_layout);
    unsigned final_cost = layout_cost(schedules, final_layout);

    /* Print the final layout cost */
    if (local_search::verbose) {
        double gain = 100.0 * (initial_cost - final_cost) / initial_cost;
        std::cout << final_cost << ", improvement=" << (initial_cost - final_cost) << ", gain=" << gain << "%"
                  << std::endl;
    }

    /* Output the final layout */
    for(barcode_index k : final_layout.get_barcode_order())
        std::cout << barcodes[k] << std::endl;

    return 0;
}
