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

    if (argc != 2) {
        std::cout << "Usage: 2opt <file>" << std::endl;
        std::cout << " file: barcode file\n" << std::endl;
        return -1;
    }

    /* Read barcodes from file */
    std::vector<barcode> barcodes;
    std::ifstream infile(argv[1]);
    std::string line;
    while (std::getline(infile, line))
        barcodes.emplace_back(line);

    //std::cout << "Excess: " << barcodes.size() - layout::row_count*layout::col_count << std::endl;

    /* Calculate synthesis schedules */
    std::vector<synthesis_schedule> schedules;
    for (const auto &barcode: barcodes) // for each barcode
        schedules.emplace_back(barcode);

    /* Create an initial layout */
    //layout initial_layout = random_layout(barcodes);
    layout initial_layout = input_layout(barcodes);
    unsigned initial_cost = layout_cost(schedules, initial_layout);

    /* Print the initial layout cost */
    //std::cout << layout_cost(schedules, initial_layout) << std::endl;

    /* Start the local search */
    //layout final_layout = local_search_host(schedules, initial_layout);
    layout final_layout = local_search(schedules, initial_layout);
    unsigned final_cost = layout_cost(schedules, final_layout);

    /* Print the final layout cost */
    double gain = 100.0 * (initial_cost - final_cost) / initial_cost;
    std::cout << final_cost << ", improvement=" << (initial_cost - final_cost) << ", gain=" << gain << "%" << std::endl;

    return 0;
}
