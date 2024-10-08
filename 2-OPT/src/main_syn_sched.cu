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

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cout << "Usage: 2opt_syn_sched <file> [<options> ...]" << std::endl;
        std::cout << " file: file with synthesis schedules" << std::endl;
        std::cout << " options are:" << std::endl;
        std::cout << "    -v: verbose mode (additional output)" << std::endl;
        return -1;
    }

    /* Parse command line arguments */
    if (argc > 2 && std::string(argv[2]) == "-v") {
        local_search::verbose = true;
    }

    /* Read synethesis schedules from file */
    std::vector<synthesis_schedule> schedules;
    std::ifstream infile(argv[1]);
    std::string line;
    while (std::getline(infile, line))
        schedules.emplace_back(line);

    if (schedules.size() < layout::col_count * layout::row_count) {
        std::cerr << "Error! Not enough synthesis schedules! At least " << layout::col_count * layout::row_count << " needed!" <<
            std::endl;
        return -2;
    }

    /* Create an initial layout */
    layout initial_layout = input_layout(schedules.size());
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

    return 0;
}
