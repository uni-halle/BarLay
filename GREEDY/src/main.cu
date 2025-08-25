#include "barcodes.cuh"
#include "layout.cuh"
#include "algorithm.cuh"

#include <cstdio>

int main(int argc, char *argv[]) {
    const auto schedules = static_cast<barcodes::ScheduleSet *>(malloc(sizeof(barcodes::ScheduleSet)));
    auto layout = static_cast<layout::Layout *>(malloc(sizeof(layout::Layout)));

    barcodes::read(stdin, schedules);

    algorithm::optimize(schedules, layout);

    // layout::initialize(layout);
    layout::print(stdout, schedules, layout);

    free(schedules);
    free(layout);
}