#include "barcodes.cu"
#include "layout.cu"
#include "algorithm.cu"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    barcodes::ScheduleSet *schedules = (barcodes::ScheduleSet *)malloc(sizeof(barcodes::ScheduleSet));
    layout::Layout *layout = (layout::Layout *)malloc(sizeof(layout::Layout));

    barcodes::read(stdin, schedules);

    algorithm::optimize(schedules, layout);

    // layout::initialize(layout);
    layout::print(stdout, schedules, layout);

    free(schedules);
    free(layout);
}