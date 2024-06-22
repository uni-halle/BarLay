#include "barcodes.cu"
#include "layout.cu"
#include "algorithm.cu"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    barcodes::Set *barcodes = (barcodes::Set *)malloc(sizeof(barcodes::Set));
    barcodes::ScheduleSet *schedules = (barcodes::ScheduleSet *)malloc(sizeof(barcodes::ScheduleSet));
    layout::Layout *layout = (layout::Layout *)malloc(sizeof(layout::Layout));

    barcodes::read(stdin, barcodes, schedules);

    algorithm::optimize(schedules, layout);

    layout::print(stdout, barcodes, layout);

    free(barcodes);
    free(schedules);
    free(layout);
}