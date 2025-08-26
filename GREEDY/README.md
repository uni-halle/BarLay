# BarLay/GREEDY

This repository provides a CUDA-optimized implementation of the GREEDY algorithm to optimize the layout of DNA-barcodes on 2d arrays.

## Dependencies

You need the follwing dependencies present on your system.

- CUDA `11.x` or `12.x` with `nvcc` in the search path

## Building the program

The algorithm can be compiled with different parameters:

- NUMBER_OF_SCHEDULES: This parameter specifies the number of newline separated schedules (words over `(0|1)^SCHEDULE_LENGTH`) to read from stdin.
- SCHEDULE_LENGTH: This specifies the length of each schedule in chars.
- ROW_COUNT and COL_COUNT: These specifiy the dimensions of the array on which the schedules should be layed out. Note that `(ROW_COUNT * COL_COUNT) <= NUMBER_OF_SCHEDULES` must be true.

Now you can build the program using CMAKE. Adjust the build args using

```
-DROW_COUNT=1024
-DCOL_COUNT=768
-DNUMBER_OF_SCHEDULES=1024*768
-DSCHEDULE_LENGTH=34*4
-DNEIGHBORHOOD=NEIGHBORHOOD_N8
-DOCCUPANCY=OCCUPANCY_FULL or OCCUPANCY_CHESSBOARD
```

## Scheduling the barcodes

See [our tooling CLI](../tooling/README.md) for information about how to schedule barcodes.
This CLI contains common but simple and computationally not demanding functionalities.

## Running the program

The program reads the specified number of schedules from `stdin` and writes the result to `stdout`. Additionally, log output is written to `stderr`.

Test it out using the following command.

```
cat <MY_SCHEDULE_LIBRARY> | <BINARY> 1> result.txt
```

After the progress bar reaches `100%` you should find a resulting layout as a newline separated row major list in `./result.txt`.

## Neighborhood Strategies

The neighborhood strategy defines how the neighbors of a position influence the quality of a candidate for the position.

In order to add a new strategy:

1. Define a new preprocessor constant in `src/config.cuh.in`
2. Extend the conditional compilation in `src/neighborhood.cuh` accordingly
3. Compile the program for the new strategy by using `-DNEIGHBORHOOD=<your constant>`

## Occupancy Strategies

The occupancy strategy defines which positions in the array will be assigned a schedule.
This allows for sparse array layouts.

In order to add a new strategy:

1. Define a new preprocessor constant in `src/config.cuh.in`
2. Extend the conditional compilation in `src/occupancy.cuh` accordingly
3. Compile the program for the new strategy by using `-DOCCUPANCY=<your constant>`