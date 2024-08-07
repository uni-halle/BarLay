# barcode-layout

This repository provides a CUDA-optimized implementation of the GREEDY algorithm to optimize the layout of DNA-barcodes on 2d arrays.

## Dependencies

You need the follwing dependencies present on your system.

- CUDA `11.x` or `12.x` with `nvcc` in the search path

## Building the program

The algorithm can be compiled with different parameters:

- NUMBER_OF_BARCODES: This parameter specifies the number of newline separated barcodes to read from stdin.
- BARCODE_LENGTH: This specifies the length of each barcode in bytes.
- ROW_COUNT and COL_COUNT: These specifiy the dimensions of the array on which the barcodes should be layed out. Note that `(ROW_COUNT * COL_COUNT) <= NUMBER_OF_BARCODES` must be true.
- NEIGHBORHOOD: This describes the neighborhood strategy to apply. A default strategy which uses 8 unweighted neighbors is provided.

Now you can build the program using

```
./scripts/build.sh <NUMBER_OF_BARCODES> <BARCODE_LENGTH> <ROW_COUNT> <COL_COUNT> [ <NEIGHBORHOOD_NAME> ]
```

Depending on your platform, the shell script is not executable by default. Set the necessary permissions using `chmod` or the equivalent on your platform.

## Running the program

The program reads the specified number of barcodes from `stdin` and writes the result to `stdout`. Additionally, log output is written to `stderr`.

Test it out using the following command.

```
cat <MY_BARCODE_LIBRARY> | ./bin/barlay 1> result.txt
```

After the progress bar reaches `100%` you should find a resulting layout as a newline separated row major list in `./result.txt`.

## Customizing the neighborhood

Neighborhood definitions can be found in `./src/neighborhood/*`.
Run `./scripts/define_new_neighborhood.sh <NEIGHBORHOOD_NAME>` to create the boilerplate for a new neighborhood.
In the created file you must define a type named `Neighborhood` and two functions `load` and `nquality`. GREEDY will use these to determine the best candidate at a position `(x,y)`.

Take `./src/neighborhood/n8.cu` as an example.
The code defined here runs on the GPU so take this in mind for performance reasons.