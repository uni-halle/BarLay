# barcode-layout

This repository provides a CUDA-optimized implementation of the GREEDY algorithm to optimize the layout of DNA-barcodes on 2d arrays.

## Dependencies

You need the follwing dependencies present on your system.

- CUDA 11.x with `nvcc` in the search path

## Building the program

The algorithm can be compiled with different parameters:

- NUMBER_OF_BARCODES: This parameter specifies the number of newline separated barcodes to read from stdin.
- BARCODE_LENGTH: This specifies the length of each barcode in bytes.
- DIM_X and DIM_Y: These specifiy the dimensions of the array on which the barcodes should be layed out. Note that `(DIM_X * DIM_Y) <= NUMBER_OF_BARCODES` must be true.
- NEIGHBORHOOD: This describes the neighborhood strategy to apply. A default strategy which uses 8 unweighted neighbors is provided.

Now you can build the program using

```
./scripts/build_greedy.sh <NUMBER_OF_BARCODES> <BARCODE_LENGTH> <DIM_X> <DIM_Y> [ <NEIGHBORHOOD_NAME> ]
```

## Running the program

The program reads the specified number of barcodes from `stdin` and writes the result to `stdout`. Additionally, log output is written to `stderr`.

Test it out using the following command.

```
cat <MY_BARCODE_LIBRARY> | ./bin/barlay 1> result.json
```

After the progress bar reaches `100%` you should find a resulting layout in `./result.json`.

## Customizing the neighborhood

Neighborhood definitions can be found in `./src/neighborhood/*`.
Run `./scripts/define_new_neighborhood.sh <NEIGHBORHOOD_NAME>` to create the boilerplate for a new neighborhood.
In the created file you must define a type named `Neighborhood` and two functions `load` and `nquality`. GREEDY will use these to determine the best candidate at a position `(x,y)`.

Take `./src/neighborhood/n8.cu` as an example.
The code defined here runs on the GPU so take this in mind for performance reasons.

## Running on HP-Clusters using slurm

1. SSH into the login node of the HP cluster (Currently this works only if CUDA 12.3 is installed on the cluster nodes and nvcc is located at `/usr/local/cuda-12.3/bin/nvcc`).

2. `git clone` this repo somewhere. Stay where you are, do not cd into the cloned repo.

3. Copy the barcode set somewhere onto the login node (eg next to the cloned repo).

4. Configure greedy using `barcode-layout/scripts/configure_greedy.sh <NUMBER_OF_BARCODES> <BARCODE_LENGTH> <DIM_X> <DIM_Y> [ <NEIGHBORHOOD_NAME> ]`.

5. Queue the job using `sbatch barcode-layout/scripts/batch.sh <KUERZEL> <PATH_TO_BARCODE_SET>`.

6. Follow the progress using `tail -f slurm-<JOB_ID>.err`. Once finished, the generated layout will be in `greedy-layout-<JOB_ID>.json`.