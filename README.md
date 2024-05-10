# BarLay - Barcode Layout Optimization

## Problem Description

Given a rectangular grid, the **barcode layout problem** asks for an assignment of so called _barcodes_ (i.e. short DNA sequences of fixed length) to the grid positions such that a certain error function is minimized. It is an important subproblem in bioinformatics, in particular in the field of spatial transcriptomics.

todo: Add more details

## Project Organization

This project is separated in several subprojects, each associated to some heuristic. Each subproject contains with a README file in which the installation process and command line options are explained. At the moment, the project contains the following subprojects:

[2-OPT Local Search](https://github.com/uni-halle/BarLay/tree/main/2-OPT)

## Input and Output Format

Input of the tools are text files with one barcode per line. The barcodes are interpreted in row-major-order. Thus, assuming a layout with $`m`$ rows and $`n`$ columns, the barcodes in row $`i`$ are the ones in the lines $`i\cdot n + 1, i\cdot n + 2, ..., (i+1)\cdot n`$. A layout file is allowed to contain more barcodes than there are layout positions, in which case the barcode after line $`m\cdot n`$ can be used as excess.

For example, a layout file could start with the lines

    CGTCCATCTTCATATGCGTCCTCGACTAGTGCTA
    GCCACCTGCCACTGTCAGCTACGATGCACTCGCA
    TATACTGCGTTTCCGTGAGATCGGCGTGCCGGTC
    ATACGCAACCCTCTCCCTCGCAAGATCTGGTTGT
    CGGCTCAGGACATTAGCTCTGCGGCGATTTGTCC
    TTAAGGGCCAGCAGAGGATGCATGCATCGACGGG
    CGGTAACACCTGACAGTAAGGGAAGACTTTCGGC
    ATGTGCTTGTAAACACGCATCTCAGTGTTACACC
    CCTCACGAGCTAGCTATTTGATACCACTGCACAC
    ACCTTGACAGCATTGACCTTTATCAATGTGTTCT
    ...

The output format is identical to the input. 

## Installation

1. Check out the repository.

   ````
   git clone https://github.com/uni-halle/BarLay.git
   ````

2. Create a build directory.

   ````
     cd BarLay
     mkdir build
     cd build
   ````
   
3. Build the program.

   ````
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   ````
## Data

In the article below, we evalualted the algorithms on several barcode sets. You may find these barcodes in the [Data](https://github.com/uni-halle/BarLay/tree/main/data) directory. 

## Citation

To cite this work in publication, please use
> Frederik Jatzkowski, Antonia Schmidt, Robert Mank, Steffen Schüler, and Matthias Müller-Hannemann,
> _Barcode Selection and Layout Optimization in Spatial Transcriptomics_,
> Proceedings of the 22nd International Symposium on Experimental Algorithms (SEA 2024), DOI: https://doi.org/10.4230/LIPIcs.SEA.2024.17.
