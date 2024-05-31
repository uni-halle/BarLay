# BarLay - Barcode Layout Optimization

## Problem Description

Given a rectangular grid, the **barcode layout problem** asks for an assignment of so called _barcodes_ (i.e. short DNA sequences of fixed length) to the grid positions such that a certain error function is minimized. It is an important subproblem in bioinformatics, in particular in the field of spatial transcriptomics.

### Biological Background

While the genetic information in every cell of an organism is identical, the appearances and functions of different cells in the body are different. This is due to differential gene expression in the cells, which is the object of research of the field of transcriptomics. Typically, the transcripts of cells are sequenced to measure gene expression. Spatial transcriptomics deals with the spatial differences of gene expression in tissue. One method in spatial transcriptomics uses microarrays. These can be understood as two-dimensional arrays. The tissue is applied to the array and the transcripts are bound to so-called barcodes. The barcodes are short, artificially synthesized DNA sequences that encode the position on the array. After sequencing, the transcripts can only be assigned to the original position using the barcodes.

Unfortunately, the synthesis of the barcodes is not exact, which makes it difficult to assign the barcodes to the positions. The aim of this work is to minimize synthesis errors by optimizing the placement of the barcodes on the array. To understand the background, we need to understand the process of synthesis.

The barcodes are synthesized in cycles. In each cycle, the partial sequences of the array positions are covered by a protective layer. The protective layer is now removed with focused UV exposure at exactly those array positions where a new base is to be attached. The base is added to the array and can bind to the deprotected array positions. A common source of error is caused by scattering of the UV light. The protective layer of neighboring barcodes is then removed, which leads to insertion errors.

### Optimization Problem

The aim is therefore to minimize the number of situations in which a barcode is synthesized and the neighbouring barcode is not. To do this, we define the synthesis schedules of the barcodes as binary strings, with each digit corresponding to a synthesis cycle. An entry of the string is 1 if and only if the corresponding barcode is synthesized in the corresponding synthesis cycle. We offer a tool to calculate the corresponding synthesis schedules with a leftmost embedding for a set of barcodes. Other embedding strategies can also be used. The actual optimization then works on the synthesis schedules. The errors of two neighboring barcodes are determined by the Hamming distance of their schedules. An assumption must now be made as to where we expect the light scattering to occur. Two obvious options are to choose a neighborhood of four or eight. We now want to place the barcodes in such a way that the sum of the Hamming distance of all synthesis schedules to the neighbors is minimized.

For more details we refer to our paper cited below.

## Project Organization

This project is separated in several subprojects, each associated to some heuristic. Each subproject contains a README file in which the installation process and command line options are explained. At the moment, the project contains the following subprojects:

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

In the article below, we evaluated the algorithms on several barcode sets. You may find these barcodes in the [Data](https://github.com/uni-halle/BarLay/tree/main/data) directory. 

The barcode sets have been generated by rejection sampling. Each barcode was randomly generated and rejected if it didn't fulfill certain constraints. The files are named after the following scheme: 


    <constraint>_l<barcode length>_n<set size>[_d<distance>].zip


The table below explains the constraint names:

| Name        | Constraint                                                       |
|-------------|------------------------------------------------------------------|
| random      | no constraints                                                   |
| GC          | number of bases G and C is 40-60%                                |
| maxCycles   | number of synthesis cycles is less than 93                       |
| repeats     | no homopolymers and no repeats of length $\geq3$                 |
| constrained | contraints GC + maxCycles + no repeats                           |
| distance    | constrained + pairwise Sequence-Levenshtein distance at least 9  |

## Citation

To cite this work in publication, please use
> Frederik Jatzkowski, Antonia Schmidt, Robert Mank, Steffen Schüler, and Matthias Müller-Hannemann,
> _Barcode Selection and Layout Optimization in Spatial Transcriptomics_,
> Proceedings of the 22nd International Symposium on Experimental Algorithms (SEA 2024), DOI: https://doi.org/10.4230/LIPIcs.SEA.2024.17.
