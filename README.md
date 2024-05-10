# BarLay - Barcode Layout Optimization

## Problem Description

Given a rectangular grid, the **barcode layout problem** asks for an assignment of so called _barcodes_ (i.e. short DNA sequences of fixed length) to the grid positions such that a certain error function is minimized. It is an important subproblem in bioinformatics, in particular in the field of spatial transcriptomics.



## Project Organization

This project is separated in several subprojects, each associated to some heuristic. Each subproject contains with a README file in which the installation process and command line options are explained. 

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

## Citation

You are free to use this code for your own work. Please cite todo
