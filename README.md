# BarLay - Barcode Layout Optimization

## Problem Description

todo

## Project Organization

This project is separated in several subprojects, each associated to some heuristic. Each subproject contains with a README file in which the installation process and command line options are explained. 

## Input and Output Format

Input and output are text files with one barcode per line. The barcodes are given in row-major-order. Thus, the barcodes in row i are the ones in the lines i*col_count + 1, i*col_count + 2, ..., (i+1)*col_count. For example, a layout file could start with the lines

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



## Citation

You are free to use this code for your own work. Please cite todo
