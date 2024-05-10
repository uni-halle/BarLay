# BarLay - Barcode Layout Optimization
## 2-OPT Local Search
### Description

Given an initial layout, the algorithm exhaustively searches for possible swaps of two barcodes that would improve the layout cost. Afterwards, the algorithm performs these swaps and iterates until no swap can improve the cost.

### Installation

1. Download the source files.

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


4. Run the program.

   ````
   ./2opt ../../data/example.txt
   ````
   
### Program Options

The program expects a single command line argument: The filename of a barcode file. 
