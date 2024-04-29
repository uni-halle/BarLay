# BarLay - Barcode Layout Optimization
## 2-OPT Local Search

### Installation

1. Create a build directory.

   ````
     mkdir build
     cd build
   ````
   
2. Build the program.

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
