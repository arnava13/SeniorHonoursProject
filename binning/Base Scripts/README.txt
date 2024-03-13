The various file functions (run them in this order): 

1. 
order_indices.cpp: Looks at the target parameter file and creates a file with the indices ordered as increasing in the defined parameter (set the parameter index in the file - line 39).

2. 
relabel_data_params.sh: copies example .txt files in the unordered data directory, relabels their file name to the ordered index and moves them to a new folder where they are ordered. Indices which were in the original cosmology file but aren't in the data folder are skipped, and a new cosmology file corresponding to the ordered data is generated.

3. 
bin_split_data_params.sh: Iteratively copies equal sets files from the ordered data directory and places them in a separate folder. Creates individual cosmology files for each bin.


cpp files can be compiled and run with a g++ or gcc compiler: 

> g++ relabel_params.cpp 
> ./a.out 
