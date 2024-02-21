The various file functions (run them in this order): 

1. 
relabel_params.cpp: Looks at the target parameter file and creates a new one with cosmologies ordered with selected parameter (set the index in the file - line 48) in ascending order. Also outputs the ordered indices to the terminal which should be copied and pasted in relabel_data.sh. 

2. 
relabel_data.sh: copies example .txt files in the unordered data directory, relabels their file name to the ordered index and moves them to a new folder where they are ordered.

3. 
bin_split_data.sh: Iteratively copies sets of 5k files from the ordered data directory and places them in a separate folder. 

4. 
bin_split_params.cpp : Creates binned param file from the (ordered)full param file, created in relabel_params.cpp 


cpp files can be compiled and run with a g++ or gcc compiler: 

> g++ relabel_params.cpp 
> ./a.out 


The default settings bin data from the full 20k training data set into ordered neutrino energy density fraction (index 3). It splits these into 4 bins of 5k examples each. 