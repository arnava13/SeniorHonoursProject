#!/bin/bash
echo "Initializing training data"
echo ""

# Navigate to the script's directory
cd "$(dirname "$0")"
echo "Working directory: $PWD"

# Initialize variables
indir="/Users/arnav/shp_dat/Data/Binned_DS_Data/test/ordered_all/ds_test_ordered" # Path to input directory
n_perbin=250 # Number of examples per bin, assuming a total of 100 for simplicity
cosmologyFile="/Users/arnav/shp_dat/Data/Binned_DS_Data/test/ordered_all/cosmo_ds_test_ordered.txt" # Path to the ordered cosmology file

# Function to create bins and corresponding text files
create_bin() {
    local binName=$1
    local binFile="${binName}.txt"
    local start=$((($2 - 1) * n_perbin + 1))
    local end=$(($2 * n_perbin))

    echo "Creating ${binName}..."
    mkdir $binName

    for ((i = start; i <= end; i++)); do
        local outputFile=$((i - start + 1))
        echo "Adding ${i}.txt to ${binName} as ${outputFile}.txt"
        cp "${indir}/${i}.txt" "${binName}/${outputFile}.txt"
        # Extract the entire row from the cosmology file, prepending the new file index
        echo "${outputFile} $(awk -v id="$i" 'NR==id {print $0}' $cosmologyFile)" >> $binFile
    done
}

# Create bins A, B, C, D
create_bin "binA" 1
create_bin "binB" 2
create_bin "binC" 3
create_bin "binD" 4

echo "Processing DONE!!! :D"
