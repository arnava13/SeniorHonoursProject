#!/bin/bash
echo "Initializing data reordering process..."
echo ""

cd "$(dirname "$0")"

indir=#YOUR_INPUT_UNORDERED_DATA_DIRECTORY
outdir=#YOUR_OUTPUT_ORDERED_DATA_DIRECTORY
unorderedTxt=#UNORDERED_COSMOLOGY_FILE_FOR_ENTIRE_DATASET
orderedTxt=#OUTPUT_ORDERED_COSMOLOGY_FILE_FOR_DATA_SUBSET
indicesFile=#FILE_CONTAINING_ORDERED_ORIGINAL_INDICES 

# Ensure the ordered directory exists and is empty
mkdir -p "$outdir/"
rm -f "$outdir/*"

# Ensure the ordered cosmology file is empty or created
: > "$orderedTxt"

# Initialize an empty array for new indices
declare -a mynewindices=()

# Read indices from the file into an array
read -ra mynewindices < "$indicesFile"

# Initialize counter for output file numbering
counter=1

for index in "${mynewindices[@]}"; do
    # Check if file exists in the input directory
    if [[ -f "${indir}/${index}.txt" ]]; then
        # Copy file to the output directory with new numbering
        cp "${indir}/${index}.txt" "${outdir}/${counter}.txt"
        # Find the corresponding line in the unorderedTxt and append to orderedTxt
        originalLine=$(awk -v id="${index}" '$1 == id {print; exit}' "$unorderedTxt")
        if [[ -n "$originalLine" ]]; then
            echo "$counter ${originalLine#* }" >> "$orderedTxt"
        fi
        ((counter++))
    fi
done

echo "Data reordering complete. Check $outdir for ordered files and $orderedTxt for the corresponding cosmology parameters."
