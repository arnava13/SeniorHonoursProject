# Generate_Randoms.py

This script is used to generate random spectra using an equal number of spectra from each class and applying a different filter to each spectra.

## Usage

python Generate_Randoms.py <spectra_source_dir> <filters_source_dir> <outdir> <n> <start_spectrum_number> <start_filter_number>

1. `spectra_source_dir`: Directory containing the spectra in class subfolders.
2. `filters_source_dir`: Directory contain
3. `outdir`: The output directory where the generated files will be stored.
4. `n`: The number of examples to use from each class of spectra (total number of random spectra will be n * the number of non-random classes)
5. `start_spectrum_number`: The starting index for the spectra files (1-based indexing).
6. `start_filter_number`: The starting index for the filter files (1-based indexing).

## Notes

It is recommended to use the spectra files at the end of the dataset to generate randoms, as a certain number counting up are used in training/testing.

For example, we need for our training set 1000 random spectra and we have 5 non-random classes (n = 200). There are 20k spectra in the whole provided dataset we use start_spectrum_number = 19800, and we use start_filter_number = 0.

Then, for our test set, let's say we need 10000 random spectra (n = 2000). We use start_spectrum_number = 19799-2000 = 17799, and we use start_filter_number = 200+2000 = 2200.
