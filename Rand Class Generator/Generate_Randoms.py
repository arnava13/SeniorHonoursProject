import sys
import os
import glob
import numpy as np

def load_and_process_spectrum(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return np.array([list(map(float, line.strip().split())) for line in lines])

def load_and_modify_filter(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    filter_data = np.array([list(map(float, line.strip().split())) for line in lines])
    filter_data[:, 0] = 1  # Replace first column with 1s
    return filter_data

def main():
    mode = sys.argv[1]
    spectra_source_dir = sys.argv[2]
    filters_source_dir = sys.argv[3]
    outdir = sys.argv[4]
    n = int(sys.argv[5])
    start_spectrum_index = int(sys.argv[6]) - 1
    start_filter_index = int(sys.argv[7]) - 1

    if mode == 'equal':
        classnames = sorted(glob.glob(os.path.join(spectra_source_dir, '*')))
        m = len(classnames)
        spectra_per_class = n // m
    else:
        classnames = [spectra_source_dir]
        m = 1
        spectra_per_class = n

    filter_files = sorted(glob.glob(os.path.join(filters_source_dir, '*')))[start_filter_index:start_filter_index + n]

    total_processed = 0
    for i, classname in enumerate(classnames):
        spectrum_files = sorted(glob.glob(os.path.join(classname, '*')))[start_spectrum_index:]

        for j in range(spectra_per_class):
            if total_processed >= n:
                break
            if j >= len(spectrum_files):
                break

            spectrum = load_and_process_spectrum(spectrum_files[j])
            filter_data = load_and_modify_filter(filter_files[total_processed])

            filtered_spectrum = spectrum * filter_data
            outpath = os.path.join(outdir, f'{total_processed + 1}.txt')
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            np.savetxt(outpath, filtered_spectrum, fmt='%f')

            total_processed += 1

if __name__ == "__main__":
    main()
