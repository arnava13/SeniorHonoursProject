import os
import shutil
import glob
import numpy as np
import sys

def main():
    # Get arguments from command line
    mode = sys.argv[1]
    spectra_source_dir = sys.argv[2]
    filters_source_dir = sys.argv[3]
    outdir = sys.argv[4]
    n = int(sys.argv[5])
    start_spectrum_number = int(sys.argv[6])  # Adjust for 1-based indexing
    start_filter_number = int(sys.argv[7])  # Adjust for 1-based indexing

    if mode == 'equal':
        # Get all directories in the source directory
        classnames = glob.glob(os.path.join(spectra_source_dir, '*'))

        # Get number of classes
        m = len(classnames)

    if mode == 'lcdm':
        m = 1

    # Generate an empty array to store the spectra
    spectra_array = np.zeros([m * n, 500, 5])

    # Get all files in the filters directory
    filter_files = glob.glob(os.path.join(filters_source_dir, '*'))
    filter_files.sort()

    # Get n filter files starting from the start file
    filters = filter_files[start_filter_number - 1 : start_filter_number + n - 1]  # Adjust for 1-based indexing

    if mode == 'equal':
        # Loop through each directory
        for i, directory in enumerate(classnames):
            # Get all files in the directory
            spectrum_files = glob.glob(os.path.join(directory, '*'))
            spectrum_files.sort()
            # Get n spectrum files starting from the start file
            spectra = spectrum_files[start_spectrum_number - 1 : start_spectrum_number + n - 1]  # Adjust for 1-based indexing
            for j, spectrum in enumerate(spectra):
                with open(spectrum, 'r') as f:
                    spectrum_lines = f.readlines()
                    spectrum_lines = [line.strip() for line in spectrum_lines]
                for k, line in enumerate(spectrum_lines):
                    line = line.split()
                    line = np.array(line, dtype = float)
                    spectra_array[i*n+j, k] = line
    
    if mode == 'lcdm':
        spectrum_files = glob.glob(os.path.join(spectra_source_dir, '*'))
        spectra = spectrum_files[start_spectrum_number - 1 : start_spectrum_number + n - 1]  # Adjust for 1-based indexing
        for j, spectrum in enumerate(spectra):
            with open(spectrum, 'r') as f:
                spectrum_lines = f.readlines()
                spectrum_lines = [line.strip() for line in spectrum_lines]
            for k, line in enumerate(spectrum_lines):
                line = line.split()
                line = np.array(line, dtype = float)
                spectra_array[j, k] = line
        classnames = ['lcdm']

    filter_files = glob.glob(os.path.join(filters_source_dir, '*'))

    # Sort files by name
    filter_files.sort()

    # Get the first 5n filters
    filters = filter_files[:5*n]
    print(m)
    filters_array = np.zeros([m * n, 500, 5])
    for i in range(m):
        for j in range(n):
            filter = filters[i*n + j]
            with open(filter, 'r') as f:
                filter_lines = f.readlines()
                filter_lines = [line.strip() for line in filter_lines]
            for k, line in enumerate(filter_lines):
                    line = line.split()
                    line = np.array([float(l) for l in line])
                    line[0] = 1.0
                    if mode == 'lcdm':
                        filters_array[j, k] = line
                    if mode == 'equal':
                        filters_array[i*n+j, k] = line

    # Apply filters and save filtered spectra
    for i, classname in enumerate(classnames):
        for j in range(n):
            if mode == 'lcdm':
                filtered_spectrum = np.multiply(spectra_array[j], filters_array[j])
            if mode == 'equal':
                filtered_spectrum = np.multiply(spectra_array[i*n+j], filters_array[i*n+j])
            # Reshape filtered_spectrum into chunks of five
            filtered_spectrum = filtered_spectrum.reshape(-1, 5)
            # Convert each chunk into a string and join them with a space to create a line
            filtered_spectrum = [" ".join(str(line) for line in chunk) for chunk in filtered_spectrum]
            # Join each line with a newline
            filtered_spectrum = '\n'.join(filtered_spectrum)
            outpath = os.path.join(outdir, f'{j + i*n + 1}.txt')
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with open(outpath, 'w') as f:
                f.write(filtered_spectrum)


if __name__ == "__main__":
    main()