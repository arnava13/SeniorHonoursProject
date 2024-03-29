import sys
import os
import glob
import numpy as np

def main():
    mode = sys.argv[1]
    spectra_source_dir = sys.argv[2]
    filters_source_dir = sys.argv[3]
    outdir = sys.argv[4]
    n = int(sys.argv[5])
    start_spectrum_index = int(sys.argv[6]) - 1  # Adjust for 0-based indexing
    start_filter_index = int(sys.argv[7]) - 1  # Adjust for 0-based indexing

    if mode == 'equal':
        classnames = glob.glob(os.path.join(spectra_source_dir, '*'))
        m = len(classnames)
    else:  # Assuming 'lcdm' mode
        m = 1

    spectra_array = np.zeros([n, 500, 5])

    filter_files = sorted(glob.glob(os.path.join(filters_source_dir, '*')))
    filters = filter_files[start_filter_index : start_filter_index + n]

    if mode == 'equal':
        spectra_per_class = int(n / m)
    else:  # 'lcdm' mode
        spectra_per_class = n

    total_spectra_count = 0
    for i, classname in enumerate(classnames if mode == 'equal' else [spectra_source_dir]):
        spectrum_files = sorted(glob.glob(os.path.join(classname, '*')))
        spectra = spectrum_files[start_spectrum_index : start_spectrum_index + spectra_per_class]

        for j, spectrum in enumerate(spectra):
            with open(spectrum, 'r') as f:
                for k, line in enumerate(f):
                    values = np.array(line.strip().split(), dtype=float)
                    spectra_array[total_spectra_count + j, k, :] = values

        total_spectra_count += len(spectra)

    filters_array = np.zeros_like(spectra_array)
    for i, filter_path in enumerate(filters):
        with open(filter_path, 'r') as f:
            for k, line in enumerate(f):
                values = np.array(line.strip().split(), dtype=float)
                values[0] = 1.0  # Set first column to 1
                filters_array[i, k, :] = values

    for i in range(n):
        filtered_spectrum = np.multiply(spectra_array[i], filters_array[i])
        output_lines = [' '.join(map(str, row)) for row in filtered_spectrum]
        output_path = os.path.join(outdir, f'{i + 1}.txt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_lines))

if __name__ == "__main__":
    main()
