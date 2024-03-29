import sys
import os
import glob
import numpy as np

def load_and_apply_filter(spectrum_file, filter_file):
    # Load spectrum and filter, modify filter's first column, and apply it to the spectrum
    spectrum = np.loadtxt(spectrum_file)
    filter_data = np.loadtxt(filter_file)
    filter_data[:, 0] = 1  # Set the first column to 1s
    return spectrum * filter_data

def main():
    mode = sys.argv[1]
    spectra_source_dir = sys.argv[2]
    filters_source_dir = sys.argv[3]
    outdir = sys.argv[4]
    n = int(sys.argv[5])
    start_spectrum_index = int(sys.argv[6]) - 1  # Adjusting for 1-based indexing
    start_filter_index = int(sys.argv[7]) - 1  # Adjusting for 1-based indexing

    filter_files = sorted(glob.glob(os.path.join(filters_source_dir, '*.txt')))
    if not filter_files:
        print("No filter files found.")
        return

    if mode == 'equal':
        classnames = sorted([d for d in os.listdir(spectra_source_dir) if os.path.isdir(os.path.join(spectra_source_dir, d))])
        m = len(classnames)
        total_spectra_needed = n
        spectra_per_class = max(1, total_spectra_needed // m)
        extra_spectra = total_spectra_needed % m

        output_counter = 0
        for i, classname in enumerate(classnames):
            class_dir = os.path.join(spectra_source_dir, classname)
            spectrum_files = sorted(glob.glob(os.path.join(class_dir, '*.txt')))[start_spectrum_index:]
            spectra_to_process = spectra_per_class + (1 if i < extra_spectra else 0)

            for spectrum_file in spectrum_files[:spectra_to_process]:
                if output_counter >= n:
                    break

                filter_file = filter_files[(start_filter_index + output_counter) % len(filter_files)]
                filtered_spectrum = load_and_apply_filter(spectrum_file, filter_file)

                output_path = os.path.join(outdir, f'{output_counter + 1}.txt')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.savetxt(output_path, filtered_spectrum, fmt='%f')

                output_counter += 1

    elif mode == 'lcdm':
        spectrum_files = sorted(glob.glob(os.path.join(spectra_source_dir, '*.txt')))[start_spectrum_index:start_spectrum_index + n]

        for i, spectrum_file in enumerate(spectrum_files):
            if i >= n:
                break

            filter_file = filter_files[(start_filter_index + i) % len(filter_files)]
            filtered_spectrum = load_and_apply_filter(spectrum_file, filter_file)

            output_path = os.path.join(outdir, f'{i + 1}.txt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savetxt(output_path, filtered_spectrum, fmt='%f')

if __name__ == "__main__":
    main()
