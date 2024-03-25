import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import sys
import os

def sigma(k, z, k_values):
    """Calculate the cosmic variance level for a given k and z value."""
    if z == 1.5:
        V = 10.43e9
    elif z == 0.783:
        V = 6.27e9
    elif z == 0.478: 
        V = 3.34e9
    elif z == 0.1:
        V = 0.283e9
    else:
        raise(ValueError('z must be 1.5, 0.785, 0.478, or 0.1'))
    constsquared = 4*np.pi**2 / V
    k_index = np.where(k_values == k)[0][0]
    delta_k = np.abs(k_values[k_index] - k_values[k_index - 1])
    sig = np.sqrt(constsquared/(delta_k*k**2))
    return(sig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theoryerr_mode', type=str, default='averaged', help='Mode for theory error. Options are \'single\' or \'averaged\'')
    parser.add_argument('--sigma_curves', type=float, default=0.05, help='The scale factor for the theory error curves used in training.')
    parser.add_argument('spectrum_dir', type=str, help='Path to the data file')
    parser.add_argument('theoryerr_dir', type=str, help='Path to the directory containing the theory error curves')
    parser.add_argument('fracnoisediff', type=float, help='The minimum ratio of variation in theory error to cosmic variance desired for k > k_min')
    args = parser.parse_args()
    spectrum_dir = args.spectrum_dir
    theoryerr_dir = args.theoryerr_dir
    theoryerror_mode = args.theoryerr_mode
    sigma_curves = args.sigma_curves
    fracnoisediff = args.fracnoisediff

    with open(spectrum_dir) as example_spectrum:
        example_spectrum = pd.read_csv(example_spectrum, sep=r'\s+', header=None, engine='python')
        k_values = example_spectrum.iloc[:, 0].to_numpy()
    
    if theoryerror_mode == 'single':
        if theoryerr_dir.endswith(".txt"):
            with open(theoryerr_dir) as theory_err:
                theoryerr = pd.read_csv(theory_err, sep=r'\s+', header=None, engine='python')
                theoryerr.rename(columns={0: 'k', 1: '1.5', 2: '0.783', 3: '0.478', 4: '0.1'}, inplace=True)
        else:
            raise Exception('In single curve mode, a specific .txt file must be provided for the theory error.')

    if theoryerror_mode == 'averaged':
        if theoryerr_dir.endswith(".txt"):
            raise Exception('In averaged curve mode, a directory must be provided containing the theory error curves')
        else:
            files = [f for f in os.listdir(theoryerr_dir) if f.endswith('.txt')]
            df_list = []
            for file in files:
                with open(os.path.join(theoryerr_dir, file)) as f:
                    df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
                    df_list.append(df)
            theoryerr = pd.concat(df_list)
            theoryerr = theoryerr.groupby(theoryerr.index).mean()
    
    theoryerr.rename(columns={0: 'k', 1: '1.5', 2: '0.783', 3: '0.478', 4: '0.1'}, inplace=True)

    plt.figure(1, figsize=(7, 5))
    plt.xlabel('k')
    plt.xlim(0.01,5.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Theory Error (Scaled)')
    plt.title('Theory Error Across Redshifts')

    plt.figure(2, figsize=(7, 5))
    plt.xlabel('k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.01,5.0)
    plt.ylabel('Cosmic Variance')
    plt.title('Cosmic Variance Across Redshifts')

    plt.figure(3, figsize=(7, 5))
    plt.xlabel('k')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Total Noise')
    plt.title('Total Noise Across Redshifts')

    k_min_forzs = np.zeros(4)
    for z in [1.5, 0.783, 0.478, 0.1]:
        theoryerror_forz = np.array([])
        cosmicvariance_forz = np.array([])
        for i, k in enumerate(k_values):
            sig = sigma(k, z, k_values)
            theoryerr_ind = sigma_curves * theoryerr[str(z)].iloc[i]
            cosmicvariance_forz = np.append(cosmicvariance_forz, sig)
            theoryerror_forz = np.append(theoryerror_forz, theoryerr_ind)
    
        plt.figure(1)
        plt.plot(k_values, theoryerror_forz, label=f'z={z}')

        plt.figure(2)
        plt.plot(k_values, cosmicvariance_forz, label=f'z={z}')

        plt.figure(3)
        plt.plot(k_values, theoryerror_forz + cosmicvariance_forz, label=f'z={z}')

    k_min = np.max(k_min_forzs)
    z_index = np.where(k_min_forzs == k_min)[0][0]

    plt.figure(1)
    plt.legend()

    plt.figure(2)
    plt.legend()

    plt.figure(3)
    plt.xlim(k_min, 5.0)
    plt.ylim(0.05, 0.05 + y_var)
    plt.legend()

    plt.show()
            
    
if __name__ == "__main__":
    main()