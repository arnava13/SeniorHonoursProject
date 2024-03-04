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
    args = parser.parse_args()
    spectrum_dir = args.spectrum_dir
    theoryerr_dir = args.theoryerr_dir
    theoryerror_mode = args.theoryerr_mode
    sigma_curves = args.sigma_curves

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
            sys.exit(1)

    if theoryerror_mode == 'averaged':
        if theoryerr_dir.endswith(".txt"):
            raise Exception('In averaged curve mode, a directory must be provided containing the theory error curves')
            sys.exit(1)
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

    plt.figure(1)
    plt.xlabel('k')
    plt.xscale('log')
    plt.ylabel('Theory Error (Scaled)')
    plt.title('Theory Error for Different Redshifts')

    plt.figure(2)
    plt.xlabel('k')
    plt.xlim(0,0.1)
    plt.ylabel('Cosmic Variance')
    plt.title('Cosmic Variance for Different Redshifts')

    plt.figure(3)
    plt.xlabel('k')
    plt.xscale('log')
    plt.xlim(0.01,3.0)
    plt.ylabel('Total Noise')
    plt.title('Total Noise for Different Redshifts')

    for z in [1.5, 0.783, 0.478, 0.1]:
        theoryerror_forz = np.array([])
        cosmicvariance_forz = np.array([])
        noise = np.array([])
        for k in k_values:
            k_index = np.where(k_values == k)[0][0]
            sig = sigma(k, z, k_values)
            theoryerr_ind = sigma_curves * theoryerr[str(z)].iloc[k_index]
            cosmicvariance_forz = np.append(cosmicvariance_forz, sig)
            theoryerror_forz = np.append(theoryerror_forz, theoryerr_ind)
            indnoise = sig + theoryerr_ind
            noise = np.append(noise, indnoise)

        plt.figure(1)
        plt.plot(k_values, theoryerror_forz, label=f'z={z}')

        plt.figure(2)
        plt.plot(k_values, cosmicvariance_forz, label=f'z={z}')

        plt.figure(3)
        plt.plot(k_values, theoryerror_forz + cosmicvariance_forz, label=f'z={z}')

    plt.figure(1)
    plt.legend()

    plt.figure(2)
    plt.legend()

    plt.figure(3)
    plt.legend()

    plt.show()
    
            
    
if __name__ == "__main__":
    main()