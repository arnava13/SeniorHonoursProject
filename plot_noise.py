import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def sigma(k, z, k_values, delta_k = 0.055):
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
    sig = np.sqrt(constsquared/(delta_k*k**2))
    return(sig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spectrum_dir', type=str, help='Path to the data file')
    parser.add_argument('theoryerr_dir', type=str, help='Path to the directory containing the theory error curves, or single curve if in single mode')
    parser.add_argument('--theoryerr_mode', type=str, default='averaged', help='Mode for theory error. Options are \'single\' or \'averaged\'')
    parser.add_argument('--sigma_curves', type=float, default=0.05, help='The scale factor for the theory error curves used in training.')
    parser.add_argument('--sigma_curves_default', type=float, default=0.05, help='Generation scale factor of theory error curves.')
    parser.add_argument('--delta_k', type=float, default=0.055, help='Spacing of k values.')
    args = parser.parse_args()
    spectrum_dir = args.spectrum_dir
    theoryerr_dir = args.theoryerr_dir
    theoryerror_mode = args.theoryerr_mode
    sigma_curves = args.sigma_curves
    sigma_curves_default = args.sigma_curves_default

    plt.figure(1, figsize=(7, 5))
    plt.xlabel('k')
    plt.xlim(0.01,5.0)
    plt.ylabel('Theory Error (Scaled)')
    plt.title('Theory Error Across Redshifts')

    plt.figure(2, figsize=(7, 5))
    plt.xlabel('k')
    plt.xlim(0.01,5.0)
    plt.ylabel('Cosmic Variance')
    plt.title('Cosmic Variance Across Redshifts')

    plt.figure(3, figsize=(7, 5))
    plt.xlabel('k')
    plt.ylabel('Total Noise')
    plt.title('Total Noise Across Redshifts')

    with open(spectrum_dir) as example_spectrum:
        example_spectrum = np.loadtxt(example_spectrum)
        k_values = example_spectrum[:,0]
    
    if theoryerror_mode == 'single':
        if theoryerr_dir.endswith(".txt"):
            theoryerr = np.loadtxt(theoryerr_dir)[:,1:]
        else:
            raise Exception('In single curve mode, a specific .txt file must be provided for the theory error.')

    if theoryerror_mode == 'averaged':
        if theoryerr_dir.endswith(".txt"):
            raise Exception('In averaged curve mode, a directory must be provided containing the theory error curves')
        else:
            files = [f for f in os.listdir(theoryerr_dir) if f.endswith('.txt')]
            curves_list = []
            for file in files:
                curve = np.loadtxt(os.path.join(theoryerr_dir, file))[:,1:]
                curves_list.append(curve)
            theoryerr = np.mean(curves_list, axis=0)

    for i, z in enumerate([1.5, 0.783, 0.478, 0.1]):
        theoryerr_forz = np.array([])
        cosmicvariance_forz = np.array([])
        for k in k_values:
            sig = sigma(k, z, k_values)
            theoryerr_forz = (theoryerr[:,i] - 1) * sigma_curves/sigma_curves_default
            cosmicvariance_forz = np.append(cosmicvariance_forz, sig)
    
        plt.figure(1)
        plt.plot(k_values, theoryerr_forz, label=f'z={z}')

        plt.figure(2)
        plt.plot(k_values, cosmicvariance_forz, label=f'z={z}')

        plt.figure(3)
        plt.plot(k_values, theoryerr_forz + cosmicvariance_forz, label=f'z={z}')

    plt.figure(1)
    plt.legend()

    plt.figure(2)
    plt.legend()

    plt.figure(3)
    plt.legend()

    plt.show()
            
    
if __name__ == "__main__":
    main()