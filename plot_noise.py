"""Plots cosmic variance and a random theory error curve, as well as the total noise."""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    const = 2*np.pi / np.sqrt(V*delta_k)
    sig = const/k
    return(sig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('theoryerr_dir', type=str, help='Path to the directory containing the theory error curves, or single curve if in single mode')
    parser.add_argument('--sigma_curves', type=float, default=0.05, help='The scale factor for the theory error curves used in training.')
    parser.add_argument('--sigma_curves_default', type=float, default=0.05, help='Generation scale factor of theory error curves.')
    parser.add_argument('--delta_k', type=float, default=0.055, help='Spacing of k values.')
    args = parser.parse_args()
    theoryerr_dir = args.theoryerr_dir
    sigma_curves = args.sigma_curves
    sigma_curves_default = args.sigma_curves_default

    random_curve_num = np.random.randint(0, 1000)
    single_therr_path = os.path.join(theoryerr_dir, str(random_curve_num) + ".txt")
    theoryerr = np.loadtxt(single_therr_path)[:,1:]
    k_values = np.loadtxt(single_therr_path)[:,0]

    colors = cm.get_cmap('Reds', 10)

    # Create figures outside of the loop
    fig_theory_error = plt.figure(1)
    plt.title("Theory Error", fontsize = 16)
    plt.xlabel("k", fontsize = 13)
    plt.ylabel("Fractional Error", fontsize = 13)
    plt.legend()
    plt.xscale('log')

    fig_cosmic_variance = plt.figure(2)
    plt.title("Cosmic Variance", fontsize = 16)
    plt.xlabel("k", fontsize = 13)
    plt.ylabel("Fractional Error", fontsize = 13)
    plt.legend()
    plt.xscale('log')

    fig_total_noise = plt.figure(3)
    plt.title("Total Noise", fontsize = 16)
    plt.xlabel("k", fontsize = 13)
    plt.ylabel("Fractional Error", fontsize = 13)
    plt.legend()
    plt.xscale('log')


    theory_errors = []
    cosmic_variances = []
    total_noises = []

    for i, z in enumerate([1.5, 0.783, 0.478, 0.1]):
        theoryerr_forz = (1 - theoryerr[:,i]) * sigma_curves/sigma_curves_default
        cosmicvariance = np.array([])
        for k in k_values:
            sig = sigma(k, z, k_values)
            cosmicvariance = np.append(cosmicvariance, sig)

        theory_errors.append(theoryerr_forz)
        cosmic_variances.append(cosmicvariance)
        total_noises.append(np.abs(theoryerr_forz)+cosmicvariance)
            
        plt.figure(i+4)  # Refer to the figure for the current z value
        plt.title("Absolute Theory Error and Cosmic Variance \n z = " + str(z), fontsize = 16)
        plt.xlabel("k", fontsize = 13)
        plt.ylabel("Fractional Error", fontsize = 13)
        plt.plot(k_values, cosmicvariance, 'k-', label="Cosmic Variance")
        plt.xlim(0.01, 5)
        plt.xscale('log')
        plt.ylim(0, 0.075)

        for j in range(10):
            random_curve_num = np.random.randint(0, 1000)
            single_therr_path = os.path.join(theoryerr_dir, str(random_curve_num) + ".txt")
            theoryerr_file = np.loadtxt(single_therr_path)[:,1:]
            therr = (1 - theoryerr_file[:,i]) * sigma_curves/sigma_curves_default
            if j == 1:
                plt.plot(k_values, therr, color=colors(j/10), label="Theory Error")
            else:
                plt.plot(k_values, therr, color=colors(j/10), label="_nolegend_")
        plt.legend()


    for i, z in enumerate([1.5, 0.783, 0.478, 0.1]):
        plt.figure(fig_theory_error.number)
        plt.plot(k_values, theory_errors[i], label="z = " + str(z), color='red')
        plt.figure(fig_cosmic_variance.number)
        plt.plot(k_values, cosmic_variances[i], label="z = " + str(z), color='red')
        plt.figure(fig_total_noise.number)
        plt.plot(k_values, total_noises[i], label="z = " + str(z), color='red')
   
    plt.show()
        
if __name__ == "__main__":
    main()