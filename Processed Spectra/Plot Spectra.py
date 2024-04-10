import numpy as np
import matplotlib.pyplot as plt
import os


print(os.curdir)
specs = [np.loadtxt(str(fname)) for fname in ["processed_spectra_zbin{}.txt".format(num) for num in [0,1,2,3]]]
for spec in specs:
    plt.figure()
    plt.plot(spec[0,1:], spec[1,1:])
    plt.xlabel('k')
    plt.ylabel('P(k)/P_planck(k) - 1')
    plt.show()

"""spec = np.loadtxt("planck_ee2.txt")
plt.figure()
plt.plot(spec[:,0], spec[:,1])
plt.show()"""