import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'serif'
plt.rcParams["mathtext.fontset"] = "cm"
latex_path = '/usr/local/texlive/2024/bin/universal-darwin/'
if latex_path not in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + latex_path
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'

print(os.curdir)

specs = [np.loadtxt(str(fname)) for fname in ["processed_spectra_zbin{}.txt".format(num) for num in [0,1,2,3]]]
zbins = [0.1, 0.478, 0.783, 1.5]
specnum = np.random.randint(1, len(specs[0][:,0]))
for i, spec in enumerate(specs):
    zbin = zbins[i]
    plt.figure()
    plt.title("Processed Spectra for z = {}".format(zbin), fontsize=16)
    plt.plot(spec[0,1:], spec[specnum,1:])
    plt.xlabel(r'k', fontsize= 13)
    plt.ylabel(r'$P(k)/P_{\textrm{Planck}}(k)$ - 1', fontsize=13)
    plt.show()