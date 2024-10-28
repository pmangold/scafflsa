import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
palette = sns.color_palette("colorblind")

import csv

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import pickle
import glob
import os

plt.rcParams.update({
    'font.size' : 22,                  
    'axes.labelsize': 24,              
    'legend.fontsize': 18,             
    'font.family': 'lmodern',
    'text.usetex': True
})

markers = ["o", "+", "^", "x"]

T = "1000"
results_dir = "results_bias/" + T + "/*.pkl"
os.makedirs("tables/" + T + "/", exist_ok=True)


steps = [0.001, 0.01, 0.1, 1]
Hs = [1, 10, 100]
Ns = [1, 10, 100, 1000]
hg_kerns = [0.02, 10]

table_fedtd = {}
bias_fedtd = {}
table_scafftd = {}

# plot the legend
handles = [plt.plot([],[],marker=markers[0], color=palette[0], ls="solid")[0],
           plt.plot([],[],marker=markers[1], color=palette[1], ls="solid")[0],
           plt.plot([],[],marker=markers[2], color=palette[2], ls="solid")[0],
           plt.plot([],[],marker=markers[2], color=palette[3], ls="solid")[0],
           plt.plot([],[],color="black", ls="dashed")[0],
           plt.plot([],[],color="black", ls="solid")[0]
           ]
labels = ["$\\eta = 0.001$", "$\\eta = 0.01$", "$\\eta=0.1$", "$\\eta=1$", "FedLSA", "SCAFFLSA"]
legend = plt.legend(handles, labels, frameon=False, ncol=6)

plt.axis('off')
fig  = legend.figure
bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("tables/" + T + "/" + "legend.pdf", dpi="figure", bbox_inches=bbox)



def save_table(results_fedtd, results_scafftd, results_bias_fedtd, nlocal, hg_kern, name):

    data_bias_fedtd = [ [0 for _ in Ns] for _ in steps ]
    data_fedtd = [ [0 for _ in Ns] for _ in steps ]
    min_fedtd = [ [0 for _ in Ns] for _ in steps ]
    max_fedtd = [ [0 for _ in Ns] for _ in steps ]
    data_scafftd = [ [0 for _ in Ns] for _ in steps ]
    min_scafftd = [ [0 for _ in Ns] for _ in steps ]
    max_scafftd = [ [0 for _ in Ns] for _ in steps ]


    for key, value in results_fedtd.items():
        nenvs_, nlocal_, step_, hg_kern_ = key
        if nlocal_ == nlocal and hg_kern_ == hg_kern:
            data_fedtd[steps.index(step_)][Ns.index(nenvs_)] = value[0]
            min_fedtd[steps.index(step_)][Ns.index(nenvs_)] = value[1]
            max_fedtd[steps.index(step_)][Ns.index(nenvs_)] = value[2]


    for key, value in results_bias_fedtd.items():
        nenvs_, nlocal_, step_, hg_kern_ = key
        if nlocal_ == nlocal and hg_kern_ == hg_kern:
            data_bias_fedtd[steps.index(step_)][Ns.index(nenvs_)] = value

    for key, value in results_scafftd.items():
        nenvs_, nlocal_, step_, hg_kern_ = key
        if nlocal_ == nlocal and hg_kern_ == hg_kern and nenvs in Ns and step_ in steps:
            data_scafftd[steps.index(step_)][Ns.index(nenvs_)] = value[0]
            min_scafftd[steps.index(step_)][Ns.index(nenvs_)] = value[1]
            max_scafftd[steps.index(step_)][Ns.index(nenvs_)] = value[2]

        

    # make plot

    for plot_name in ["main", "bias"]:

        fig, ax = plt.subplots(1, 1, figsize=(4,3))
        for i, step in enumerate(steps):

        
            plt.plot(Ns, 1/np.array(Ns) * data_fedtd[i][0], label="FedTD -- step=" + str(step),
                     color="black",
                     ls="dotted")
        
            plt.plot(Ns, data_fedtd[i], label="FedTD -- step=" + str(step),
                     marker=markers[i],
                     color=palette[i],
                     ls="dashed")
            plt.fill_between(Ns,
                             min_fedtd[i],
                             max_fedtd[i],
                             color=palette[i],
                             edgecolor=palette[i],
                             alpha=0.3)
            plt.plot(Ns, data_scafftd[i], label="ScaffTD -- step=" + str(step),
                     marker=markers[i],
                     color=palette[i])
            plt.fill_between(Ns,
                             min_scafftd[i],
                             max_scafftd[i],
                             color=palette[i],
                             edgecolor=palette[i],
                             alpha=0.3)

            if plot_name == "bias":
                plt.plot(Ns, data_bias_fedtd[i], label="FedTD -- step=" + str(step),
                         color=palette[i])

            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Number of agents")

            ax.set_xticks([1, 10, 100, 1000], ["$1$", "$10^{1}$", "$10^{2}$", "$10^{3}$"])
            plt.savefig("tables/" + T + "/" + plot_name + "_" + name + ".pdf", bbox_inches="tight")
    
    # make table
    data = [ data_fedtd[i] + data_scafftd[i]  for i in range(len(steps)) ]
    
    with open('tables/' + name + ".tex", 'w', newline='') as f:
        f.write("\\begin{tabular}{c|cccc|cccc}")
        f.write("$\\step$ & $\\nagent = 1$ & $\\nagent = 10$ & $\\nagent = 100$ & $\\nagent = 1000$ & $\\nagent = 1$ & $\\nagent = 10$ & $\\nagent = 100$ & $\\nagent = 1000$ \\\\ \n")
        for i, dataline in enumerate(data):
            f.write("$\\step=" + str(steps[i]) + "$ & $ " + " $ & $".join(['{:.2e}'.format(value) for value in dataline]) + " $ \\\\ \n")
        f.write("\\end{tabular}")
        
    

for path in glob.glob(results_dir):
    name = path.split("/")[-1][:-4]
    with open(path, 'rb') as f:
        results = pickle.load(f)
        mean_fedtd = np.array(results["l2_norm_fedtd"]).mean(axis=0)
        mean_fedtd_debiased = np.array(results["l2_norm_fedtd_debiased"]).mean(axis=0)
        mean_scafftd = np.array(results["l2_norm_scafftd"]).mean(axis=0)
        min_fedtd = np.array(results["l2_norm_fedtd"]).min(axis=0)
        max_fedtd = np.array(results["l2_norm_fedtd"]).max(axis=0)
        min_scafftd = np.array(results["l2_norm_scafftd"]).min(axis=0)
        max_scafftd = np.array(results["l2_norm_scafftd"]).max(axis=0)
        fedtd_bias = np.linalg.norm(results["fedtd_bias"])**2

        nenvs = results["params"]["nenvs"]
        nlocal = results["params"]["n_local"]
        step = results["params"]["step"]
        hg_kern = results["params"]["hg_kern"]

        table_fedtd[(nenvs, nlocal, step, hg_kern)] = (mean_fedtd[-1], min_fedtd[-1], max_fedtd[-1])
        bias_fedtd[(nenvs, nlocal, step, hg_kern)] = fedtd_bias
        table_scafftd[(nenvs, nlocal, step, hg_kern)] = (mean_scafftd[-1], min_scafftd[-1], max_scafftd[-1])

for nlocal in Hs:
    for hg_kern in hg_kerns:
        save_table(table_fedtd, table_scafftd, bias_fedtd, nlocal, hg_kern, "results_" + str(nlocal) + "_" + str(hg_kern))
