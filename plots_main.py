import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
palette = sns.color_palette("colorblind")

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import pickle
import glob
import os

plt.rcParams.update({
    'font.size' : 18,
    'axes.labelsize': 20,
    'legend.fontsize': 14,
    'font.family': 'lmodern',
    'text.usetex': True
})

N = "500000.0"
results_dir = "results/" + N + "/*.pkl"
os.makedirs("plots/" + N + "/", exist_ok=True)

def ema(scalars, weight=0.5):
    last = scalars[0]  
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point 
        smoothed.append(smoothed_val)                       
        last = smoothed_val                           

    return np.array(smoothed)

def smooth(scalars, weight=0.1, start=0):
    return np.concatenate((scalars[:start], ema(scalars[start:], weight)))

def smooth_upper(scalars, variance, weight=0.1, start=0):
    return smooth(scalars + variance, weight=weight, start=start)

def smooth_lower(scalars, variance, min_val, weight=0.1, start=0):
    return smooth(np.maximum(scalars - variance, min_val), weight=weight, start=start)


for path in glob.glob(results_dir):
    name = path.split("/")[-1][:-4]

    print(name, "hg_kernel,0.02" in name)
    
    with open(path, 'rb') as f:
        results = pickle.load(f)
        mean_fedtd = np.array(results["l2_norm_fedtd"]).mean(axis=0)
        mean_fedtd_debiased = np.array(results["l2_norm_fedtd_debiased"]).mean(axis=0)
        mean_scafftd = np.array(results["l2_norm_scafftd"]).mean(axis=0)
        std_fedtd = np.array(results["l2_norm_fedtd"]).std(axis=0)
        std_fedtd_debiased = np.array(results["l2_norm_fedtd_debiased"]).std(axis=0)
        std_scafftd = np.array(results["l2_norm_scafftd"]).std(axis=0)
        min_fedtd = np.array(results["l2_norm_fedtd"]).min(axis=0)
        min_fedtd_debiased = np.array(results["l2_norm_fedtd_debiased"]).min(axis=0)
        min_scafftd = np.array(results["l2_norm_scafftd"]).min(axis=0)
        fedtd_bias = np.linalg.norm(results["fedtd_bias"])**2
        noise_level = np.linalg.norm(results["noise_level"])


        idx = np.linspace(0, len(mean_fedtd) - 1, min(len(mean_fedtd), 200), dtype=int)
        

        for plot_name in ["main", "debias"]:
            fig, ax = plt.subplots(1, 1, figsize=(4,3))
        
            plt.plot(idx, smooth(mean_fedtd[idx]),
                     color=palette[0], marker="o", label="FedLSA", markevery=int(len(idx)/10))
            plt.plot(idx, smooth(mean_scafftd[idx]),
                     color=palette[1], marker="x", label="SCAFFLSA", markevery=int(len(idx)/10))

            plt.fill_between(idx,
                             smooth_lower(mean_fedtd[idx], std_fedtd[idx], min_fedtd[idx], start=0),
                             smooth_upper(mean_fedtd[idx], std_fedtd[idx], start=0),
                             color=palette[0], edgecolor=palette[0],
                             alpha=0.5)
            plt.fill_between(idx,
                             smooth_lower(mean_scafftd[idx], std_scafftd[idx], min_scafftd[idx], start=0),
                             smooth_upper(mean_scafftd[idx], std_scafftd[idx], start=0),
                             color=palette[1], edgecolor=palette[1],
                             alpha=0.5)


            if plot_name == "debias":
                plt.plot(idx, smooth(mean_fedtd_debiased[idx]),
                         color=palette[4], marker="^", label="FedTD without bias", markevery=int(len(idx)/10))
                plt.fill_between(idx,
                                 smooth_lower(mean_fedtd_debiased[idx], std_fedtd_debiased[idx], min_fedtd_debiased[idx], start=0),
                                 smooth_upper(mean_fedtd_debiased[idx], std_fedtd_debiased[idx], start=0),
                                 color=palette[4], edgecolor=palette[4],
                                 alpha=0.5)


            xlims = ax.get_xlim()
                
            plt.plot([-max(idx),2*max(idx)], [fedtd_bias] * 2, color=palette[2], ls="dashed", lw=2, label="FedLSA's bias")


            if len(mean_fedtd_debiased) > 500:
                plt.xlabel("Communications (×" + str(len(mean_fedtd_debiased)//50) + ")")
            else:
                plt.xlabel("Communications")
            
            ax.tick_params(axis='both', which='major')
            ax.tick_params(axis='both', which='minor')

            ax.set_xticks([i * max(idx)//5 for i in range(6)], [0, 10, 20, 30, 40, 50])

            if results["params"]["n_local"] == 1000 and results["params"]["nenvs"] == 100:
                plt.legend(loc="upper right")

            ax.set_yscale("log")

            ax.set_xlim(xlims)

            ax.set_ylim(5e-5, 13)
            if "step,0.01" in name:
                plt.ylim(5e-6, 13)
            
            
            plt.savefig("plots/" + N + "/" + name + "_" + plot_name + ".pdf", bbox_inches="tight")
