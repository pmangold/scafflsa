import numpy as np
import random
import os
import argparse
import sys
from utils import *
from tqdm import tqdm
from GarnetEnv import Garnet
from fed_td_algorithms import fed_td, lyap, fed_td_control
from copy import deepcopy
import pickle
import glob
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

import functools

steps = [0.001, 0.01, 0.1, 1]
Hs = [1, 10, 100]
Ns = [1, 10, 100, 1000]
hg_kerns = [0.02, 10]
HT = 1000

seed_value = 0
def seed():
    global seed_value
    seed_value += 1
    return seed_value
    
def compute_bias(environnements, step, n_local):

    nenvs = environnements.nenvs

    i_minus_gamma = np.array([
        np.eye(environnements.p) - np.linalg.matrix_power(np.eye(environnements.p) - step * environnements.Abarc[c], n_local)
        for c in range(nenvs)])
    big_inv = np.linalg.inv(np.mean(i_minus_gamma, axis=0))
    diff_theta = np.array([
        i_minus_gamma[c] @ (environnements.theta_c[c] - environnements.thetalim)
        for c in range(nenvs)]).mean(axis=0)
    diff_theta_sum = np.array([
        i_minus_gamma[c] @ (environnements.theta_c[c] - environnements.thetalim)
        for c in range(nenvs)]).sum(axis=0)
    bias = big_inv @ diff_theta

    return bias

def run_fed_td_experiment(sample_rng, envs, theta0, n_local, T, step, bias):
    envs = deepcopy(envs)
    envs.set_sample_rng(sample_rng)

    ret =  ("fed_td", fed_td(theta0, envs, n_local, T, step, bias, verbose=False))

    return ret
    
def run_scafftd_experiment(sample_rng, envs, theta0, n_local, T, step):
    envs = deepcopy(envs)
    envs.set_sample_rng(sample_rng)

    return ("scafftd", fed_td_control(theta0, envs, n_local, T, step, verbose=False))

def smap(f):
    return f[0](f[1])

def run_experiment(
        nenvs, T,
        hg_kern, steps, Hs,
        num_rep=10,
        ns=30, na=2, p=8, b=2):

    os.makedirs("results_bias/" + str(T), exist_ok=True)

    sample_rngs = [np.random.default_rng(seed=seed()) for _ in range(num_rep * 2)]
    envs_param = {"ns": ns, "na": na, "b": b, "p": p, "gamma": 0.95, "nenvs": nenvs,
                  "heteregoneity_kern": hg_kern, "heteregoneity_reward": hg_kern,
                  "gen_seed": 42 }
    envs = Garnet(ns=ns, na=na, b=b, p = p, gamma=0.95, nenvs=nenvs,
                  heteregoneity_kern=hg_kern, heteregoneity_reward=hg_kern, gen_seed=42)
    envs.set_sample_rng(np.random.default_rng(seed=seed()))
    
    for step in steps:
        for n_local in Hs:
            # setup
            T = max(3, int(HT / n_local))
            theta0 = envs.thetalim
            bias = compute_bias(envs, step, n_local)

            print("---- N =", nenvs, ", T=", T, ", H =", n_local, ", step =", step, ", hg =", hg_kern, "----")
            print("bias:", np.linalg.norm(bias))
            
            # run experiment
            func_td = functools.partial(run_fed_td_experiment,
                                        envs=envs, theta0=theta0+bias,
                                        n_local=n_local, T=T, step=step, bias=bias)
            func_scafftd = functools.partial(run_scafftd_experiment,
                                             envs=envs, theta0=theta0,
                                             n_local=n_local, T=T, step=step)

            

            with ProcessPoolExecutor(max_workers=10) as executor:
                res = list(executor.map(smap,
                                        sum([[(func_td, sample_rngs[2*i]),
                                              (func_scafftd, sample_rngs[2*i+1])
                                              ]
                                             for i in range(num_rep)], [])))


            results_fed_td = [elt[1] for elt in res if elt[0] == "fed_td" ]
            results_scafftd = [elt[1] for elt in res if elt[0] == "scafftd" ]

            l2_norm_scafftd, lyapunov, l2_norm_fedtd, l2_norm_debiased = [], [], [], []
            for i in range(num_rep):
                thetas_scafftd , l2_norm_scafftd_i, lyapunov_i = results_scafftd[i]
                thetas_fedtd , l2_norm_fedtd_i, l2_norm_debiased_i = results_fed_td[i]

                l2_norm_fedtd.append(l2_norm_fedtd_i)
                l2_norm_debiased.append(l2_norm_debiased_i)

                l2_norm_scafftd.append(l2_norm_scafftd_i)
                lyapunov.append(lyapunov_i)
            name = ('step,' + str(step) + ',nlocal,' + str(n_local) + ',nrounds,' + str(T) + ',nenvs,' + str(nenvs) +
                        ',hg_kernel,' + str(hg_kern)) 
            with open('results_bias/' + str(HT) + "/" + name +'.pkl', 'wb') as f:
                pickle.dump(
                    {
                        "params": {"step": step,
                                   "n_local": n_local,
                                   "T": T,
                                   "nenvs": nenvs,
                                   "hg_kern": hg_kern,
                                   "hg_reward": hg_kern
                                   },
                        "l2_norm_fedtd": l2_norm_fedtd,
                        "l2_norm_scafftd": l2_norm_scafftd,
                        "lyapunov": lyapunov,
                        "l2_norm_fedtd_debiased": l2_norm_debiased,
                        "fedtd_bias": bias,
                        "thetalim": envs.thetalim,
                        "thetalimc": envs.theta_c
                    }, f)
                    
if __name__ == '__main__':
    ns = 30
    na = 2
    p = 8
    b = 2
    for N in Ns:
        for hg_kern in hg_kerns:
            print("---- N =", N, "----")
            run_experiment(N, HT, hg_kern=hg_kern, steps=steps, Hs=Hs)
            print()

