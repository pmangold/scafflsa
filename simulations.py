import numpy as np
import random
import os
import argparse
import sys
from utils import *
from tqdm import tqdm
from GarnetEnv import Garnet
from fed_td_algorithms import fed_td, lyap, fed_td_control
import pickle
import glob
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

import functools

steps = [0.01, 0.1]
Hs = [1, 10, 100, 1000, 10000]
Ns = [10, 100]
hg_kerns = [0.02, 10]
TH = 5e5

seed_value = 0
def seed():
    global seed_value
    seed_value += 1
    return seed_value

def compute_bias(environnements, step, n_local, n_rounds):

    if n_rounds > 100:
        return compute_bias2(environnements, step, n_local)
    
    nenvs = environnements.nenvs

    rho_c = np.array([
        np.dot(np.eye(environnements.p) - np.linalg.matrix_power(np.eye(environnements.p) - step * environnements.Abarc[c], n_local), environnements.theta_c[c] - environnements.thetalim)
        for c in range(nenvs)])
    avg_rho = np.mean(rho_c, axis=0)

    gamma_c = np.array([
        np.linalg.matrix_power(np.eye(environnements.p) - step * environnements.Abarc[c], n_local)
        for c in range(nenvs)])
    avg_gamma = np.mean(gamma_c, axis=0)

    ret = 0
    for s in range(1, n_rounds+1):
        ret += np.linalg.matrix_power(avg_gamma, n_rounds - s) @ avg_rho

    return ret
    
def compute_bias2(environnements, step, n_local):

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

def compute_noise(environnements, step, n_est=100):

    nenvs = environnements.nenvs
    sumA, sumb = 0, 0
    for _ in range(n_est):
        Asample, bsample = environnements.sample_A_and_b()
        diffA = np.array([(environnements.Abarc[c] - Asample[c]) @ environnements.theta_c[c] for c in range(nenvs)])
        diffb = environnements.bbarc - bsample

        sumA += np.linalg.norm(diffA, ord=2, axis=1)**2
        sumb += np.linalg.norm(diffb, ord=2, axis=1)**2

        varA = np.mean(sumA)/n_est
        varb = np.mean(sumb)/n_est

    return step * (varA + varb) / nenvs

def run_fed_td_experiment(sample_rng, envs_param, theta0, n_local, T, step, bias):
    envs = Garnet(**envs_param, sample_seed=seed())
    envs.set_sample_rng(sample_rng)

    ret =  ("fed_td", fed_td(theta0, envs, n_local, T, step, bias, verbose=False))

    return ret
    
def run_scafftd_experiment(sample_rng, envs_param, theta0, n_local, T, step):
    envs = Garnet(**envs_param, sample_seed=seed())
    envs.set_sample_rng(sample_rng)

    return ("scafftd", fed_td_control(theta0, envs, n_local, T, step, verbose=False))

def smap(f):
    return f[0](f[1])

def run_experiment(
        nenvs, TH,
        hg_kern, steps, Hs,
        num_rep=10,
        ns=30, na=2, p=8, b=2):

    os.makedirs("results/" + str(TH), exist_ok=True)

    sample_rngs = [np.random.default_rng(seed=seed()) for _ in range(num_rep * 2)]
    
    for step in steps:
        for n_local in Hs:
            envs_param = {"ns": ns, "na": na, "b": b, "p": p, "gamma": 0.95, "nenvs": nenvs,
                          "heteregoneity_kern": hg_kern, "heteregoneity_reward": hg_kern,
                          "gen_seed": 42 }
            envs = Garnet(ns=ns, na=na, b=b, p = p, gamma=0.95, nenvs=nenvs,
                          heteregoneity_kern=hg_kern, heteregoneity_reward=hg_kern, gen_seed=42)
            envs.set_sample_rng(np.random.default_rng(seed=seed()))
            
            
            # setup
            T = max(50, int( TH / n_local ))
            theta0 = envs.thetalim + 1 #np.zeros(p)
            bias = compute_bias(envs, step, n_local, T)

            print("---- N =", nenvs, ", T=", T, ", H =", n_local, ", step =", step, ", hg =", hg_kern, "----")
            print("bias:", np.linalg.norm(bias))
            
            # run experiment
            func_td = functools.partial(run_fed_td_experiment,
                                        envs_param=envs_param, theta0=theta0,
                                        n_local=n_local, T=T, step=step, bias=bias)
            func_scafftd = functools.partial(run_scafftd_experiment,
                                             envs_param=envs_param, theta0=theta0,
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
            with open('results/' + str(TH) + "/" + name +'.pkl', 'wb') as f:
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
                        "thetalimc": envs.theta_c,
                        "noise_level": compute_noise(envs, step)
                    }, f)
                    
if __name__ == '__main__':
    ns = 30
    na = 2
    p = 8
    b = 2
    for N in Ns:
        for hg_kern in hg_kerns:
            print("---- N =", N, "----")
            run_experiment(N, TH, hg_kern=hg_kern, steps=steps, Hs=Hs)
            print()

