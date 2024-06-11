import numpy as np
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

buffer_len = 1000

def lyap(theta, thetalim, xi, xilim, step, n_local):
    N = xi.shape[0]
    return N*np.linalg.norm( theta - thetalim ) ** 2 + 1.0/(step*n_local)**2 * np.linalg.norm( xi - xilim ) ** 2

def fed_td(theta0, environnements, n_local, n_rounds, step, bias, num_workers=1, verbose=True):

    n_agents = environnements.nenvs
    thetalim = environnements.thetalim
    theta = theta0.copy()
    theta_hist = np.zeros((n_rounds + 1, len(theta)))
    l2_norm = np.zeros(n_rounds + 1)
    l2_norm_debiased = np.zeros(n_rounds + 1)
    theta_hist[0] = theta.copy()
    l2_norm[0] = np.linalg.norm(theta - thetalim) ** 2
    l2_norm_debiased[0] = np.linalg.norm(theta - bias - thetalim) ** 2

    k = 0
    for t in tqdm(range(n_rounds)):
        new_theta = np.array([theta.copy() for _ in range(n_agents)])

        for h in range(n_local):
            if k == 0:
                As_buffer, bs_buffer = environnements.sample_A_and_b(buffer_len)
                
            As_h, bs_h = As_buffer[:, k, ...], bs_buffer[:, k, ...]
            new_theta -= step * (np.matmul(As_h, new_theta[:,:,None]).reshape(new_theta.shape) - bs_h)

            k = (k+1) % buffer_len

            
        theta = np.mean(new_theta, axis=0)
        theta_hist[t+1] = theta
        l2_norm[t+1] = np.linalg.norm(theta - thetalim) ** 2
        l2_norm_debiased[t+1] = np.linalg.norm( theta - bias - thetalim )**2 

        if verbose:
            print("Round", t)
            print("theta error:",  l2_norm[t+1], "theta error without bias", l2_norm_debiased[t+1])
            print()

    return theta_hist, l2_norm, l2_norm_debiased


def fed_td_control(theta0, environnements, n_local, n_rounds, step, xi0=None, verbose=False, num_workers=1):
    n_agents = environnements.nenvs
    theta = theta0.copy()
    thetalim = environnements.thetalim
    xilim = np.zeros((n_agents, environnements.p))
    for agent in range(n_agents):
        xilim[agent] = np.dot(environnements.Abarc[agent], environnements.theta_c[agent] - environnements.thetalim)
    theta_hist = np.zeros((n_rounds + 1, len(theta)))
    theta_hist[0] = theta.copy()
    xi = np.zeros((n_agents, environnements.p))
    l2_norm = np.zeros(n_rounds + 1)
    lyapunov = np.zeros(n_rounds + 1)
    l2_norm[0] = np.linalg.norm( theta - thetalim ) ** 2
    lyapunov[0] = lyap(theta, thetalim, xi, xilim, step, n_local)

    if xi0 == "xilim":
        xi = xilim.copy()

    k = 0
    for t in tqdm(range(n_rounds)):
        new_theta = np.array([theta.copy() for _ in range(n_agents)])

        for h in range(n_local):
            if k == 0:
                As_buffer, bs_buffer = environnements.sample_A_and_b(buffer_len)
            
            As_h, bs_h = As_buffer[:, k, ...], bs_buffer[:, k, ...]
            new_theta -= step * (np.matmul(As_h, new_theta[:,:,None]).reshape(new_theta.shape) - bs_h - xi)
            k = (k+1) % buffer_len

        theta = np.mean(new_theta, axis=0)
        theta_hist[t+1] = theta
        xi += 1.0/(n_local*step) * (theta - new_theta)
        l2_norm[t+1] = np.linalg.norm( theta - thetalim ) ** 2
        new_lyap = lyap(theta, thetalim, xi, xilim, step, n_local)
        lyapunov[t+1] = new_lyap.copy()
        if verbose:
            print("Round", t)
            print("theta error:",  l2_norm[t+1], "Laypunov:", lyapunov[t+1] )
            print()
    
    return theta_hist, l2_norm, lyapunov
