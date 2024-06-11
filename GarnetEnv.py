import numpy as np
import random
class Garnet():
    def __init__(self, ns=10, na=10, b=3, p = 4, gamma = 0.95,
                 nenvs = 5, heteregoneity_kern = 0.01, heteregoneity_reward = 0.001,
                 gen_seed = 10, sample_seed = 42):
        self.gen_rng = np.random.default_rng(gen_seed)
        self.sample_rng = None
        self.ns = ns
        self.na = na
        self.b = b
        self.p = p
        self.gamma = gamma
        self.nenvs = nenvs
        self.heteregoneity_kern = heteregoneity_kern
        self.heteregoneity_reward = heteregoneity_reward
        
        # Generate nevs times the same probability transition matrix
        tr_kernel = np.zeros(( ns, na, ns))
        tr_kernel2 = np.zeros(( ns, na, ns))
        
        if self.heteregoneity_kern < 1:
            # Make sure that every state is connected to b other states exactly
            for s in range(ns):
                for a in range(na):
                    tr_kernel[s][a][:b] = 1
                    self.gen_rng.shuffle(tr_kernel[s][a])

            # Duplicate the transition kernel nevs times
            tr_kernels = np.broadcast_to(tr_kernel, ( nenvs, ns, na, ns))
            # Perturbate the environnement with an epsilon heterogeneity factor and normalise
            perturbation = self.gen_rng.uniform(low = 0.0, high = heteregoneity_kern, size=(nenvs, ns, na, ns)) * (tr_kernels != 0.0)
            tr_kernels = np.abs(tr_kernels + perturbation)
            a = tr_kernels.sum(axis = 3, keepdims = 1)
            self.tr_kernels = tr_kernels/ a

        else:
            # Generate nevs times the same probability transition matrix
            tr_kernels = np.zeros(( nenvs, ns, na, ns))

            # Make sure that every state is connected to b other states exactly
            for s in range(ns):
                for a in range(na):
                    tr_kernel[s][a][:b] = 1
                    self.gen_rng.shuffle(tr_kernel[s][a])
                    tr_kernel2[s][a][:b] = 1
                    self.gen_rng.shuffle(tr_kernel2[s][a])

            # Make sure that every state is connected to b other states exactly
            for e in range(nenvs):
                for s in range(ns):
                    for a in range(na):
                        tr_kernels[e] = tr_kernel.copy() if e%2==0 else tr_kernel2.copy()

                        #tr_kernels[e,s,a,:b] = 1
                        #rng.shuffle(tr_kernels[e,s,a])
            perturbation = self.gen_rng.uniform(low = 0.0, high =0.02, size=(nenvs, ns, na, ns)) * (tr_kernels != 0.0)
            tr_kernels = np.abs(tr_kernels + perturbation)
            a = tr_kernels.sum(axis = 3, keepdims = 1)

            self.tr_kernels = tr_kernels/ a

        # We handle now the reward function. We create a reward and then perturbate the chain of rewards

        reward =  self.gen_rng.uniform(low = 0.0, high = 1.0, size=ns)
        rewards = np.broadcast_to(reward, ( nenvs, ns))
        perturbation = self.gen_rng.uniform(low = 0.0, high = heteregoneity_reward, size= ( nenvs, ns))
        self.rewards = (rewards + perturbation) #/ ( 1 + heteregoneity_reward )

        # if heterogeneous, make rewards very heterogeneous
        if self.heteregoneity_kern > 1:
            self.rewards = rewards * np.array([1 if c%2==0 else -1 for c in range(nenvs)])[:,None] #/ ( 1 + heteregoneity_reward )
        
        # We create an embedding now of the state space
        feat_map = self.gen_rng.uniform(low = 0.0, high = 1.0, size=(ns, p))
        part_sum = feat_map.sum(axis = 1,  keepdims = 1 )
        self.feat_map = feat_map/ part_sum
        # define a policy
        self.policy = np.zeros(( ns, na))
        for s in range(ns):
                self.policy[s][:na-1] = 1/(na-1)
                self.gen_rng.shuffle( self.policy[s])
        # Compute the corresponding Markov rewards process
        reward_tr_kernels = np.zeros(( nenvs, ns, ns))
        for env in range(nenvs):
            for s in range(ns):
                for sprime in range (ns):
                    reward_tr_kernels[env][s][sprime] = np.dot(self.tr_kernels[env, s, :, sprime], self.policy[s, :])
        self.reward_tr_kernels = reward_tr_kernels

        # Find the stationnary distribution of every Markov reward process
        self.stat_dist = np.zeros(( nenvs, ns))
        for env in range(nenvs):
            evals, evecs = np.linalg.eig(self.reward_tr_kernels[env].T)
            evec1 = evecs[:,np.isclose(evals, 1)]
            evec1 = evec1[:,0]
            stationary = evec1 / evec1.sum()
            self.stat_dist[env] = stationary.real
            self.stat_dist[env] =  abs(self.stat_dist[env]/self.stat_dist[env].sum())
        # Compute thetalim
        Abarc = np.zeros((self.nenvs, self.p , self.p))
        Abar = np.zeros(( self.p , self.p))
        bbarc = np.zeros((self.nenvs, self.p))
        bbar = np.zeros(self.p)
        theta_c = np.zeros((self.nenvs, self.p))
        for env in range(self.nenvs):
            for s in range(ns):
                for sprime in range(ns):
                    r = self.rewards[env][s]
                    phi_s = self.feat_map[s]
                    phi_sprime = self.feat_map[sprime].reshape((self.p, 1))
                    p_s_sprime = self.stat_dist[env][s] * self.reward_tr_kernels[env][s][sprime]
                    Abarc[env] += p_s_sprime* np.dot(phi_s.reshape((self.p, 1)), (phi_s.reshape((self.p, 1)) - self.gamma*phi_sprime).T)
                    bbarc[env] +=  p_s_sprime * r*phi_s
            Abar += Abarc[env]
            bbar += bbarc[env]

            theta_c[env] = np.linalg.solve(Abarc[env], bbarc[env])
            thetalim = np.linalg.solve(Abar, bbar)

        self.theta_c = theta_c
        self.thetalim = thetalim
        self.Abarc = Abarc
        self.Abar = Abar
        self.bbarc = bbarc
        self.bbar = bbar
        
        # Compute Kappa, expectation_trace_sigma_epsilon and small a
        SigAbarc = np.zeros((self.nenvs, self.p , self.p))
        Sigepsilon_c = np.zeros((self.nenvs, self.p , self.p))
        
        for env in range(self.nenvs):
            for s in range(ns):
                for sprime in range(ns):
                    r = self.rewards[env][s]
                    phi_s = self.feat_map[s]
                    phi_sprime = self.feat_map[sprime].reshape((self.p, 1))
                    p_s_sprime = self.stat_dist[env][s] * self.reward_tr_kernels[env][s][sprime]
                    Ac_Z = p_s_sprime* np.dot(phi_s.reshape((self.p, 1)), (phi_s.reshape((self.p, 1)) - self.gamma*phi_sprime).T)
                    bc_Z =  p_s_sprime * r*phi_s
                    SigAbarc[env] += np.dot(Ac_Z - self.Abarc[env] ,(Ac_Z - self.Abarc[env]).T)
                    epsilon = np.dot(Ac_Z - self.Abarc[env], self.thetalim ) - (bc_Z - self.bbarc[env])
                    Sigepsilon_c[env] += np.dot(epsilon, epsilon.T)
        lambdaminlist = np.zeros(self.nenvs)
        for env in range(self.nenvs):
            sigma_phi_c =  np.zeros(( self.p , self.p))
            for s in range(ns):
                phi_s = self.feat_map[s]
                sigma_phi_c += self.stat_dist[env][s] * np.dot(phi_s.reshape((self.p, 1)), phi_s.reshape((self.p, 1)).T)
            evals, evecs = np.linalg.eig(sigma_phi_c)
            lambdamin = np.min(evals)
            lambdaminlist[env] = lambdamin
        lmin = np.min(lambdaminlist)
        self.a = (1 - self.gamma) * lmin /2
        self.kappa = 0.0
        self.tracesigeps = 0.0
        self.distthetas = 0.0
        for env in range(self.nenvs):
            self.kappa += np.linalg.norm(SigAbarc[env]) * np.linalg.norm(self.thetalim - self.theta_c[env])**2
            self.tracesigeps += np.trace(Sigepsilon_c[env])
            self.distthetas += np.linalg.norm(self.thetalim - self.theta_c[env])
        self.kappa = self.kappa / self.nenvs
        self.tracesigeps = self.tracesigeps / self.nenvs
        self.distthetas = self.distthetas /  self.nenvs
        self.etainfty = (1 - self.gamma)/4
        self.L = 4/(1 -self.gamma)
        self.Ca = 2*(1 + self.gamma)


    def set_sample_rng(self, sample_rng):
        self.sample_rng = sample_rng
        
    def rand_choice_vec2(self, probs):
        #Probs is a tensor with shape (n_envs, n_samples, ns)
        return (probs.cumsum(-1) > self.sample_rng.random((probs.shape[0], probs.shape[1]))[..., None]).argmax(-1)


    def rand_choice_vec1(self, probs, n_samples):
        #Probs is a matrix with shape (n_envs, ns)
        return (probs.cumsum(-1)[None, ...] > self.sample_rng.random((n_samples, probs.shape[0]))[..., None]).argmax(-1).T

    def sample_state_reward_state(self, n_samples):
        state_reward_state = np.zeros((n_samples, self.nenvs, 3))

        state = self.rand_choice_vec1(self.stat_dist, n_samples)
        assert state.shape == (self.nenvs, n_samples)
        reward = np.take_along_axis(self.rewards, state, axis=1)

        assert reward[0, 0] == self.rewards[0, state[0, 0]]
        # assert reward[1, 0] == self.rewards[1, state[1, 0]]
        assert reward.shape == (self.nenvs, n_samples)

        probs = np.take_along_axis(self.reward_tr_kernels, np.repeat(state[..., None], self.ns, axis=-1), axis=1) #shape: (nenvs, n_samples, ns)

        assert np.allclose(probs[0, 0], self.reward_tr_kernels[0, state[0, 0]])
        # assert np.allclose(probs[1, 0], self.reward_tr_kernels[1, state[1, 0]])

        state_prime = self.rand_choice_vec2(probs)

        return np.stack([state, reward, state_prime], axis=-1) #state_reward_state

    def sample_A_and_b(self, n_samples=1):

        As = np.zeros((self.nenvs, n_samples, self.p , self.p))
        bs = np.zeros((self.nenvs, n_samples, self.p))
        state_reward_state = self.sample_state_reward_state(n_samples) # (n_envs, n_samples, 3)

        s = state_reward_state[..., 0].astype(int)
        r = state_reward_state[..., 1]
        sprime = state_reward_state[..., 2].astype(int)

        phi_s = self.feat_map[s][..., None]
        phi_sprime = self.feat_map[sprime][..., None]
        As = np.matmul(phi_s, (phi_s - self.gamma * phi_sprime).transpose((0, 1, 3, 2)))
        bs = r[..., None] * phi_s.squeeze(-1)

        return (As, bs)
    
    def sample_Markov_chain(self,ep_lenght):
        pass
