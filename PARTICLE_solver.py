### particle solver
## imports
import numpy as np
import matplotlib.pyplot as plt

## intial conditions and variables
# grid 
L = 200        # number of gridpoints on Torus [0,1]
xlim = 1
dx = xlim / L
#time
T = 10.0 
obs_dt = 0.01 # interval with which we store
times_obs = np.arange(0, T, obs_dt)
M = times_obs.size

# variables
N = 1000        # number of particles
beta = 1.33
gamma = 0.0233
lam = 0.7
r_gamma = gamma / (dx ** 2)
r_lam = lam / dx
def cw_rate(sigma, m, beta=2.0):
    return np.exp(-beta * sigma * m)

## fucntions
## functions
def emperical_densities(pos, sigma):
    counts_p = np.bincount(pos[sigma==1], minlength=L)
    counts_m = np.bincount(pos[sigma==-1], minlength=L)
    rho_p = counts_p / (N * dx)
    rho_m = counts_m / (N * dx)
    return rho_p, rho_m

def local_magnetization(rho_p, rho_m):
    tot = rho_p + rho_m
    m_local = np.zeros_like(tot)
    nonzero = tot > 0
    m_local[nonzero] = (rho_p[nonzero] - rho_m[nonzero]) / tot[nonzero]
    return m_local

def fourier_ampl(u):
    var = np.var(u)
    u_hat = np.fft.fft(u)
    amp = np.abs(u_hat)
    return var, u_hat, amp

## Simulation
start_pos = np.random.randint(0, L, size=N)
start_sigma = np.where(np.random.rand(N)< 0.5, 1, -1)
def run_particle_simulation():
    # initial distribution 
    pos = start_pos
    sigma = start_sigma
    t = 0.0
    obs_idx = 0

    #storage (and store initial distr)
    # store with obs_dt
    rho_p_list = np.zeros((M, L), dtype=float)
    rho_m_list = np.zeros((M, L), dtype=float)
    total_list = np.zeros((M, L), dtype=float)
    m_local_list = np.zeros((M, L), dtype=float)
    rho_hat_complex = np.zeros((M,L), dtype=complex)
    fft_amp_list = np.zeros((M, L), dtype=float)
    m_global = np.zeros(M, dtype=float)
    var_list = np.zeros(M, dtype=float)

    #storage (store initial distribution)
    rho_p, rho_m = emperical_densities(pos, sigma)
    rho_p_list[0] = rho_p
    rho_m_list[0] = rho_m
    total_list[0] = rho_p + rho_m
    m_local_list[0] = local_magnetization(rho_p, rho_m)
    m_global[0] = np.mean(sigma)
    var, u_hat, amp  = fourier_ampl(rho_p + rho_m)
    rho_hat_complex[0] = u_hat
    fft_amp_list[0] = amp
    var_list[0] = var
    obs_idx += 1
    ## simulation
    while t < T:
        m = np.mean(sigma)
        cvec = cw_rate(sigma, m, beta)
        rates = (r_gamma + r_lam) + cvec
        R = np.sum(rates)
        tau = np.random.exponential(1.0 / R)
        t += tau
        if t > T:
            break
        #store varialbes if time eligible 
        while obs_idx < M and times_obs[obs_idx] <= t:
            rho_p, rho_m = emperical_densities(pos, sigma)
            rho_p_list[obs_idx] = rho_p
            rho_m_list[obs_idx] = rho_m
            total_list[obs_idx] = rho_p + rho_m
            m_local_list[obs_idx] = local_magnetization(rho_p, rho_m)
            m_global[obs_idx] = np.mean(sigma)
            var, u_hat, amp  = fourier_ampl(rho_p + rho_m)
            rho_hat_complex[obs_idx] = u_hat
            fft_amp_list[obs_idx] = amp
            var_list[obs_idx] = var
            obs_idx += 1

        if obs_idx >= M:
            break

        #we want to find a index i with P(i) rates[i] / R this takes 3 lines:
        u = np.random.rand() * R
        cumsum = np.cumsum(rates)
        i = np.searchsorted(cumsum, u)
        # using the rates of index we choose diff, act or tumble
        r_tumble = cvec[i]
        r_sum = rates[i]
        v = np.random.rand() * r_sum
        if v < r_gamma:
            #diffusion, jump to +/- with prob 1/2
            step = 1 if np.random.rand() < 0.5 else -1
            pos[i] = (pos[i] + step) % L        #the % L makes sure we have periodic BCs
        elif v < r_gamma + r_lam:
            #active jump with sigma[i]
            pos[i] = (pos[i] + sigma[i]) % L
        else: 
            #switch sign
            sigma[i] = -sigma[i]

    return rho_p_list ,rho_m_list, total_list, m_local_list, m_global, rho_hat_complex, fft_amp_list, var_list

n_realizations = 20

rho_p_ens = np.zeros((M, L), dtype=float)
rho_m_ens = np.zeros((M, L), dtype=float)
total_ens = np.zeros((M, L), dtype=float)
m_local_ens = np.zeros((M, L), dtype=float)
m_global_ens = np.zeros(M, dtype=float)
rho_hat_complex_ens = np.zeros((M,L), dtype=complex)
fft_amp_ens = np.zeros((M, L), dtype=float)
var_ens = np.zeros(M, dtype=float)

for i in range(n_realizations):
    print(i)
    rho_p_list, rho_m_list, total_list, m_local_list, m_global, rho_hat_complex, fft_amp_list, var_list = run_particle_simulation()
    rho_p_ens += rho_p_list
    rho_m_ens += rho_m_list
    total_ens += total_list
    m_local_ens += m_local_list
    rho_hat_complex_ens += rho_hat_complex
    m_global_ens += m_global
    fft_amp_ens += fft_amp_list
    var_ens += var_list   

rho_p_ens /= n_realizations
rho_m_ens /= n_realizations
total_ens /= n_realizations
m_local_ens /= n_realizations
m_global_ens /= n_realizations
fft_amp_ens /= n_realizations
rho_hat_complex_ens /= n_realizations
var_ens /= n_realizations

## visualize
# t / m^N(t) plot
plt.plot(times_obs, m_global_ens)
plt.xlabel('t')
plt.ylabel('m^N(t)')
plt.xlim(0,T)
plt.grid()
plt.show()

# t / A_k(t) plot
colors = ['paleturquoise', 'aquamarine', 'turquoise', 'steelblue', 'blue', 'navy']
for k in range(1, 7):
    plt.plot(times_obs, fft_amp_ens[:,k] / L, label=k, color = colors[6 - k], alpha = 0.6)
plt.legend()
plt.xlabel('t')
plt.ylabel('|A_k(t)|')
plt.xlim(0,T)
plt.grid()
plt.show()

# unwrapped angle_K/t plot
for k in range(1, 7):
    plt.plot(times_obs, np.unwrap(np.angle(rho_hat_complex_ens[:,k])), label=k, color = colors[6 - k], alpha = 0.6)
plt.xlabel('t')
plt.ylabel('unwrapped Arg(A_k(t))')
plt.xlim(0,T)
plt.legend()
plt.grid()
plt.show()
# angle_K/t plot
for k in range(1, 7):
    plt.plot(times_obs, np.angle(rho_hat_complex_ens[:,k]), label=k, color = colors[6 - k], alpha = 0.6)
plt.xlabel('t')
plt.ylabel('unwrapped Arg(A_k(t))')
plt.xlim(0,T)
plt.legend()
plt.grid()
plt.show()

# t / Var plot
plt.plot(times_obs, var_ens)
plt.xlabel('t')
plt.ylabel('Var(t)')
plt.xlim(0,T)
plt.grid()
plt.show()

# x / rho plot
j = len(times_obs) // 2
x = np.linspace(0,1,total_list.shape[1], endpoint=False)
plt.plot(x,rho_p_ens[j], label='rho_+', color ='lightblue', alpha = 0.6)
plt.plot(x,rho_m_ens[j], label='rho_-', color ='blue', alpha = 0.6)
plt.plot(x,total_ens[j], label='total', color ='navy', alpha = 0.6)
plt.legend()
plt.xlabel('t')
plt.ylabel('rho')
plt.xlim(0,xlim)
plt.grid()
plt.show()

# space time plot total / t
plt.imshow(m_local_ens,aspect='auto',origin='upper',extent=[0, 1, T, 0], cmap='viridis')
plt.colorbar(label=r'$\rho_+ - \rho_-$')
plt.xlabel('x')
plt.xlim(0,xlim)
plt.ylabel('time')
plt.ylim(0,T)
plt.tight_layout()
plt.show()