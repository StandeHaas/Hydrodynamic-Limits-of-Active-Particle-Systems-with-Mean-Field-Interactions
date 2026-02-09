### IMEX PDE solver
## imports
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.fft import rfft, irfft

## intial conditions and variables
# grid 
L = 1000        # number of gridpoints on Torus [0,1]
xlim = 1
dx = xlim / L
x = np.linspace(0,1,L,endpoint=False)
#time
dt = 5e-4       # time step
T = 10
nsteps = int(T/dt)

# variables
gaussian_kernel = False
local = True
periodic = False

beta = 1.5
gamma = 0.000233
lam = 0.006
J = 1
def cw_rate(sigma, m, beta, J):
    rate = np.exp(-J * beta * sigma * m) / (2* np.cosh(beta * J * m))
    return rate

# initial distribution of rho (for the poisson to stay a poisson we must not start with 'crazy' variance)
rho_p = 1 + 0.35 * np.random.randn(L) #generates array of length L with N(0,1)
rho_m = 1 + 0.35 * np.random.randn(L)
rho_p = np.clip(rho_p, 0 , None)        #makes sure the rho's are positive
rho_m = np.clip(rho_m, 0 , None)
tot = np.sum(rho_p + rho_m)
rho_p /= tot
rho_m /= tot

#storage
m_series = np.zeros(nsteps+1)
tot_series = np.zeros(nsteps+1)
rho_hat_complex = np.zeros((nsteps+1, int(L/2 + 1)), dtype=complex)
fft_amp_list = np.zeros((nsteps+1, int(L/2 + 1)), dtype=float)
var_list = np.zeros(nsteps+1, dtype=float)
#snapshot storage for "videos"
snapshot_interval = 50 
n_snaps = nsteps // snapshot_interval + 1
total_snapshots = np.zeros((n_snaps, L)) 
m_snapshots = np.zeros((n_snaps, L)) 
times = np.zeros(n_snaps)

## functions for IMEX
#implicit functions
def advective_derivative(rho, sigma, lam): #makes sure the minus sign is correct since there is a sigma term in the PDE
    # this should avoid non-physical oscillations, (since this is the physical correct direction of the transport)
    # so if forward we also do +1 - 0 (i think like in the generator code this prevents movement throughout the entire system in one timestep)
    if periodic: 
        if - sigma * lam > 0 :
            return -(rho - np.roll(rho, 1)) / dx # backward difference
        else: 
            return -(np.roll(rho, -1) - rho) / dx # forward difference
    else: 
        deriv = np.zeros_like(rho)
        if - sigma * lam < 0 :
            deriv[1:] = (rho[1:] - rho[:-1]) / dx
            # left boundary i=0: use zero slope (Neumann) -> derivative 0
            deriv[0] = 0.0  
        return deriv 

#explicit solve for diffusion part
D = diags([1,-2,1], offsets=[-1,0,1], shape=(L,L)).tolil() #.tolil speeds up computations since D containts very many 0's
D[-1,-1] = -2
if periodic == True: 
    D[0,-1] = D[-1, 0] = 1 # periodic BCs
else: 
    D[0,1] = D[-1, -2] = 2 # u_x = 0
D = D.tocsr() #makes it even faster

I = diags(np.ones(L), 0)
A_diff = (I - gamma * dt * D / (dx**2)).tocsr() #D wasn't yet divided by dx**2

def implicit_diffusion_step(rho):
    return spsolve(A_diff, rho)

#explicit solve for active and tumble
if local == False:
    def magnetization(rho_p, rho_m):
        num = np.sum((rho_p - rho_m)) * dx
        den = np.sum((rho_p + rho_m)) * dx
        return num / den 
else:
    if gaussian_kernel == True:
        sigma = 0.02

        def gaussian_kernel_fft(L, dx, sigma):
            i = np.arange(L)
            dist = np.minimum(i, L - i) * dx
            kernel = np.exp(-0.5 * (dist / sigma)**2)
            kernel /= kernel.sum()
            return kernel

        kernel = gaussian_kernel_fft(L, dx, sigma)
        kernel_hat = rfft(kernel)

        def magnetization(rho_p, rho_m, eps=1e-6):
            # local (smoothed) magnetization using periodic convolution
            num_conv = irfft(rfft(rho_p - rho_m) * kernel_hat, n=L)
            den_conv = irfft(rfft(rho_p + rho_m) * kernel_hat, n=L)
            return num_conv / (den_conv + eps)
    else:
        def magnetization(rho_p, rho_m, eps=1e-6):
            num = (rho_p - rho_m) * dx
            den = (rho_p + rho_m + eps) * dx
            return num / den 


def reaction_term(rho_p, rho_m, beta):
    m = magnetization(rho_p, rho_m)
    R_p = cw_rate(-1, m, beta, J) * rho_m - cw_rate(1, m, beta, J) * rho_p
    R_m = cw_rate(1, m, beta, J) * rho_p  - cw_rate(-1, m, beta, J) * rho_m 
    return R_p, R_m

## simulation
def pde_step(rho_p, rho_m, dt, gamma, lam, beta):
    #diffusion
    rho_p_star = implicit_diffusion_step(rho_p)
    rho_m_star = implicit_diffusion_step(rho_m)
    #active
    adv_p = -1 * lam * advective_derivative(rho_p_star, +1, lam)
    adv_m = +1 * lam * advective_derivative(rho_m_star, -1, lam)
    #tumble
    R_p, R_m = reaction_term(rho_p_star, rho_m_star, beta)
    #new
    rho_p_new = rho_p_star + dt * (adv_p + R_p)
    rho_m_new = rho_m_star + dt * (adv_m + R_m)
    rho_p_new = np.clip(rho_p_new, 0, None) #makes sure that rho is positive
    rho_m_new = np.clip(rho_m_new, 0, None)

    # we must renormalize rho_p + rho_m
    M0 = (rho_p + rho_m).sum()
    M_now = rho_p_new.sum() + rho_m_new.sum()
    
    factor = M0 / M_now
    rho_p_new *= factor
    rho_m_new *= factor


    return rho_p_new, rho_m_new

#store initial
if local == False:
    m_series[0] = magnetization(rho_p, rho_m)
else: 
    m_series[0] = np.mean(magnetization(rho_p, rho_m))
tot_series[0] = np.sum(rho_p + rho_m)
fft_result = np.fft.rfft(rho_p + rho_m) / L
fft_amp_list[0] = np.abs(fft_result) #since we work we real values, the DFT is Hermitian so we can neglet half of the spectrum
rho_hat_complex[0] = fft_result   
var_list[0] = np.var(rho_p + rho_m)
total_snapshots[0] = rho_p + rho_m
m_snapshots[0] = rho_p - rho_m
times[0] = 0.0
snap_idx = 1

#run simulation
for n in range(nsteps):
    #simulation
    rho_p, rho_m = pde_step(rho_p, rho_m, dt, gamma, lam, beta)
    #store new variables
    if local == False:
        m_series[n+1] = magnetization(rho_p, rho_m)
    else: 
        m_series[n+1] = np.mean(magnetization(rho_p, rho_m))
    tot_series[n+1] = np.sum(rho_p + rho_m)
    fft_result = np.fft.rfft(rho_p + rho_m) / L
    fft_amp_list[n+1] = np.abs(fft_result) #since we work we real values, the DFT is Hermitian so we can neglet half of the spectrum
    rho_hat_complex[n+1] = fft_result
    var_list[n+1] = np.var(rho_p + rho_m)
    if (n + 1) % snapshot_interval == 0:
        total_snapshots[snap_idx] = rho_p + rho_m
        m_snapshots[snap_idx] = rho_p - rho_m
        times[snap_idx] = (n + 1) * dt
        snap_idx += 1


## visualize 
# plot final rho
plt.plot(x, rho_p, label='rho_+ (final)')
plt.plot(x, rho_m, label='rho_- (final)')
plt.legend()
plt.xlabel('x')
plt.xlim(0,xlim)
plt.grid()
plt.show()
# plot m/t
t = np.linspace(0, T, nsteps+1)
plt.plot(t, m_series)
plt.xlabel('t')
plt.ylabel('m(t)')
plt.xlim(0,T)
plt.grid()
plt.show()
# plot tot/t
plt.plot(t, tot_series)
plt.xlabel('t')
plt.ylabel('sum(rho_p + rho_m)(t)')
plt.xlim(0,T)
plt.grid()
plt.show()
# plot A_K/t
colors = ['paleturquoise', 'aquamarine', 'turquoise', 'steelblue', 'blue', 'navy']
for k in range(1, 7):
    plt.plot(t, fft_amp_list[:,k], label=k, color = colors[6 - k], alpha = 0.6)
plt.xlabel('t')
plt.ylabel('|A_k(t)|')
plt.xlim(0,T)
plt.legend()
plt.grid()
plt.show()
# unwrapped angle_K/t plot
for k in range(1, 7):
    plt.plot(t, np.unwrap(np.angle(rho_hat_complex[:,k])), label=k, color = colors[6 - k], alpha = 0.6)
plt.xlabel('t')
plt.ylabel('unwrapped Arg(A_k(t))')
plt.xlim(0,T)
plt.legend()
plt.grid()
plt.show()
# angle_K/t plot
for k in range(1, 7):
    plt.plot(t, np.angle(rho_hat_complex[:,k]), label=k, color = colors[6 - k], alpha = 0.6)
plt.xlabel('t')
plt.ylabel('unwrapped Arg(A_k(t))')
plt.xlim(0,T)
plt.legend()
plt.grid()
plt.show()

# plot Var/t
plt.plot(t, var_list)
plt.xlabel('t')
plt.ylabel('Var(t)')
plt.xlim(0,T)
plt.grid()
plt.show()
# space time plot total / t
plt.imshow(total_snapshots,aspect='auto',origin='upper',extent=[0, 1, times[-1], 0], cmap='viridis')
plt.colorbar(label=r'$\rho_+ + \rho_-$')
plt.xlabel('x')
plt.xlim(0,xlim)
plt.ylabel('time')
plt.ylim(0,T)
plt.tight_layout()
plt.show()
# space time plot total / t
plt.imshow(m_snapshots,aspect='auto',origin='upper',extent=[0, 1, times[-1], 0], cmap='viridis')
plt.colorbar(label=r'$\rho_+ - \rho_-$')
plt.xlabel('x')
plt.xlim(0,xlim)
plt.ylabel('time')
plt.ylim(0,T)
plt.tight_layout()
plt.show()