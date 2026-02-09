#Gillepsie simulation for biological system
#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Optional, Tuple, Dict, Any
from vispy import app, scene, io
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.optimize import fixed_point, curve_fit
from scipy.stats import nbinom, poisson

from PARTICLE_solver_CLASS import ParticleSystem

#### create initial conditions
def make_exp_gradient(
    L, 
    N, 
    frac_plus,
    decay_length,
    anchor_positions=(0.25, 0.60),
    anchor_peak_width=0.01,
    anchor_peak_mass=0.03,
):
    xs = np.arange(L) / float(L)

    # exp. free +
    plus_unscaled = np.exp(-xs / decay_length)
    # small base +
    minus_unscaled = 0.05 * np.ones_like(xs)
    # gaussian peaks at acnhors -
    if anchor_positions is not None:
        for a in anchor_positions:
            minus_unscaled += anchor_peak_mass * np.exp(-0.5 * ((xs - a) / anchor_peak_width) ** 2)

    #split up using frac_plus
    plus_shape = plus_unscaled / plus_unscaled.sum()
    minus_shape = minus_unscaled / minus_unscaled.sum()

    rho_plus = N * frac_plus * plus_shape
    rho_minus = N * (1 - frac_plus) * minus_shape

    #produce callable fucntions PartSys expects
    def rho0_plus(x):
        # map x to nearest index
        idx = int(np.clip(np.round(x * L), 0, L - 1))
        return float(rho_plus[idx])
          
    def rho0_minus(x):
        idx = int(np.clip(np.round(x * L), 0, L - 1))
        return float(rho_minus[idx])
    
    return [rho0_plus, rho0_minus, rho_plus, rho_minus]

######## Sweep over 1 beta n times ##################################
def sweep_beta_ensemble(
    beta, 
    n_runs=10,
    ps_kwargs=None, 
    init_kwargs=None,
    run_kwargs=None, 
    rng_seeds=None
):
    if ps_kwargs is None:
        ps_kwargs = {}
    if run_kwargs is None:
        run_kwargs = {}

    v_list = []
    D_list = []
    m_list = []
    rho_eff_list = []
    out_list = []
    block_list = []
    for run_i in range(n_runs):
        print("     run =", run_i + 1)
        if rng_seeds is None:
            rng = None   
        if rng_seeds is None:
            rng = None
        else:
            rng = np.random.default_rng(int(rng_seeds[run_i]))
        ps = ParticleSystem(beta=beta, rng=rng, **ps_kwargs, **init_kwargs)
        out = ps.run(**run_kwargs)
        mean_v, v_eff_ts, times, si, ei, frac_boundary = compute_v_eff_and_window(out, ps, boundary_xmin=0.99,max_buondary_fraction=0.06,min_window_fraction=0.10)
        v_list.append(mean_v)
        D_eff = compute_D_eff_active(out, ps, start_idx=si, end_idx=ei)
        D_list.append(D_eff)
        m_mean = compute_mean_magnetizatoin(out, si, ei)
        m_list.append(m_mean)
        rho_eff = compute_rho_eff(out, si , ei)
        rho_eff_list.append(rho_eff)
        out_list.append(out)
        block = compute_blocking_probability(out, si, ei)
        block_list.append(block)

    v_array = np.array(v_list, dtype=float)
    D_array = np.array(D_list)
    m_array = np.array(m_list)
    rho_array = np.array(rho_eff_list)
    block_array = np.array(block_list)

    mean = float(v_array.mean())
    std  = float(v_array.std(ddof=1)) if v_array.size > 1 else 0.0
    se   = std / np.sqrt(max(1, v_array.size))
    D_mean = D_array.mean()
    D_std  = D_array.std(ddof=1)
    D_se   = D_std / np.sqrt(len(m_array))
    m_mean = m_array.mean()
    m_std  = m_array.std(ddof=1)
    m_se   = m_std / np.sqrt(len(m_array))
    rho_mean = rho_array.mean()
    rho_se = rho_array.std(ddof=1) / np.sqrt(len(m_array))
    block_mean = block_array.mean()
    block_se = block_array.std(ddof=1) / np.sqrt(len(m_array))

    return mean, std, se, v_array, out_list, m_mean, m_std, m_se, rho_mean, rho_se, block_mean, block_se, D_mean, D_se

###### Compute theoretical curves ###########################

##### V_EFF #################################
##### Compute v_eff from results (RIGHT by rule)
def compute_v_eff_and_window(out, ps, 
                             boundary_xmin=0.99,
                             max_buondary_fraction=0.06,
                             min_window_fraction=0.10):
    times = out['times_obs']
    total_density = out['total_list'] 
    M, L = total_density.shape

    x_grid = np.linspace(0, 1.0, L)
    dx = x_grid[1] - x_grid[0]

    boundary_mask = (x_grid >= boundary_xmin)
    boundary_count = (total_density[:, boundary_mask].sum(axis=1) * dx)

    N_t = (total_density.sum(axis=1) * dx)
    N0 = N_t[0]
    frac_boundary = boundary_count / (N_t + 1e-12)
    # safe where less then 10% at the boundary 
    safe = np.where(frac_boundary >= max_buondary_fraction)[0]
    if safe.size == 0:
        start_idx = int(0.65 * M)
        end_idx = M
    else: 
        start_idx = int(0.65 * M)
        unsafe_rel = np.where(~safe[start_idx:])[0]
        if len(unsafe_rel) == 0:
            end_idx = M
        else:
            end_idx = start_idx + unsafe_rel[0]
        min_len = max(3, int(min_window_fraction * M))
        if end_idx - start_idx < min_len:
            end_idx = min(M, start_idx + min_len)

    x_grid = np.linspace(0, 1.0, ps.L)
    mean_x = (total_density * x_grid).sum(axis=1) / (total_density.sum(axis=1)+1e-12)
    v_eff = np.gradient(mean_x, times)
    
    mean_v = float(np.mean(v_eff[start_idx:end_idx]))

    return mean_v, v_eff, times, start_idx, end_idx, frac_boundary

##### Compute rho_eff for v_eff (WRONG)
def compute_rho_eff(out, start_idx, end_idx, window_fraction=0.05):
    total_density = np.asarray(out['total_list'])
    M, L = total_density.shape

    x_grid = np.linspace(0, 1.0, L)
    dx = x_grid[1] - x_grid[0]

    ell = window_fraction  # fraction of system length

    rho_eff_list = []

    for t in range(start_idx, end_idx):
        rho_t = total_density[t]

        # front position (rightmost occupied site)
        occupied = np.where(rho_t > 0)[0]
        if len(occupied) == 0:
            continue

        x_max = x_grid[occupied[-1]]

        # front window
        mask = (x_grid >= x_max - ell) & (x_grid <= x_max)
        if mask.sum() == 0:
            continue

        rho_front = rho_t[mask].sum() * dx / ell
        rho_eff_list.append(rho_front)

    return float(np.mean(rho_eff_list))

### Compute blocking P for v_eff (RIGHT)
def compute_blocking_probability(out, start_idx, end_idx):
    """
    Estimate P_block = probability that a forward (+) move is blocked.
    """
    total_density = np.asarray(out["total_list"])
    rho_p = np.asarray(out["rho_p_list"])  # + particles only

    M, L = total_density.shape
    x_grid = np.linspace(0, 1.0, L)
    dx = x_grid[1] - x_grid[0]

    blocked = 0.0
    attempts = 0.0

    for t in range(start_idx, end_idx):
        rho_t = total_density[t]
        rho_p_t = rho_p[t]

        # loop over sites with + particles
        for i in np.where(rho_p_t > 0)[0]:
            j = i + 1
            if j >= L:
                continue

            attempts += rho_p_t[i]

            if rho_t[j] >= 1.0:
                blocked += rho_p_t[i]

    if attempts == 0:
        return 0.0

    return blocked / attempts

### Compute mangetizition both adjusted and not (RIGHT)
def compute_m_of_beta(beta_values, rho_bar, K, lambda_eff):
    beta_values = np.asarray(beta_values, dtype=float)
    m_list = []
    for b in beta_values:
        if b == 0:
            m_list.append(0.0)
            continue
        try: #+ 0.5 * np.log(1 + lambda_eff * (1 + m) /2 * (1 - phi_poisson(rho_bar, K))* np.exp(-b * m))
            m_sol = fixed_point(lambda m: np.tanh(b * m), 0.5, xtol=1e-12, maxiter=200)
        except Exception:
            guesses = [0.0, 0.1, 0.5, 0.9]
            m_sol = None
            for g in guesses:
                try: #+ 0.5 * np.log(1 + lambda_eff * (1 + m) /2 * (1 - phi_poisson(rho_bar, K))* np.exp(-b * m))
                    s = fixed_point(lambda m: np.tanh(b * m), g, xtol=1e-12, maxiter=200)
                    m_sol = s
                    break
                except Exception:
                    continue
            if m_sol is None:
                m_sol = 0.0
        m_list.append(float(m_sol))       
    return np.array(m_list, dtype=float)

def compute_m_of_beta_non(beta_values, rho_bar, K, lambda_eff):
    beta_values = np.asarray(beta_values, dtype=float)
    m_list = []
    for b in beta_values:
        if b == 0:
            m_list.append(0.0)
            continue
        try: #+ 0.5 * np.log(1 + lambda_eff * (1 + m) /2 * (1 - phi_poisson(rho_bar, K))* np.exp(-b * m))
            m_sol = fixed_point(lambda m: np.tanh(b * m), 0.5, xtol=1e-12, maxiter=200)
        except Exception:
            guesses = [0.0, 0.1, 0.5, 0.9]
            m_sol = None
            for g in guesses:
                try: #+ 0.5 * np.log(1 + lambda_eff * (1 + m) /2 * (1 - phi_poisson(rho_bar, K))* np.exp(-b * m))
                    s = fixed_point(lambda m: np.tanh(b * m), g, xtol=1e-12, maxiter=200)
                    m_sol = s
                    break
                except Exception:
                    continue
            if m_sol is None:
                m_sol = 0.0
        m_list.append(float(m_sol))       
    return np.array(m_list, dtype=float)

### v_eff apprixmations
def phi_poisson(rho_bar, K):
    mu = rho_bar / K
    return 1 - poisson.cdf(K - 1, mu)

def phi_nb(rho_bar, K, r_disp):
    r = float(r_disp)
    mu = rho_bar / K
    p = r / (r + mu)
    cdf = nbinom.cdf(K - 1, r, p)
    return 1 - float(cdf)

def v_pred_from_phi(phi_values, lambda_eff, m_beta, beta_values):
    p_plus = (1.0 + m_beta) * 0.5 
    return lambda_eff * p_plus * (1 - phi_values)

def v_pred_TASEP(lambda_eff, rho_bar, K, m_beta):
    p_plus = (1.0 + m_beta) * 0.5 
    return lambda_eff * p_plus * (1 - rho_bar / K)

def v_pred_new_theory(lambda_eff, rho_bar, K, beta, m_beta, m_beta_non, gamma_eff, q = 1):
    return  lambda_eff * (1- rho_bar / K) * (1 + m_beta) / 2 * (1 - 0.25 / np.cosh(beta * m_beta_non) + 0.1 / (np.cosh(beta * m_beta_non)**2))
  #/ (1 + q**2 * gamma_eff / (2 * np.cosh(beta * m_beta_non)))
  #lambda_eff * (1 - rho_bar / K) * (((m_beta + 1) / 2) ** 2 + rho_bar / K * (1 - (m_beta*0.5 + 0.5) **2) / (4 * np.cosh(beta * m_beta_non)))
  #lambda_eff * (1 - rho_bar / K) * (m_beta - rho_bar / K * lambda_eff * (m_beta - np.tanh(beta * m_beta)))

def v_pred_without_phi(lambda_eff, m_beta, beta_values):
    return lambda_eff * 0.5 *  (1.0 + m_beta) 

def v_pred_block(lambda_eff, m_beta_dense, beta_dense, rho_bar, K):
    m_beta_non = compute_m_of_beta_non(beta_dense, rho_bar, K, lambda_eff)
    return lambda_eff * 0.5 * (1.0 + m_beta_dense) * (1 - rho_bar / K * (1.37 + 4/(np.cosh(beta_dense * m_beta_non)))) #- m_beta_non * np.sinh(beta_dense * m_beta_non))))

def v_eff_fit(rho_bar, k, beta, lambda_eff, m_beta, m_beta_non):
    return lambda_eff * (1 +  np.tanh(beta * m_beta)) / 2 * (1 - rho_bar / k * ((1.2552899764748897 - 0.1 - 0.6022927624714487 * rho_bar / k) + (0.15327283599951863 + 0.01) / ((rho_bar/ k) ** (1.5)) / np.cosh(beta * m_beta_non)))

def compute_mean_magnetizatoin(out, start_idx, end_idx):
    m_ts = np.asarray(out["m_global"], dtype=float)
    m_window = m_ts[start_idx:end_idx]
    return float(np.mean(m_window))

### plot v_eff and fits
def fit_and_plot_v_eff(
    beta_values,
    ps_kwargs,
    means,
    stds,
    ses,
    m_means, 
    m_stds,
    m_ses, 
    rho_means, 
    rho_ses,
    block_means, 
    block_ses,
    theta_guess = 500,
    tau_guess = 1,
    bounds=([1e2, 0], [1e3, 10]),
    plot_result=True,
    return_all=True,
):
    beta_values = np.asarray(beta_values, dtype=float)
    empirical_means = np.asarray(means, dtype=float)
    emperical_std =  np.asarray(stds, dtype=float)
    empirical_ses = np.asarray(ses, dtype=float)
    m_means = np.asarray(m_means, dtype=float)
    m_stds = np.asarray(m_stds, dtype=float)
    m_ses   = np.asarray(m_ses, dtype=float)
        
    K = int(ps_kwargs['site_capacity'])
    rho_bar = float(ps_kwargs['N']) / float(ps_kwargs['L'])
    dx = float(ps_kwargs['xlim']) / float(ps_kwargs['L'])
    lambda_eff = float(ps_kwargs['rate_active']) * dx
    gamma_eff = 0.5 * float(ps_kwargs['rate_diffusion']) * (dx ** 2)
    m_beta = compute_m_of_beta(beta_values, rho_bar, K, lambda_eff)
    k_exit = float(ps_kwargs['k_exit'])

    #model function
    """
    def v_model(beta_array, r0, alpha):
        m_beta = compute_m_of_beta(beta_array, rho_bar, K, lambda_eff)

        r_arr = r0 / (1.0 + alpha * m_beta)
        r_arr = np.clip(r_arr, 1e-8, 1e12)

        Phi = np.array([phi_nb(rho_bar, K, r_arr[i]) for i in range(r_arr.size)])

        return v_pred_from_phi(Phi, lambda_eff, m_beta)      
    """
    def compute_r_and_Phi_with_gamma(beta_array, theta, gamma, rho_bar, K, lambda_eff):
        beta_array = np.asarray(beta_array, float)
        m_beta = compute_m_of_beta(beta_array, rho_bar, K, lambda_eff)  # your fixed-point solver (use resummed h_active)
        p_plus = 0.5 * (1.0 + m_beta)

        # front mean density model
        rho_front = rho_bar * (1.0 + gamma * m_beta)   # array

        # initial guess for Phi: Poisson tail at rho_front
        Phi = np.array([phi_poisson(rho_front[i], K) for i in range(len(beta_array))])

        # iterate self-consistently to find r and Phi
        for _ in range(6):
            # compute r according to physical surrogate (theta absorbs k_relax/c0)
            denom = (lambda_eff * p_plus * (1.0 - Phi) + 1e-14)
            r_arr = theta * (rho_front**2) / denom
            r_arr = np.clip(r_arr, 1e-6, 1e12)

            # update Phi using NB tail at mean rho_front and dispersion r_arr
            Phi = np.array([phi_nb(rho_front[i], K, r_arr[i]) for i in range(len(beta_array))])

        return r_arr, Phi, m_beta, rho_front
    
    def v_model_theta_gamma(beta_array, theta, gamma):
        r_arr, Phi, m_beta, rho_front = compute_r_and_Phi_with_gamma(
            beta_array, theta, gamma, rho_bar, K, lambda_eff
        )
        return v_pred_from_phi(Phi, lambda_eff, m_beta, beta_array)

    #curve-fit
    p0 = [float(theta_guess), float(tau_guess)]
    sigma = np.copy(empirical_ses)
    popt, pcov = curve_fit(v_model_theta_gamma, beta_values, empirical_means, p0=p0, sigma=sigma, absolute_sigma=True, bounds=bounds, maxfev=2000000)
    theta_fit, tau_fit = popt[0], popt[1]

    # for plots
    beta_dense = np.linspace(beta_values.min(), beta_values.max(), 400)
    r_fitted_arr, Phi_nb_fit, m_beta_dense, rho_front_dense = compute_r_and_Phi_with_gamma(
        beta_dense, theta_fit, tau_fit, rho_bar, K, lambda_eff
    )
    v_nb_fit = v_pred_from_phi(Phi_nb_fit, lambda_eff, m_beta_dense, beta_dense)

    Phi_po = phi_poisson(rho_bar, K)
    v_po = v_pred_from_phi(Phi_po, lambda_eff, m_beta_dense, beta_dense)

    v_m = v_pred_without_phi(lambda_eff, m_beta_dense, beta_dense)
    
    v_TASEP = v_pred_TASEP(lambda_eff, rho_bar, K, m_beta_dense)
    m_beta_dense = compute_m_of_beta(beta_dense, rho_bar, K, lambda_eff)
    m_beta_non = compute_m_of_beta_non(beta_dense, rho_bar, K, lambda_eff)
    v_block = v_eff_fit(rho_bar, K, beta_dense, lambda_eff, m_beta_dense, m_beta_non)

    v_theory = v_pred_new_theory(lambda_eff, rho_bar, K, beta_dense, m_beta_dense, m_beta_non, gamma_eff)

    if plot_result:
        plt.figure(figsize=(7,5))
        plt.errorbar(beta_values, empirical_means, yerr=empirical_ses, fmt='o', capsize=3, label='simulation ± SE', color='blue')
        #plt.plot(beta_dense, v_po, '--', label='theory: Poisson', linewidth=1.5, color='blue')
        plt.plot(beta_dense, v_m, '--', label='theory: non-exclusion', linewidth=1.5, color='lightblue')
        plt.plot(beta_dense, v_TASEP, '--', label='theory: TASEP', linewidth=1.5, color='royalblue')
        plt.plot(beta_dense, v_block, '--', label='prediction: exclusion', linewidth=1.5, color='navy')
        #plt.plot(beta_dense, v_theory, '--', label='theory: exclusion', linewidth=1.5, color='navy')

        #plt.plot(beta_dense, v_nb_fit, '--', label=f'fit NB: theta={theta_fit:.3g}, tau = {tau_fit:.3g}', linewidth=1.5, color='navy')
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$v_{\mathrm{eff}}$')
        plt.legend()
        plt.xlim(0, 3)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('v_eff_beta_plot_theory.png', dpi=200)
        plt.close()

    if plot_result:
        plt.figure(figsize=(6,4))
        plt.errorbar(beta_values, m_means, yerr=m_ses,
                    fmt='o', capsize=3, label='simulation ± SE')
        plt.plot(beta_dense, m_beta_dense, '--', color='navy',
                label=r'theory: $m=\tanh(\beta m)$')
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$m$')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("global_m_vs_theory.png", dpi=200)
        plt.close()


    m_beta_dense = compute_m_of_beta_non(beta_dense, rho_bar, K, lambda_eff)

    if plot_result:
        plt.figure(figsize=(6,4))
        #plt.errorbar(beta_values, rho_means, yerr=rho_ses, fmt='o', capsize=3, label=r'$\rho_{eff}$')
        plt.errorbar(beta_values, block_means, yerr=block_ses,
            fmt='o', capsize=3, label=r'$p_{block} \pm$SE', color = 'blue')
        plt.plot(beta_dense, 1 - (1 - rho_bar / K * ((1.2552899764748897 - 0.18 - 0.6022927624714487 * rho_bar / K) + (0.15327283599951863 + 0.019) / ((rho_bar/ K) ** (1.5)) / np.cosh(beta_dense * m_beta_dense))), '--', color='navy', label='prediction: exclusion', linewidth=1.5)  # 1 * rho_bar / K + rho_bar / K * 1/2 / (np.cosh(beta_dense * m_beta_dense))) #- m_beta_dense * np.sinh(beta_dense * m_beta_dense)))
        #plt.plot(beta_dense, 0.66 + 1.5 * (1 - np.sqrt(1 - (2 * 0.666666n * 0.333)/(np.cosh(beta_dense * m_beta_dense) + m_beta_dense * np.sinh(m_beta_dense  * beta_dense)))))
        plt.hlines(rho_bar / K, 0, beta_values[-1],linestyles='--', color='royalblue', label='theory: TASEP', linewidth=1.5)
        plt.xlabel(r'$\beta$')
        plt.ylabel(r'$p$')
        plt.legend()
        plt.xlim(0,3)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("rho_vs_rho.png", dpi=200)
        plt.close()

    out = {  
        'popt': popt,
        'pcov': pcov,
        'theta_fit': theta_fit,
        'beta': beta_values,
        'm_beta': m_beta,
        'r_fitted_arr': r_fitted_arr,
        'Phi_nb_fit': Phi_nb_fit,
        'v_nb_fit': v_nb_fit,
        'Phi_poisson': Phi_po,
        'v_poisson': v_po,
        'rho_bar': rho_bar,
        'lambda_eff': lambda_eff,
    }

    if return_all:
        return popt, pcov, out
    else:
        return popt, pcov

##### D_EFF ################################
####  Compute D_eff from results
def compute_D_eff_active(out, ps, start_idx, end_idx):
    times = out['times_obs']
    pos_list = out['pos_list']
    dx = ps.dx

    S_vals = []
    t_vals = []

    pos0 = pos_list[start_idx] * dx
    t0 = times[start_idx]

    for k in range(start_idx + 1, end_idx):
        pos_t = pos_list[k] * dx
        n = min(len(pos0), len(pos_t))
        if n < 2:
            continue

        ri = pos_t[:n] - pos0[:n]
        rbar = np.mean(ri)

        S = np.sum((ri - rbar)**2) / (n - 1)
        S_vals.append(S)
        t_vals.append(times[k] - t0)

    coef = np.polyfit(t_vals, S_vals, 1)
    return coef[0]   # slope = D_eff

##### Theoretical predictions
def D_eff_global(beta, m_beta, gamma_eff, lambda_eff):
    return gamma_eff + (lambda_eff**2) / (np.cosh(beta * m_beta)**3)

def D_eff_local(beta, m_beta, gamma_eff, lambda_eff):
    return (gamma_eff + lambda_eff**2) * m_beta / m_beta

def D_eff_theory(beta, m_beta, gamma_eff, lambda_eff, m_beta_non, rho_bar, K):
    return gamma_eff +  lambda_eff**2 * (1 - rho_bar / K) / (np.cosh(beta * m_beta_non)**3) 
    #gamma_eff - lambda_eff**2 / (2 * (1 - rho_bar / K * beta / (np.cosh(beta * m_beta_non**2)))) * ((1 - 2 * rho_bar / K) + np.tanh(beta * m_beta)**2 / ((1 - rho_bar / K * beta / (np.cosh(beta * m_beta_non**2)))**2))

def D_eff_theory_2(beta, m_beta, gamma_eff, lambda_eff, m_beta_non, rho_bar, K):
    return gamma_eff + lambda_eff**2 * (1 - rho_bar / K) * abs(1 -  rho_bar / K) / (np.cosh(beta * m_beta_non)**3) 

def D_eff_theory_3(beta, m_beta, gamma_eff, lambda_eff, m_beta_non, rho_bar, K):
    return gamma_eff + lambda_eff**2 * (1 - rho_bar / K) * abs(1 - 2 *  rho_bar / K) / (np.cosh(beta * m_beta_non)) - lambda_eff**2 * (1- rho_bar / K)**2 * np.sinh(beta * m_beta_non)**2 / (np.cosh(beta * m_beta_non)**3)

def D_eff_theory_4(beta, m_beta, gamma_eff, lambda_eff, m_beta_non, rho_bar, K):
    return gamma_eff + lambda_eff**2 * (1 - rho_bar / K) / np.cosh(beta * m_beta_non) * (abs(1 - 2* rho_bar / K) + rho_bar / K / (np.cosh(beta * m_beta_non)**2))

### Fit and plots for exclusion
def f(beta, m_beta, rho_bar, K): 
    C0 = 1.2552899764748897 
    C1 = 0.6022927624714487
    C2 = 0.15327283599951863
    
    return  rho_bar / K * (C0 - C1 * rho_bar / K + C2 / ((rho_bar / K)**1.5 * np.cosh(beta * m_beta)))

def h(A, beta, m_beta):
    return A / np.cosh(beta* m_beta)

def fit_D_eff(beta, m_beta, rho_bar, K, gamma_eff, lambda_eff):
    A = 2.5
    return gamma_eff + lambda_eff**2 / (np.cosh(beta * m_beta)**3) * (1 - f(beta, m_beta, rho_bar, K))**2 * h(A, beta, m_beta)

#### Plot D_eff and fits
def plot_D_eff_vs_beta(
    beta_values,
    D_means,
    D_ses,
    ps_kwargs,
    plot_name_prefix="D_eff",
):  
    print(D_means)
    beta_values = np.asarray(beta_values, float)
    D_means = np.asarray(D_means, float)
    D_ses = np.asarray(D_ses, float)

    K = int(ps_kwargs['site_capacity'])
    rho_bar = float(ps_kwargs['N']) / float(ps_kwargs['L'])

    dx = ps_kwargs['xlim'] / ps_kwargs['L']
    lambda_eff = ps_kwargs['rate_active'] * dx
    gamma_eff = 0.5 * ps_kwargs['rate_diffusion'] * dx**2

    # magnetization (non-exclusion, consistent with theory)

    beta_dense = np.linspace(beta_values.min(), beta_values.max(), 400)
    m_dense = compute_m_of_beta_non(beta_dense, rho_bar, K, lambda_eff)
    m_beta_reg = compute_m_of_beta(beta_dense, rho_bar, K, lambda_eff)

    D_global = D_eff_global(beta_dense, m_dense, gamma_eff, lambda_eff)
    D_local  = D_eff_local(beta_dense, m_dense, gamma_eff, lambda_eff)

    D_fit = fit_D_eff(beta_dense, m_dense, rho_bar, K, gamma_eff, lambda_eff)

    D_theory = D_eff_theory(beta_dense, m_beta_reg, gamma_eff, lambda_eff, m_dense, rho_bar, K)
    D_theory_2 = D_eff_theory_2(beta_dense, m_beta_reg, gamma_eff, lambda_eff, m_dense, rho_bar, K)
    D_theory_3 = D_eff_theory_3(beta_dense, m_beta_reg, gamma_eff, lambda_eff, m_dense, rho_bar, K)
    D_theory_4 = D_eff_theory_4(beta_dense, m_beta_reg, gamma_eff, lambda_eff, m_dense, rho_bar, K)
    
    plt.figure(figsize=(6,4))

    plt.errorbar(
        beta_values,
        D_means*2.5,
        yerr=D_ses*2.5,
        fmt='o',
        capsize=3,
        label='simulation ± SE',
        color='blue',
    )

    plt.plot(
        beta_dense,
        D_global,
        '--',
        color='royalblue',
        label='theory: non-exclusion',
    )

    plt.plot(
        beta_dense,
        D_theory,
        '--',
        color='navy',
        label='theory: exclusion'
    )
    """
    plt.plot(
        beta_dense,
        D_theory_2,
        '--',
        color='blue',
        label='prediction: exclusion 2'
    )
    plt.plot(
        beta_dense,
        D_theory_3,
        '--',
        color='lightblue',
        label='prediction: exclusion 3'
    )
    """
    plt.plot(
        beta_dense,
        D_theory_4,
        '--',
        color='black',
        label='prediction: exclusion'
    )
   
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$D_{\mathrm{eff}}$')
    plt.legend()
    plt.xlim(0, 3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_name_prefix}_beta.png", dpi=200)
    plt.close()

##### OUTS #################################
### plot outs and fits
def plot_outs(
    beta_values,
    n_runs_per_beta,
    ps_kwargs,
    run_kwargs,
    outs,
    do_theory_fit = True,
    plot_theory = True,
):
        beta_values = np.array(beta_values)
        K = int(ps_kwargs['site_capacity'])
        rho_bar = float(ps_kwargs['N']) / float(ps_kwargs['L'])
        dx = float(ps_kwargs['xlim']) / float(ps_kwargs['L'])
        lambda_eff = float(ps_kwargs['rate_active']) * dx
        m_beta = compute_m_of_beta(beta_values, rho_bar, K, lambda_eff)
        n_beta = len(beta_values)

        # anchors (use ps_kwargs to know anchors)
        anchor_positions = np.array(ps_kwargs["anchor_positions"])
        xlim = ps_kwargs["xlim"]
        L = ps_kwargs["L"]
        anchor_idxs = np.unique(np.round((anchor_positions / xlim) * (L - 1)).astype(int))
        centers = np.array(anchor_idxs, dtype=int)
        nA = len(centers)

        dx = 1.0 / L
        anchor_radius = ps_kwargs["anchor_radius"]
        r_idx = int(np.ceil(anchor_radius / dx))

        # model parameters
        T_sim = run_kwargs['T']
        k_exit = ps_kwargs["k_exit"]
        k_on = ps_kwargs["k_on"]
        k_off = ps_kwargs["k_off"]
        rho_bar = ps_kwargs["N"] / L / ps_kwargs['site_capacity']

        # storage (mean over runs)
        total_mean = np.zeros(n_beta)
        total_std  = np.zeros(n_beta)
        region_mean = np.zeros((n_beta, nA))
        region_std  = np.zeros((n_beta, nA))

        for iB in range(n_beta):
            total_runs = []
            region_runs = []

            for run in range(n_runs_per_beta):
                out = outs[iB][run]
                exit_x = np.array(out["exit_positions"], dtype=int)

                site_to_gid = np.full(L, -1, dtype=int)
                for a, c in enumerate(centers):
                    left  = max(0, c - r_idx)
                    right = min(L - 1, c + r_idx)
                    site_to_gid[left:right+1] = a

                gids = np.array([site_to_gid[x] if 0 <= x < L else -1 for x in exit_x])

                total_runs.append(len(exit_x))
                per_anchor = np.array([(gids == a).sum() for a in range(nA)])
                region_runs.append(per_anchor)

            # convert to arrays
            total_runs = np.array(total_runs)
            region_runs = np.array(region_runs)

            total_mean[iB] = total_runs.mean()
            total_std[iB]  = total_runs.std()

            region_mean[iB, :] = region_runs.mean(axis=0)
            region_std[iB, :]  = region_runs.std(axis=0)

        plt.figure(figsize=(9,6))
        colors = plt.get_cmap("Blues")

        for a in range(nA):
            plt.errorbar(
                beta_values,
                region_mean[:, a],
                yerr=region_std[:, a],
                fmt='o',
                markersize=5,
                capsize=3,
                color=colors(0.5 + 0.1 * a),
                label=f"anchor {a}",
            )

        # total exits
        plt.errorbar(
            beta_values,
            total_mean,
            yerr=total_std,
            fmt='o',
            markersize=6,
            capsize=3,
            color=colors(0.9),
            label="total exits"
        )

        if do_theory_fit:
            #prefactor and shape m
            A = T_sim * k_exit * (k_on / (k_exit + k_off))
            shape_beta = 0.5 * (1.0 - m_beta)
            
            def model(beta_arr, *S_list):
                S_arr = np.array(S_list)
                pred = np.zeros_like(beta_arr)
                for a in range(nA):
                    pred += S_arr[a] * rho_bar * shape_beta
                return A * pred

            # fit each region individually
            S_fits = []
            S_covs = []

            for a in range(nA):

                def region_model(beta_arr, S_i):
                    return A * (rho_bar * S_i) * shape_beta

                popt, pcov = curve_fit(
                    region_model,
                    beta_values,
                    region_mean[:, a],
                    sigma=region_std[:, a] + 1e-8,
                    absolute_sigma=True,
                    p0=[1.0],
                    maxfev=2000000,
                )

                S_fits.append(popt[0])
                S_covs.append(pcov)

            S_fits = np.array(S_fits)

            if plot_theory:
                beta_dense = np.linspace(beta_values.min(), beta_values.max(), 400)
                m_dense = compute_m_of_beta(beta_dense, rho_bar, K, lambda_eff)
                shape_dense = 0.5 * (1.0 - m_dense)

                # Per-anchor theory curves
                for a in range(nA):
                    S_i = S_fits[a]
                    theory_i = A * (rho_bar * S_i) * shape_dense
                    plt.plot(beta_dense, theory_i, '-', color=colors(0.55 + 0.1*a),
                            label=f"anchor {a} (theory: S={S_i:.3g})")

                # total theory
                total_theory = np.zeros_like(beta_dense)
                for a in range(nA):
                    S_i = S_fits[a]
                    total_theory += A * (rho_bar * S_i) * shape_dense

                plt.plot(beta_dense, total_theory, '--', color=colors(0.9),
                        label="total (theory)", linewidth=2)
            
        plt.xlabel("β")
        plt.ylabel("Number of exits (final timestep)")
        plt.title("Exits per anchor vs β")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("exits_vs_beta.png", dpi=200)
        plt.close()

        return total_mean, total_std, region_mean, region_std
        
######### Sweep over b betas n times ###############################
def sweep_over_betas(beta_values, n_runs_per_beta=10, run=True, save_dict=None):
    run_kwargs = dict(
            T = 20,
            obs_dt = 0.1,
            record_fft=True,
            record_var=True,
        )

    if run:
        ps_kwargs = dict(
            L = 1000,
            xlim = 1,
            rate_diffusion = 0.02,
            rate_active = 5,
            flip_rate_fn = None,
            init = 'poisson',
            N = 500,
            scale_rates = False,
            local_kernel_sigma = 0.005, #0.2, #0.005, # 0.02, #0.005,
            minus_anchor = True,
            periodic = False, #False,
            immobilize_when_anchored = True,
            anchor_radius= 0.003,
            anchor_positions=None, #[0.25, 0.60, 0.80],
            site_capacity = 1,
            crowding_suppresses_rates = False,
            k_on =0, #10,
            k_off =0, # 5,
            k_exit =0, #5,
        )
        
        init_kwargs = dict(
            rho0_plus = make_exp_gradient(
                L = 1000,
                N = 500,
                frac_plus= 0.75,
                decay_length= 0.35,
                anchor_positions=None, #(0.25, 0.60, 0.80),
                anchor_peak_width=0.01,
                anchor_peak_mass=0.03,
            )[0],
            rho0_minus= make_exp_gradient(
                L = 1000,
                N = 500,
                frac_plus= 0.75,
                decay_length= 0.2,
                anchor_positions=None, #(0.25, 0.60, 0.80),
                anchor_peak_width=0.01,
                anchor_peak_mass=0.03,
            )[1],
        )

        means = []
        stds = []
        ses = []
        raw_by_beta = []
        outs = []
        D_means = []
        D_ses = []
        m_means = []
        m_stds = []
        m_ses = []
        rho_means = []
        rho_ses = []
        block_means = []
        block_ses = []

        for b in beta_values:
            print("beta =", b)
            m, s, se, v_array, out_list, m_mean, m_std, m_se, rho_mean, rho_se, block_mean, block_se, D_mean, D_se = sweep_beta_ensemble(
                beta=b,
                n_runs=n_runs_per_beta,
                ps_kwargs=ps_kwargs,
                init_kwargs = init_kwargs,
                run_kwargs=run_kwargs,
                rng_seeds=None,
            )
            means.append(m)
            stds.append(s)
            ses.append(se)
            raw_by_beta.append(v_array)
            outs.append(out_list)
            D_means.append(D_mean)
            D_ses.append(D_se)
            m_means.append(m_mean)
            m_stds.append(m_std)
            m_ses.append(m_se)
            rho_means.append(rho_mean)
            rho_ses.append(rho_se)
            block_means.append(block_mean)
            block_ses.append(block_se)

        means = np.array(means)
        stds = np.array(stds)
        ses = np.array(ses)
        D_means = np.array(D_means)
        D_ses = np.array(D_ses)
        m_means = np.array(m_means)
        m_stds = np.array(m_stds)
        m_ses = np.array(m_ses)
        rho_means = np.array(rho_means)
        rho_ses = np.array(rho_ses)
        block_means = np.array(block_means)
        block_ses = np.array(block_ses)

    else:
        data = np.load("CHANGES_simulation_out_sweep.npz", allow_pickle=True)
        save_dict = dict(data)
        beta_values = save_dict['beta_values']
        means = save_dict['means']
        stds = save_dict['stds']
        ses = save_dict['ses']
        D_means = save_dict['D_means']
        D_ses = save_dict['D_ses']
        m_means = save_dict['m_means']
        m_stds = save_dict['m_stds']
        m_ses = save_dict['m_ses']
        rho_means = save_dict['rho_means']
        rho_ses = save_dict['rho_ses']
        block_means = save_dict[ 'block_means']
        block_ses = save_dict[ 'block_ses']
        ps_kwargs = save_dict['ps_kwargs'].item()
        outs = save_dict['outs']

    pre_dict = {
        'beta_values': beta_values, 
        'means': means,
        'stds': stds,
        'ses': ses,
        'D_means': D_means,
        'D_ses': D_ses,
        'm_means': m_means,
        'm_stds': m_stds,
        'm_ses': m_ses, 
        'rho_means': rho_means,
        'rho_ses': rho_ses,
        'block_means': block_means,
        'block_ses': block_ses, 
        'ps_kwargs': ps_kwargs,
        'outs': outs, 
    }
    if run: 
        np.savez("CHANGES_simulation_out_sweep.npz", **pre_dict)

    print(means, ses, D_means, D_ses)

    popt, pcov, fit_out = fit_and_plot_v_eff(
        beta_values,
        ps_kwargs,
        means,
        stds,
        ses,   
        m_means,
        m_stds,
        m_ses,
        rho_means,
        rho_ses,
        block_means, 
        block_ses,
    )

    
    plot_D_eff_vs_beta(beta_values, D_means, D_ses, ps_kwargs)

    """
    total_mean, total_std, region_mean, region_std = plot_outs(
        beta_values,
        n_runs_per_beta,
        ps_kwargs, 
        run_kwargs,
        outs,
    )
    """

    save_dict = {
    'beta_values': beta_values, 
    'means': means,
    'stds': stds,
    'ses': ses,
    'D_means': D_means,
    'D_ses': D_ses,
    'm_means': m_means,
    'm_stds': m_stds,
    'm_ses': m_ses,
    'rho_means': rho_means,
    'rho_ses': rho_ses,
    'block_means': block_means,
    'block_ses': block_ses, 
    'raw_by_beta': popt,
    'popt': popt,
    'pcov': pcov,
    'fit_out': fit_out,
    'ps_kwargs': ps_kwargs, 
    'outs': outs, 
    #'total_mean': total_mean,
    #'total_std': total_std,
    #'region_mean': region_mean,
    #'region_std': region_std,   
    }

    return save_dict

beta_values = np.linspace(0, 3, 11)
run = True
save_dict = sweep_over_betas(beta_values, n_runs_per_beta=3, run=run)
if run: 
    np.savez("CHANGES_after_simulation_out_sweep.npz", **save_dict)

