####### LOCAL STRUCTURE
### IMPORTS
from PARTICLE_solver_CLASS import ParticleSystem
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Dict, Any
from vispy import app, scene, io
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.optimize import fixed_point, curve_fit
from scipy.stats import nbinom, poisson
import os
SAVE_DIR = "local_structure_results"
os.makedirs(SAVE_DIR, exist_ok=True)

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

def extract_structure_observables_from_out(
    out,
    start_fraction=0.5,
    k_max=None
):
    T = len(out['times_obs'])
    start_idx = int(start_fraction * T)

    # Variance (density clustering proxy)
    var_ts = np.asarray(out['var_list'], dtype=float)
    var_mean = var_ts[start_idx:].mean()
    var_std  = var_ts[start_idx:].std(ddof=1)

    # Fourier amplitudes
    fft_amp = np.asarray(out['fft_amp_list'], dtype=float)  # (T, k)
    if k_max is not None:
        fft_amp = fft_amp[:, :k_max]

    fft_mean = fft_amp[start_idx:].mean(axis=0)
    fft_std  = fft_amp[start_idx:].std(axis=0, ddof=1)

    # ignore k=0 mode
    dominant_k = int(np.argmax(fft_mean[1:]) + 1)

    # low-k clustering strength
    k_cut = min(25, fft_mean.shape[0])
    low_k_power = float(np.sum(fft_mean[1:k_cut]))

    # pLocal magnetization statistics
    m_local = np.asarray(out['m_local_list'], dtype=float)  # (T, L)
    m_local_ss = m_local[start_idx:]
    m_local_var = float(np.var(m_local_ss))

    #low-k var.
    k_cut = min(25, fft_amp.shape[1])
    lowk_variance = float(
        np.mean(np.sum(fft_amp[start_idx:, 1:k_cut]**2, axis=1))
    )

    return {
        'var_mean': var_mean,
        'var_std': var_std,
        'fft_mean': fft_mean,
        'fft_std': fft_std,
        'dominant_k': dominant_k,
        'low_k_power': low_k_power,
        'm_local_var': m_local_var,
        'lowk_variance': lowk_variance,
    }

def sweep_beta_structure_ensemble(
    beta,
    n_runs,
    ps_kwargs,
    init_kwargs,
    run_kwargs,
    start_fraction=0.5,
    k_max=None,
    rng_seeds=None,
):
    ensemble_results = []

    for i in range(n_runs):
        if rng_seeds is None:
            rng = None
        else:
            rng = np.random.default_rng(int(rng_seeds[i]))

        ps = ParticleSystem(beta=beta, rng=rng, **ps_kwargs, **init_kwargs)
        out = ps.run(**run_kwargs)

        obs = extract_structure_observables_from_out(
            out,
            start_fraction=start_fraction,
            k_max=k_max,
        )

        ensemble_results.append({
            **obs,
            'out': out
        })

    # ---- ensemble averages
    var_means = np.array([r['var_mean'] for r in ensemble_results])
    lowk_means = np.array([r['low_k_power'] for r in ensemble_results])
    dom_ks = np.array([r['dominant_k'] for r in ensemble_results])
    mloc_vars = np.array([r['m_local_var'] for r in ensemble_results])
    lowk_variance_means = np.array([r['lowk_variance'] for r in ensemble_results])

    fft_mean_stack = np.stack([r['fft_mean'] for r in ensemble_results], axis=0)

    return {
        'var_mean': var_means.mean(),
        'var_se': var_means.std(ddof=1) / np.sqrt(n_runs),

        'low_k_power_mean': lowk_means.mean(),
        'low_k_power_se': lowk_means.std(ddof=1) / np.sqrt(n_runs),

        'dominant_k_mode': int(np.round(dom_ks.mean())),

        'm_local_var_mean': mloc_vars.mean(),
        'm_local_var_se': mloc_vars.std(ddof=1) / np.sqrt(n_runs),

        'fft_mean_mean': fft_mean_stack.mean(axis=0),
        'fft_mean_se': fft_mean_stack.std(axis=0, ddof=1) / np.sqrt(n_runs),

        'lowk_var_mean': lowk_variance_means.mean(),
        'lowk_var_se':lowk_variance_means.std(ddof=1) / np.sqrt(n_runs),

        'raw': ensemble_results,
    }

def sweep_betas_for_structures(
    beta_values,
    n_runs_per_beta,
    ps_kwargs,
    init_kwargs,
    run_kwargs,
    start_fraction=0.5,
    k_max=None,
):
    results = {}

    for beta in beta_values:
        print(f"β = {beta}")

        res = sweep_beta_structure_ensemble(
            beta=beta,
            n_runs=n_runs_per_beta,
            ps_kwargs=ps_kwargs,
            init_kwargs=init_kwargs,
            run_kwargs=run_kwargs,
            start_fraction=start_fraction,
            k_max=k_max,
        )

        results[beta] = res

    return results

def time_to_pattern(out, threshold=0.05, k=1):
    fft_amp = np.asarray(out['fft_amp_list'])
    times = out['times_obs']

    for i, amp in enumerate(fft_amp[:, k]):
        if amp > threshold:
            return times[i]
    return np.nan
def ensemble_time_to_pattern(raw_outs, k=1, threshold=0.05):
    times = []
    for out in raw_outs:
        t = time_to_pattern(out, threshold=threshold, k=k)
        if not np.isnan(t):
            times.append(t)
    return np.mean(times), np.std(times) / np.sqrt(len(times))
def cluster_size_distribution(rho, threshold):
    occupied = rho > threshold
    clusters = []
    count = 0
    for val in occupied:
        if val:
            count += 1
        elif count > 0:
            clusters.append(count)
            count = 0
    if count > 0:
        clusters.append(count)
    return np.array(clusters)
def temporal_autocorrelation(out, lag=1):
    total = np.asarray(out['total_list'])
    corr = []

    for t in range(len(total) - lag):
        c = np.mean(total[t] * total[t + lag])
        corr.append(c)

    return np.mean(corr)
def lowk_variance_time(out, k_cut=25):
    fft_amp = np.asarray(out['fft_amp_list'])  # shape (T, k)
    return np.sum(fft_amp[:, 1:k_cut+1]**2, axis=1)
def spectral_entropy(fft_mean, k_max=None):
    if k_max is not None:
        fft_mean = fft_mean[:k_max]

    power = fft_mean[1:]**2  # drop k=0
    p = power / np.sum(power)
    return -np.sum(p * np.log(p + 1e-12))
def mode_competition_ratio(fft_mean):
    amps = fft_mean[1:]  # drop k=0
    k_star = np.argmax(amps)
    return amps[k_star] / (np.sum(amps) - amps[k_star] + 1e-12)
def extract_growth_rate(out, k=1, t_min=0.0, t_max=None, amp_min=1e-4):
    times = np.asarray(out['times_obs'])
    amps = np.asarray(out['fft_amp_list'])[:, k]

    mask = times >= t_min
    if t_max is not None:
        mask &= times <= t_max
    mask &= amps > amp_min

    if np.sum(mask) < 3:
        return np.nan  # not enough data to fit

    t_fit = times[mask]
    a_fit = amps[mask]

    # log-linear fit
    coeffs = np.polyfit(t_fit, np.log(a_fit), 1)
    gamma = coeffs[0]
    return gamma

# definately keep
def plot_lowk_power_vs_beta(results):
    betas = np.array(sorted(results.keys()))
    power = np.array([results[b]['low_k_power_mean'] for b in betas])
    se = np.array([results[b]['low_k_power_se'] for b in betas])
    print('lowk_power_mean = ', power)
    print('lowk_power_se = ', se)
    plt.figure(figsize=(6, 4))
    plt.errorbar(betas, power, yerr=se, fmt='o-', capsize=3, color='navy')
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\sum_{k=1}^{k_c} |A_k(t)|^2$")
    plt.xlim(0.05, betas[-1]+0.05) #results[0]['raw'][0]['out']['times_obs'][-1]
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/lowk_power_vs_beta.png", dpi=300)
    plt.close()
    print('power = ', power)
    print('se = ', se)


    betas = np.array(sorted(results.keys()))
    k_star = np.array([results[b]['dominant_k_mode'] for b in betas])
    wavelength = L / k_star

    plt.figure(figsize=(6, 4))
    plt.plot(betas, wavelength, 'o-')
    plt.xlabel(r"$\beta$")
    plt.ylabel("Dominant wavelength")
    plt.xlim(0.05, betas[-1]+0.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/dominant_wavelength_vs_beta.png", dpi=300)
    plt.close()
def plot_fft_spectrum_heatmap(results):
    betas = np.array(sorted(results.keys()))
    spectra = np.array([results[b]['fft_mean_mean'] for b in betas])

    # Skip k=0 and limit to k=1..100
    spectra = spectra[:, 1:201]  # columns 1 to 100
    k_vals = np.arange(1, spectra.shape[1]+1)  # 1..100

    plt.figure(figsize=(8, 5))
    # Flip axes by transposing spectra and swapping extent
    plt.imshow(
        spectra.T,  # transpose to flip axes
        aspect='auto',
        origin='lower',
        extent=[betas[0], betas[-1], k_vals[0], k_vals[-1]],
        cmap='viridis'
    )
    plt.colorbar(label=r"$\langle |A_k| \rangle$")
    plt.xlabel(r"$\beta$")
    plt.ylabel("mode k")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/fft_spectrum_heatmap.png", dpi=300)
    plt.close()
def plot_lowk_modes_vs_beta(results, k_max=5):
    betas = np.array(sorted(results.keys()))

    plt.figure(figsize=(7, 5))

    cmap = plt.cm.ocean
    colors = cmap(np.linspace(0.1, 0.9, k_max))

    for k, color in zip(range(1, k_max + 1), colors):
        amps = np.array([results[b]['fft_mean_mean'][k] for b in betas])
        errs = np.array([results[b]['fft_mean_se'][k] for b in betas])

        plt.errorbar(
            betas,
            amps,
            yerr=errs,
            marker='o',
            linestyle='-',
            color=color,
            capsize=3,
            label=f"k={k}",
        )

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |A_k| \rangle$")
    plt.legend()
    plt.xlim(0.05, betas[-1]+0.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/lowk_modes_vs_beta.png", dpi=300)
    plt.close()

def plot_lowk_variance_time(beta_results_dict, k_cut=25, save_dir="local_structure_results"):
    plt.figure(figsize=(7, 5))

    betas = sorted(beta_results_dict.keys())
    cmap = plt.cm.ocean
    colors = cmap(np.linspace(0.1, 0.9, len(betas)))

    for beta, color in zip(betas, colors):
        out = beta_results_dict[beta]
        lv = lowk_variance_time(out, k_cut=k_cut)

        plt.plot(
            out['times_obs'],
            np.sqrt(np.array(lv)),
            color=color,
            label=f"β={beta:.1f}",
        )

        t_max = out['times_obs'][-1]

    plt.xlabel("t")
    plt.ylabel(r"$\sqrt{\sum_{k=1}^{k_c} |A_k(t)|^2}$")
    plt.legend(loc='upper left')
    plt.xlim(0, t_max)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/lowk_variance_time.png", dpi=300)
    plt.close()
def plot_mode_growth_time(beta_results_dict, k=1):
    plt.figure(figsize=(7, 5))

    betas = sorted(beta_results_dict.keys())
    cmap = plt.cm.ocean
    colors = cmap(np.linspace(0.1, 0.9, len(betas)))

    for beta, color in zip(betas, colors):
        out = beta_results_dict[beta]

        plt.plot(
            out['times_obs'],
            out['fft_amp_list'][:, k],
            color=color,
            label=f"β={beta:.1f}",
        )

        t_max = out['times_obs'][-1]

    plt.xlabel("t")
    plt.ylabel(r"$|A_k(t)|$")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, t_max)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/mode_{k}_growth_time.png", dpi=300)
    plt.close()

def plot_dominant_mode_amplitude_vs_beta(results, save_dir="local_structure_results"):
    betas = np.array(sorted(results.keys()))
    amp_star = []
    amp_star_se = []
    k_star = []

    for b in betas:
        fft_mean = results[b]['fft_mean_mean']
        fft_se   = results[b]['fft_mean_se']

        k = np.argmax(fft_mean[1:]) + 1  # dominant mode (exclude k=0)
        amp_star.append(fft_mean[k])
        amp_star_se.append(fft_se[k])
        k_star.append(k)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        betas,
        amp_star,
        yerr=amp_star_se,
        fmt='o-',
        color='navy',
        capsize=3,
    )

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$\langle |A_{k^\star}| \rangle$")
    plt.xlim(-0.05, betas[-1]+0.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dominant_mode_amplitude_vs_beta.png", dpi=300)
    plt.close()

    print('amp_star = ', amp_star) 
    print('amp_star_se = ', amp_star_se)
    print('k_star = ', k_star)

def plot_spectral_entropy_vs_beta(results, k_max=25, save_dir="local_structure_results"):
    betas = sorted(results.keys())
    H_mean = []
    H_se = []

    for b in betas:
        H_runs = [
            spectral_entropy(r['fft_mean'], k_max=k_max)
            for r in results[b]['raw']
        ]
        H_runs = np.asarray(H_runs)

        H_mean.append(H_runs.mean())
        H_se.append(H_runs.std(ddof=1) / np.sqrt(len(H_runs)))
    print("H_mean = ", H_mean)
    print("H_se = ", H_se)
    plt.figure(figsize=(6, 4))
    plt.errorbar(betas, H_mean, yerr=H_se, fmt='o', capsize=3, color='navy')
    plt.xlabel(r"$\beta$")
    plt.ylabel("H")
    plt.xlim(-0.05, betas[-1]+0.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/spectral_entropy_vs_beta.png", dpi=300)
    plt.close()
def plot_mode_competition_vs_beta(results, save_dir="local_structure_results"):
    betas = sorted(results.keys())
    R_mean = []
    R_se = []

    for b in betas:
        R_runs = [
            mode_competition_ratio(r['fft_mean'])
            for r in results[b]['raw']
        ]
        R_runs = np.asarray(R_runs)

        R_mean.append(R_runs.mean())
        R_se.append(R_runs.std(ddof=1) / np.sqrt(len(R_runs)))
    print("R_mean = ", R_mean)
    print("R_se = ", R_se)
    plt.figure(figsize=(6, 4))
    plt.errorbar(betas, R_mean, yerr=R_se, fmt='o', capsize=3, color='navy')
    plt.xlabel(r"$\beta$")
    plt.ylabel("R")
    plt.xlim(-0.05, betas[-1]+0.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mode_competition_vs_beta.png", dpi=300)
    plt.close()

def growth_rate_vs_beta(results, beta_results_dict, t_min=0.0, t_max=None,save_dir="local_structure_results"):
    betas = sorted(results.keys())

    gamma_mean = []
    gamma_se = []

    for b in betas:
        gammas_b = []

        for r in results[b]['raw']:
            fft_mean = r['fft_mean']

            # dominant mode for THIS run
            k_star = np.argmax(fft_mean[1:]) + 1

            gamma = extract_growth_rate(
                r['out'],
                k=k_star,
                t_min=t_min,
                t_max=t_max
            )

            if not np.isnan(gamma):
                gammas_b.append(gamma)

        gammas_b = np.asarray(gammas_b)

        if len(gammas_b) == 0:
            gamma_mean.append(np.nan)
            gamma_se.append(np.nan)
        else:
            gamma_mean.append(gammas_b.mean())
            gamma_se.append(gammas_b.std(ddof=1) / np.sqrt(len(gammas_b)))

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        betas,
        gamma_mean,
        yerr=gamma_se,
        fmt='o',
        color='navy',
        capsize=3,
    )

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"Growth rate $\gamma$")
    plt.xlim(-0.05, betas[-1] + 0.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/growth_rate_vs_beta.png", dpi=300)
    plt.close()
    print('gamma_mean = ', gamma_mean)
    print('gamma_se = ', gamma_se)

# not to useful
def plot_time_to_pattern_vs_beta(results, threshold=0.05, k=1):
    betas = sorted(results.keys())
    t_mean, t_se = [], []

    for b in betas:
        mean, se = ensemble_time_to_pattern(
            [r['out'] for r in results[b]['raw']],
            k=k,
            threshold=threshold
        )
        t_mean.append(mean)
        t_se.append(se)

    plt.figure(figsize=(6, 4))
    plt.errorbar(betas, t_mean, yerr=t_se, fmt='o-', capsize=3)
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$t_{\mathrm{pattern}}$")
    plt.title("Time to pattern formation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/time_to_pattern_vs_beta.png", dpi=300)
    plt.close()
def plot_cluster_distribution(out, threshold, label=None):
    rho = np.asarray(out['total_list'][-1])
    sizes = cluster_size_distribution(rho, threshold)

    plt.figure(figsize=(5, 4))
    plt.hist(sizes, bins=30, density=True, alpha=0.7)
    plt.xlabel("Cluster size")
    plt.ylabel("PDF")
    title = "Cluster-size distribution"
    if label:
        title += f" ({label})"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    fname = f"{SAVE_DIR}/cluster_distribution_{label}.png" if label else f"{SAVE_DIR}/cluster_distribution.png"
    plt.savefig(fname, dpi=300)
    plt.close()

#maybe not to useful
def plot_temporal_autocorrelation(out, max_lag=10):
    corr = [temporal_autocorrelation(out, lag=l) for l in range(1, max_lag + 1)]

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_lag + 1), corr, 'o-')
    plt.xlabel("Time lag")
    plt.ylabel("C(t)")
    plt.title("Temporal autocorrelation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/temporal_autocorrelation.png", dpi=300)
    plt.close()
def plot_autocorrelation_vs_beta(results, lag=1):
    betas = sorted(results.keys())
    corr = []

    for b in betas:
        vals = [temporal_autocorrelation(r['out'], lag=lag) for r in results[b]['raw']]
        corr.append(np.mean(vals))

    plt.figure(figsize=(6, 4))
    plt.plot(betas, corr, 'o-')
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$C(\Delta t)$")
    plt.title("Temporal correlations vs β")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/autocorrelation_vs_beta.png", dpi=300)
    plt.close()

### save functions
def save_results(fname, beta_values, results, ps_kwargs, run_kwargs):
    np.savez(
        fname,
        beta_values=beta_values,
        results=results,
        ps_kwargs=ps_kwargs,
        run_kwargs=run_kwargs,
        allow_pickle=True,
    )
def load_results(fname):
    data = np.load(fname, allow_pickle=True)
    return (
        data["beta_values"],
        data["results"].item(),
        data["ps_kwargs"].item(),
        data["run_kwargs"].item(),
    )

def run_all_plots(results): 
    plot_lowk_power_vs_beta(results)
    plot_fft_spectrum_heatmap(results)
    plot_lowk_modes_vs_beta(results)
    beta_results_dict = {}
    for beta in results:
        beta_results_dict[beta] = results[beta]['raw'][0]['out']
    plot_time_to_pattern_vs_beta(results)
    out = results[beta]['raw'][0]['out']
    rho_mean = np.mean(out['total_list'][-1])
    threshold = 1.2 * rho_mean
    plot_cluster_distribution(out, threshold)
    plot_temporal_autocorrelation(results[beta]['raw'][0]['out'])
    plot_autocorrelation_vs_beta(results)
    plot_mode_growth_time(beta_results_dict, 1)
    plot_mode_growth_time(beta_results_dict, 2)
    plot_mode_growth_time(beta_results_dict, 3)
    plot_mode_growth_time(beta_results_dict, 4)
    plot_mode_growth_time(beta_results_dict, 5)
    plot_mode_growth_time(beta_results_dict, 10)
    plot_mode_growth_time(beta_results_dict, 20)
    plot_lowk_variance_time(beta_results_dict)
    plot_dominant_mode_amplitude_vs_beta(results)
    plot_spectral_entropy_vs_beta(results)
    plot_mode_competition_vs_beta(results)
    
    growth_rate_vs_beta(results, beta_results_dict)

if __name__ == "__main__":
    RUN_SIMULATION = True   # ← switch this
    RESULTS_FILE = "beta_sweep_local_structure.npz"

    beta_values = np.linspace(0.0, 3.0, 11)   # 0 → 3 in 11 steps
    n_runs_per_beta = 3

    T = 40
    obs_dt = 1

    L = 1000
    N = 900
    if RUN_SIMULATION:
        rho0_plus, rho0_minus, _, _ = make_exp_gradient(
            L=L,
            N=N,
            frac_plus=0.75,
            decay_length=0.2,
            anchor_positions=None,
            anchor_peak_width=0.01,
            anchor_peak_mass=0.03,
        )

        ps_kwargs = dict(
            L=L,
            xlim=1,
            rate_diffusion=0.05,
            rate_active=5,
            flip_rate_fn=None,
            init='fixed',
            N=N,
            scale_rates=False,
            local_kernel_sigma= 0.005, #0.005,
            minus_anchor=True,
            periodic=False,
            immobilize_when_anchored=True,
            anchor_radius=0.003,
            anchor_positions=None,
            site_capacity=1,
            crowding_suppresses_rates=False,
            k_on=0,
            k_off=0,
            k_exit=0,
        )

        init_kwargs = dict(
            rho0_plus=rho0_plus,
            rho0_minus=rho0_minus,
        )

        run_kwargs = dict(
            T=T,
            obs_dt=obs_dt,
            record_fft=True,
            record_var=True,
        )

        print("Starting β sweep for local structure analysis...\n")
        results = sweep_betas_for_structures(
            beta_values=beta_values,
            n_runs_per_beta=n_runs_per_beta,
            ps_kwargs=ps_kwargs,
            init_kwargs=init_kwargs,
            run_kwargs=run_kwargs,
            start_fraction=0.5,
            k_max=None,
        )

        print("\nβ sweep completed.")
        save_results(
                RESULTS_FILE,
                beta_values,
                results,
                ps_kwargs,
                run_kwargs,
            )
        print("Results saved to beta_sweep_local_structure.npz")
    else:
        print("Loading existing results...")
        beta_values, results, ps_kwargs, run_kwargs = load_results(RESULTS_FILE)

    run_all_plots(results)
    print("All plots generated.")
