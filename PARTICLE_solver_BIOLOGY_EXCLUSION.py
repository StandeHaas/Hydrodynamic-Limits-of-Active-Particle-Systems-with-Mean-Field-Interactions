#Gillepsie simulation for biological system
#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Optional, Tuple, Dict, Any
from vispy import app, scene, io
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# full system
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

#### create particle system
ps = ParticleSystem(
    L = 1000,
    xlim = 1,
    rate_diffusion = 0,#0.02,
    rate_active = 5,
    beta = 0.7, #1.1
    flip_rate_fn = None,
    init = 'fixed',
    rho0_plus = make_exp_gradient(
        L = 1000,
        N = 750,
        frac_plus= 0.85,
        decay_length= 0.2,
        anchor_positions=None, #(0.25, 0.60, 0.80),
        anchor_peak_width=0.01,
        anchor_peak_mass=0.03,
    )[0],
    rho0_minus= make_exp_gradient(
        L = 1000,
        N = 750,
        frac_plus= 0.85,
        decay_length= 0.2,
        anchor_positions = None, #(0.25, 0.60, 0.80),
        anchor_peak_width=0.01,
        anchor_peak_mass=0.03,
    )[1],
    N = 750, 
    scale_rates = False,
    local_kernel_sigma = 0.002,
    minus_anchor = True,
    periodic = False,
    immobilize_when_anchored = True,
    anchor_radius= 0.003,
    anchor_positions=None, #[0.25, 0.60, 0.80],
    site_capacity = 3,
    crowding_suppresses_rates = False,
    k_on = 0, # 20,
    k_off = 0, #5,
    k_exit = 0, #30,
)
T = 20
obs_dt = 0.5
out = ps.run(T=T, obs_dt=obs_dt, record_fft=True, record_var=True)

show_k_max = 5
cmap_map = 'viridis'
save_image_path = 'PARTICLE_overview'
fps = 30
smoothing_sigma = 4
save_video_path = None #"local_evolution.mp4"
#ps.animate_profiles(out, fps=fps, smoothing_sigma=smoothing_sigma, save_path=save_video_path)
#ps.visualize_all(out, show_k_max=show_k_max, cmap_name=cmap_map, xlim=1, save_path=save_image_path, plot_fft = False)
ps.plot_individuals(out, show_k_max=show_k_max, cmap_name=cmap_map, xlim=1)