from IMEX_PDE_solver_class import IMEXPDE
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
from pathlib import Path

solver = IMEXPDE(
    L=1000,
    T=20,
    dt=5e-4,
    gamma= 0, #0.002,
    lam=0.6,
    beta=2.0,
    bc="periodic",                 # "periodic" or "neumann"
    active_model="bidirectional", # "bidirectional" or "anchored_minus"
    gaussian_kernel=True,
    kernel_sigma= 0.005, 
    snapshot_interval=50,
    outdir="IMEX_beta_3p0",
    seed=58,
)

solver.initialize(
    mode="homogeneous",   # "homogeneous" or "poisson"
    rho0=1.0,
    noise=0.3,
)

solver.solve()

out = solver.get_output()

solver.plot_all()
solver.plot_individual()