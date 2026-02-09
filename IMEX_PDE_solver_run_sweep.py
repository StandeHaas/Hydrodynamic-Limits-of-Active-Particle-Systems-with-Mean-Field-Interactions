from IMEX_PDE_solver_class import IMEXPDE
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
from pathlib import Path

beta_values = np.linspace(0, 3, 11)
n_runs = 3

t_min = 20.0
t_max = 40.0

v_mean = []
v_err = []

D_mean = []
D_err = []

for beta in beta_values:
    v_runs = []
    D_runs = []

    print(f"Running beta = {beta:.2f}")

    for run in range(n_runs):
        solver = IMEXPDE(
            L=1000,
            T=40.0,
            dt=5e-4,
            gamma= 0.2,
            lam=0.6,
            beta=beta,
            bc="periodic",                 # "periodic" or "neumann"
            active_model="bidirectional", # "bidirectional" or "anchored_minus"
            gaussian_kernel=True,
            kernel_sigma=1e5 - 10,
            snapshot_interval=50,
            outdir="IMEX_beta_3p0",
            seed=run,
        )

        solver.initialize(
            mode="homogeneous",   # "homogeneous" or "poisson"
            rho0=1.0,
            noise=0.3,
        )

        solver.solve()

        out = solver.get_output()
        t = np.linspace(0, 40, solver.nsteps + 1)

        mask = (t >= t_min) & (t <= t_max)

        v_eff = out["v_eff_series"][mask]
        D_eff = out["D_eff_series"][mask]

        v_runs.append(abs(np.nanmean(v_eff)))
        D_runs.append(np.nanmean(D_eff))

    v_runs = np.array(v_runs)
    D_runs = np.array(D_runs)

    v_mean.append(v_runs.mean())
    v_err.append(v_runs.std(ddof=1) / np.sqrt(n_runs))

    D_mean.append(D_runs.mean())
    D_err.append(D_runs.std(ddof=1) / np.sqrt(n_runs))

print(v_mean, v_err, D_mean, D_err)

beta_values = np.linspace(0, 3, 11)
v_mean = np.array(v_mean)
v_err = np.array(v_err)

D_mean = np.array(D_mean)
D_err = np.array(D_err)


m_beta = np.array([
    fixed_point(lambda m: np.tanh(beta * m), 0.5) if beta > 0 else 0.0
    for beta in beta_values
])

v_theory = 0.6 * np.tanh(beta_values * m_beta)

D_theory = 0.2 + 0.6**2 / (2 * np.cosh(beta_values * m_beta)**3)

### Plot v_eff
plt.figure(figsize=(6, 4))
plt.errorbar(
    beta_values,
    v_mean,
    yerr=v_err,
    fmt="o",
    capsize=4,
    label=r"simulation",
)
plt.plot(beta_values, v_theory, "k--", label=r"$\lambda\tanh(\beta m_\beta)$")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$v_{\mathrm{eff}}$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(Path("IMEX_output") / "PDE_v_eff_vs_beta.png", dpi=200)
plt.close()

## Plot D_eff
plt.figure(figsize=(6, 4))
plt.errorbar(
    beta_values,
    D_mean,
    yerr=D_err,
    fmt="o",
    capsize=4,
    label=r"simulation",
)
plt.plot(
    beta_values,
    D_theory,
    "k--",
    label=r"$\gamma + \lambda^2 / (2\cosh^3(\beta m_\beta))$",
)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$D_{\mathrm{eff}}$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(Path("IMEX_output") / "PDE_D_eff_vs_beta.png", dpi=200)
plt.close()