from IMEX_PDE_solver_class import IMEXPDE
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Sweep / averaging parameters
# -------------------------------
kernel_sigma_values = [
    0.0005,      # very local
    0.005,    
    0.05,
    0.1,
    1.0, # effectively global
]

n_runs = 5                   # number of realizations per sigma
base_seed = 100               # seed offset
outdir = Path("IMEX_kernel_sigma_sweep")
outdir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Solver parameters (fixed)
# -------------------------------
solver_params = dict(
    L=1000,
    T=40.0,
    dt=5e-4,
    gamma=0,
    lam=0.6,
    beta=0.5,
    bc="periodic",
    active_model="bidirectional",
    gaussian_kernel=True,
    snapshot_interval=50,
)

init_params = dict(
    mode="homogeneous",
    rho0=1.0,
    noise=0.3,
)

# -------------------------------
# Storage
# -------------------------------
m_results = {}
v_results = {}
D_results = {}
var_results = {}

# -------------------------------
# Main sweep
# -------------------------------
for k_idx, kernel_sigma in enumerate(kernel_sigma_values):
    print(f"Running kernel_sigma = {kernel_sigma}")

    m_runs = []
    v_runs = []
    D_runs = []
    var_runs = []

    for r in range(n_runs):
        seed = base_seed + 1000 * k_idx + r

        solver = IMEXPDE(
            **solver_params,
            kernel_sigma=kernel_sigma,
            outdir=outdir / f"sigma_{kernel_sigma}_run_{r}",
            seed=seed,
        )

        solver.initialize(**init_params)
        solver.solve()

        out = solver.get_output()
        m_runs.append(np.abs(out["m_series"]))
        v_runs.append(np.abs(out["v_eff_series"]))
        D_runs.append(out["D_eff_series"])
        var_runs.append(out["var_series"])

    m_results[kernel_sigma] = np.array(m_runs)
    v_results[kernel_sigma] = np.array(v_runs)
    D_results[kernel_sigma] = np.array(D_runs)
    var_results[kernel_sigma] = np.array(var_runs)



cmap = plt.cm.Blues
colors = cmap(np.linspace(0.4, 0.9, len(kernel_sigma_values)))
t = np.linspace(0, solver_params["T"], m_results[kernel_sigma_values[0]].shape[1])

# -------------------------------
# Plot 1: |m(t)|
# -------------------------------
plt.figure(figsize=(8, 5))

for color, kernel_sigma in zip(colors, kernel_sigma_values):
    data = m_results[kernel_sigma]
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    plt.plot(t, mean, color=color, lw=2,
             label=fr"$\sigma={kernel_sigma}$")
    plt.fill_between(t, mean - std, mean + std,
                     color=color, alpha=0.25)

plt.xlabel("$t$")
plt.ylabel(r"$|m(t)|$")
plt.legend()
plt.grid()
plt.xlim(0,10)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(outdir / "magnitude_magnetization_sweep.png", dpi=200)
plt.close()

# -------------------------------
# Plot 2: |v_eff(t)|
# -------------------------------
plt.figure(figsize=(8, 5))

for color, kernel_sigma in zip(colors, kernel_sigma_values):
    data = v_results[kernel_sigma]

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)

    plt.plot(t, mean, color=color, lw=2,
             label=fr"$\sigma={kernel_sigma}$")
    plt.fill_between(t, mean - std, mean + std,
                     color=color, alpha=0.25)

plt.xlabel("$t$")
plt.ylabel(r"$|v_{\mathrm{eff}}(t)|$")
plt.legend()
plt.xlim(0.05,10)
plt.ylim(bottom=0)
plt.grid()
plt.tight_layout()
plt.savefig(outdir / "magnitude_velocity_sweep.png", dpi=200)
plt.close()

# -------------------------------
# Plot 3: D_eff(t)
# -------------------------------
plt.figure(figsize=(8, 5))

for color, kernel_sigma in zip(colors, kernel_sigma_values):
    data = D_results[kernel_sigma]

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)

    plt.plot(t, mean, color=color, lw=2,
             label=fr"$\sigma={kernel_sigma}$")
    plt.fill_between(t, mean - std, mean + std,
                     color=color, alpha=0.25)

plt.xlabel("$t$")
plt.ylabel(r"$D_{\mathrm{eff}}(t)$")
plt.legend()
plt.xlim(0.05,10)
plt.ylim(bottom=0)
plt.grid()
plt.tight_layout()
plt.savefig(outdir / "diffusion_sweep.png", dpi=200)
plt.close()

# -------------------------------
# Plot 4: Variance Var(t)
# -------------------------------
plt.figure(figsize=(8, 5))

for color, kernel_sigma in zip(colors, kernel_sigma_values):
    data = var_results[kernel_sigma]

    mean = data.mean(axis=0)
    std = data.std(axis=0)

    plt.plot(
        t,
        mean,
        color=color,
        lw=2,
        label=fr"$\sigma={kernel_sigma}$"
    )
    plt.fill_between(
        t,
        mean - std,
        mean + std,
        color=color,
        alpha=0.25
    )

plt.xlabel("$t$")
plt.ylabel(r"$\mathrm{Var}(\rho_+ + \rho_-)$")
plt.legend()
plt.xlim(0.05, 40)
plt.ylim(bottom=0)
plt.grid()
plt.tight_layout()
plt.savefig(outdir / "variance_sweep.png", dpi=200)
plt.close()