import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point, least_squares
from scipy.interpolate import interp1d

x_data = np.array([
    1.183091787,
    1.793960924,
    3.863849765,
    8.986725664,
    15.35755814,
    20.41836735,
    32.14380531,
    61.52985075,
    85.80882353,
    120.7938719,
    157.2586207,
    207.754386,
    280.619469,
    350.4866071,
    415.6925373,
    475.7919162,
    527.1126126,
    572.1126126,
    605.0105422,
    629.3629518,
    655.4638554
])
x_data /= 1000

y_data = np.array([
    0.285093775,
    0.285247111,
    0.285723441,
    0.286662039,
    0.287325111,
    0.286536845,
    0.284502126,
    0.278887153,
    0.273729269,
    0.26502522,
    0.255095091,
    0.24167047,
    0.222002285,
    0.201592436,
    0.179341525,
    0.156227285,
    0.133172733,
    0.109493904,
    0.091527056,
    0.078087341,
    0.063529564
])

density = [
    1.223333333,
    3.038333333,
    5.25,
    23,
    46.25,
    90.2375,
    162.7,
    316.6306667,
    488.515
]
rho_bar = np.array(density) / 1000

v_eff_data = np.array([
    0.296666667,
    0.2965,
    0.295,
    0.2638,
    0.26025,
    0.257425,
    0.245266667,
    0.23046,
    0.18305
])
v_eff_err = np.array([
    0.005773503,
    0.005049752,
    0.007071068,
    0,
    0.009742518,
    0.013424679,
    0.018945824,
    0.026744714,
    0.027365032
])
v_eff_err[v_eff_err == 0] = np.min(v_eff_err[v_eff_err > 0])

def compute_m_of_beta_non(beta_values):
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

def v_eff_fit(rho_bar, k, beta, lambda_eff):
    return lambda_eff * (1 +  np.tanh(beta * compute_m_of_beta_non([beta])[0])) / 2 * (1 - rho_bar / k * ((1.2552899764748897 - 0.6022927624714487 * rho_bar / k) + 0.15327283599951863 / ((rho_bar/ k) ** (1.5)) / np.cosh(beta * compute_m_of_beta_non([beta])[0]))
)

def residuals(params, rho_bar, data, err):
    k, beta, lambda_eff = params
    model = v_eff_fit(rho_bar, k, beta, lambda_eff)
    return (model - data) / err

x0 = np.array([1.0, 6.0, 0.29])   # k, beta, lambda
bounds = (
    [1e-6, 1.01, 0.0],            # lower bounds
    [np.inf, 50.0, 1.0]           # upper bounds
)

# -------------------------------------------------
# Least-squares fit
# -------------------------------------------------
best_cost = np.inf
best_params = None
best_k = None

for k_try in range(1, 21):  # try k = 1, 2, ..., 20
    def residuals_cont(params):
        beta, lambda_eff = params
        return (v_eff_data - v_eff_fit(rho_bar, k_try, beta, lambda_eff)) / v_eff_err

    # Initial guess for beta, lambda
    x0 = [6.0, 0.29]

    result = least_squares(
        residuals_cont,
        x0,
        bounds=([1.01, 0.0], [50.0, 1.0]),
        method='trf'
    )

    if result.cost < best_cost:
        best_cost = result.cost
        best_params = result.x
        best_k = k_try

beta_fit, lambda_fit = best_params
k_fit = best_k

print(f"k = {k_fit:.6f}, beta = {beta_fit:.6f}, lambda = {lambda_fit:.6f}")
print("Cost:", best_cost)
v_fit_at_data = v_eff_fit(rho_bar, k_fit, beta_fit, lambda_fit)

chi2 = 2 * best_cost
dof = len(v_eff_data) - len(result.x)
chi2_red = chi2 / dof

print("Chi^2:", chi2)
print("Reduced Chi^2:", chi2_red)

# Interpolate TASEP-LK data (log-log is usually safer if x spans orders of magnitude)
interp_fit = interp1d(x_data, y_data, kind='linear', fill_value='extrapolate')

# Evaluate at the same points as v_eff_data
y_fit_at_data = interp_fit(rho_bar)
# Standard χ²
chi2 = np.sum(((v_eff_data - y_fit_at_data) / v_eff_err)**2)

# Degrees of freedom
dof = len(v_eff_data) - 0  # zero fit parameters, since you are just comparing a fixed curve

# Reduced χ²
chi2_red = chi2 / dof

print("Chi^2:", chi2)
print("Reduced Chi^2:", chi2_red)


# -------------------------------------------------
# Plot
# -------------------------------------------------
rho_plot = np.logspace(np.log10(rho_bar.min()), np.log10(rho_bar.max()* 1.4), 400)
v_fit_curve = v_eff_fit(rho_plot, k_fit, beta_fit, lambda_fit)

plt.figure(figsize=(7, 5))

plt.errorbar(
    rho_bar,
    v_eff_data,
    yerr=v_eff_err,
    fmt='o',
    color='blue',
    capsize=3,
    label=r"$v_{\mathrm{eff}}$ data"
)

plt.plot(
    rho_plot,
    v_fit_curve,
    color='navy',
    linewidth=2,
    label=r"$v_{\mathrm{eff}}$ Mean-field"
)

plt.plot(
    x_data,
    y_data, 
    color='lightblue',
    label=r'$v_{\mathrm{eff}}$ TASEP-LK'
)

plt.xscale("log")
plt.xlabel(r"$\bar{\rho}$")
plt.ylabel(r"$v_{\mathrm{eff}}$")
plt.ylim(0, 0.45)
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig('KinII_fit.png', dpi=200)
plt.close()

v_fit_at_data = v_eff_fit(rho_bar, k_fit, beta_fit, lambda_fit)
residuals_your_fit = (v_eff_data - v_fit_at_data) / v_eff_err
residuals_paper_fit = (v_eff_data - y_fit_at_data) / v_eff_err


plt.figure(figsize=(5,4))
plt.axhline(0, color='k', linestyle='--')

plt.scatter(rho_bar, residuals_your_fit, color='navy', label='Mean-field residuals')
plt.scatter(rho_bar, residuals_paper_fit, color='lightblue', label='TASEP-LK residuals')

plt.xscale('log')
plt.xlabel(r'$\bar{\rho}$')
plt.ylabel('Residuals / σ')
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig('KinII_residual.png', dpi=200)
plt.close()