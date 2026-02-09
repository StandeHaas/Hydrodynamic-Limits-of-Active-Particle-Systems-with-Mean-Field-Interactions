import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ---------------------------------------------------------------------
# Model definitions (as provided, with minimal correction for cosh^2)
# ---------------------------------------------------------------------

def D_eff_theory(beta, m_beta, gamma_eff, lambda_eff, m_beta_non, rho_bar, K):
    denom = 1 - (rho_bar / K) * beta / (np.cosh(beta * m_beta_non)**2)
    return (
        gamma_eff
        - lambda_eff**2
        / (2 * denom)
        * (
            (1 - 2 * rho_bar / K)
            + np.tanh(beta * m_beta)**2 / denom**2
        )
    )


def v_pred_new_theory(lambda_eff, rho_bar, K, beta, m_beta, m_beta_non):
    p_plus = 0.5 * (1.0 + m_beta)
    denom = 1 - (rho_bar / K) * beta / (np.cosh(beta * m_beta_non)**2)
    return lambda_eff * p_plus / denom


# ---------------------------------------------------------------------
# Solve m = tanh(beta m)
# ---------------------------------------------------------------------

def solve_magnetization(beta):
    """
    Returns (m_beta, m_beta_non).
    m_beta_non is always zero.
    m_beta is the stable non-zero solution when it exists.
    """

    if beta <= 1.0:
        return 0.0, 0.0

    def f(m):
        return m - np.tanh(beta * m)

    # Solve on (0,1)
    sol = root_scalar(f, bracket=(1e-6, 1.0), method="bisect")
    return sol.root, 0.0


# ---------------------------------------------------------------------
# Parameters (adjust as needed)
# ---------------------------------------------------------------------

gamma_eff = 2e-4
lambda_eff = 5
rho_bar = 0.5
K = 10

# ---------------------------------------------------------------------
# Sweep beta
# ---------------------------------------------------------------------

betas = np.linspace(0.01, 3.0, 400)

D_vals = []
v_vals = []

for beta in betas:
    m_beta, m_beta_non = solve_magnetization(beta)

    D_vals.append(
        D_eff_theory(
            beta,
            m_beta,
            gamma_eff,
            lambda_eff,
            m_beta_non,
            rho_bar,
            K,
        )
    )

    v_vals.append(
        v_pred_new_theory(
            lambda_eff,
            rho_bar,
            K,
            beta,
            m_beta,
            m_beta_non,
        )
    )

D_vals = np.array(D_vals)
v_vals = np.array(v_vals)

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

plt.figure()
plt.plot(betas, D_vals)
plt.xlabel(r"$\beta$")
plt.ylabel(r"$D_{\mathrm{eff}}$")
plt.title("Effective Diffusion vs β")
plt.grid(True)
plt.savefig("Effective Diffusion vs β.png", dpi=200)
plt.close()

plt.figure()
plt.plot(betas, v_vals)
plt.xlabel(r"$\beta$")
plt.ylabel(r"$v_{\mathrm{pred}}$")
plt.title("Predicted Velocity vs β")
plt.grid(True)
plt.savefig("Predicted Velocity vs β.png", dpi=200)

plt.close()
