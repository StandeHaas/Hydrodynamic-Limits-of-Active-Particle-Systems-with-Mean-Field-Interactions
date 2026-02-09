import numpy as np
import matplotlib.pyplot as plt

# ---- parameters and grids ----
p_min, p_max = 0.05, 1.0        # avoid p = 0 singularity
beta_min, beta_max = 0.0, 3.0

Np = 1000
Nbeta = 1000

p_vals = np.linspace(p_min, p_max, Np)
beta_vals = np.linspace(beta_min, beta_max, Nbeta)

P, B = np.meshgrid(p_vals, beta_vals)

# ---- solve m = tanh(beta * m) for each beta ----
def solve_m(beta, tol=1e-10, max_iter=10_000):
    if beta <= 1:
        return 0.0

    m = np.sqrt(3 * (beta - 1) / beta**3)  # mean-field asymptotic
    for _ in range(max_iter):
        m_new = np.tanh(beta * m)
        if abs(m_new - m) < tol:
            return m_new
        m = m_new
    return m


m_vals = np.array([solve_m  (beta) for beta in beta_vals])

print(m_vals)

# broadcast m(beta) to the 2D grid
M = m_vals[:, None]

# ---- compute the function ----
F = P * (
    1.39
    - 0.75 * P
    + 0.2885 / (P**(3/2) * np.cosh(B * M))
)

# ---- plot heatmap ----
vmin = 0.25
vmax = 1



im = plt.imshow(
    F,
    origin="lower",
    aspect="auto",
    extent=[p_min, p_max, beta_min, beta_max],
    vmin=vmin,
    vmax=vmax
)

plt.xlabel("p")
plt.ylabel("β")
plt.title("Heatmap of f(p, β)")
plt.tight_layout()
plt.show()
