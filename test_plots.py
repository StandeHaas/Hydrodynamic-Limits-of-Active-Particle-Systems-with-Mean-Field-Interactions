import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Data
# -----------------------
x = np.array([0.05,0.1, 1/4, 1/2, 2/3, 3/4, 4/5, 0.9])

y1 = np.array([1.37, 1.35, 1.2, 1, 6/7, 0.8, 0.77, 0.72])
y2 = np.array([4, 2.9, 1.3, 1/2, 1/3, 3/11, 1/4, 0.2])

# Dense x for smooth curves
x_fine = np.linspace(0, 1, 200)

# -----------------------
# Model 1: y = 1 - C(1 - x)
# -----------------------
# Least-squares fit for C
y1_model = 1.389 - 0.75 * x_fine

# -----------------------
# Model 2: y = C / x
# -----------------------
# Least-squares fit for C
C2 = np.sum(y2 / x) / np.sum(1 / x**2)
y2_model = C2/2 / (x_fine**1.5)

# -----------------------
# Plot set 1
# -----------------------
plt.figure()
plt.scatter(x, y1)
plt.plot(x_fine, y1_model)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Set 1: Linear decay in (1 - x)")
plt.show()

# -----------------------
# Plot set 2
# -----------------------
plt.figure()
plt.scatter(x, y2)
plt.plot(x_fine, y2_model)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0,5)
plt.title("Set 2: Inverse law y = C / x")
plt.show()

print(C2)



def solve_m_fixed_point(beta, tol=1e-12, maxit=2000):
    # Solve m = tanh(beta m) for m >= 0 (we choose positive root if exists)
    m = 0.0
    for it in range(maxit):
        m_new = np.tanh(beta * m)
        if abs(m_new - m) < tol:
            return float(m_new)
        m = m_new
    return float(m)

def m_of_beta(beta_array):
    ms = []
    for b in beta_array:
        m = solve_m_fixed_point(b)
        ms.append(m)
    return np.array(ms)

def theoretical_velocity(beta_array, lam=1.0, phi=0.0):
    ms = m_of_beta(beta_array)
    p_plus = 0.5 * (1 + ms)
    return lam * p_plus * (1.0 - phi)

# Example plotting
beta_vals = np.linspace(0.0, 3.0, 301)
lam = 1.0
phi = 0.0   # set nonzero to include crowding: e.g. phi=0.2
v_th = theoretical_velocity(beta_vals, lam=lam, phi=phi)

plt.figure(figsize=(6,4))
plt.plot(beta_vals, v_th, label='mean-field prediction')
plt.axvline(1.0, color='k', ls=':', label='β_c = 1')
plt.axhline(lam/2, color='gray', ls='--', label='λ/2 baseline')
plt.axhline(lam*(1-phi), color='gray', ls=':', label='λ (1-Φ)')
plt.xlabel('β')
plt.ylabel('v_pred(β)')
plt.legend()
plt.title('Mean-field prediction: v(β) = λ (1+m(β))/2 (no crowding)')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def self_consistent_m(beta):
    if beta == 0.0:
        return 0.0

    def f(m):
        return m - np.tanh(beta * m)

    # Try to find a nonzero solution when it exists
    sol = root_scalar(
        f,
        bracket=[0.0, 1.0],
        method='bisect'
    )
    return sol.root

# Beta range
betas = np.linspace(0.0, 3.0, 400)
values = []

for beta in betas:
    m = self_consistent_m(beta)
    val = m / ((np.cosh(beta * m) - beta * np.cosh(beta * m)))
    values.append(val)

# Plot
plt.figure()
plt.plot(betas, values)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\cosh(\beta m) - \beta \cosh(\beta m)$')
plt.title(r'Self-consistent $m=\tanh(\beta m)$')
plt.grid(True)
plt.show()
