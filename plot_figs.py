import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point
from pathlib import Path

v_mean = [0.003277260564318088, 0.00326957654341861, 0.002874661500544684, 0.0029131306865650426, 0.3255910531390897, 0.4976502181847137, 0.5501282101934136, 0.5731872927510997, 0.5842402961300152, 0.589955169620428, 0.5930821949560446] 
v_err  = [0.0013737205344573691, 0.0014099708192577838, 0.0014002849023047068, 0.0005684295489898486, 0.014517318855756325, 0.0028889216317880694, 0.0017948900800049553, 0.0011295300035629203, 0.001033351466781752, 0.0011485007205684417, 0.0011155116424200872]
D_mean = [0.3772724254174937, 0.3770436261943821, 0.3776404868452234, 0.37667491459412905, 0.29879118965547463, 0.23247000648670704, 0.21171186738984507, 0.2044486937013826, 0.20165671802451116, 0.20122323661234517, 0.20086236509495992]
D_err = [0.003977552919131897, 0.0038569269149759363, 0.004140344412536427, 0.003728223158336552, 0.002727070588056267, 0.0020691061731105843, 0.0009598069031208945, 0.0003209477163188354, 0.0005252364093868209, 0.0008870238934991939, 0.0007259894207740303]

sv_mean = [0.00029617, 0.00029213, 0.00030215, 0.00028813, 0.00049689, 0.00054726, 0.00058253, 0.00057532, 0.00059094, 0.00060268, 0.00060157]
sv_err =[4.11337508e-06, 5.32977821e-06, 2.97977874e-06, 1.16478802e-05,
         6.39396611e-06, 4.10176739e-06, 5.10393493e-06, 6.87895992e-06,
         1.00820658e-05, 6.70961090e-06, 8.09862307e-06]
sv_mean = np.array(sv_mean) * 1000
sv_err = np.array(sv_err) * 1000

sD_mean = [7.84463865e-07, 8.14462467e-07, 8.05610707e-07, 7.85492010e-07,
         5.18382657e-07, 4.61349624e-07, 4.40399856e-07, 4.14219563e-07,
         4.09681057e-07, 3.74726530e-07, 3.92329290e-07]
sD_err = [2.58863937e-08, 1.52041456e-08, 3.67934043e-08, 1.35796872e-08,
         1.27577659e-08, 1.62775693e-08, 1.97836021e-08, 2.05537466e-08,
         1.00972362e-08, 1.27872411e-08, 1.91510531e-08]

sD_mean = np.array(sD_mean) * 1000 * 1000 / 2
sD_err = np.array(sD_err) * 1000 * 1000 /2

beta_values = np.linspace(0, 3, 11)

beta_dense = np.linspace(0, 3, 400)

m_beta = np.array([
    fixed_point(lambda m: np.tanh(beta * m), 0.5) if beta > 0 else 0.0
    for beta in beta_dense
])

sv_mean = 2 * (sv_mean  - 0.6 / 2)

v_theory = 0.6 * np.tanh(beta_dense * m_beta)

D_theory = 0.2 + 0.6**2 / (2 * np.cosh(beta_dense * m_beta)**3)

### Plot v_eff
plt.figure(figsize=(6, 4))
plt.errorbar(
    beta_values,
    v_mean,
    yerr=v_err,
    fmt="o",
    capsize=4,
    label=r"Particle Sim",
)
plt.errorbar(
    beta_values,
    sv_mean,
    yerr=sv_err,
    fmt="o",
    capsize=4,
    label=r"PDE Sim",
    color='lightblue'
)
plt.plot(beta_dense, v_theory, "k--", color='navy', label=r"$\lambda\tanh(\beta m_\beta)$")

plt.xlabel(r"$\beta$")
plt.ylabel(r"$v_{\mathrm{eff}}$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("PDE_v_eff_vs_beta.png", dpi=200)
plt.close()

## Plot D_eff
plt.figure(figsize=(6, 4))
plt.errorbar(
    beta_values,
    D_mean,
    yerr=D_err,
    fmt="o",
    capsize=4,
    label=r"Particle Sim",
)
plt.errorbar(
    beta_values,
    sD_mean,
    yerr=sD_err,
    fmt="o",
    capsize=4,
    label=r"PDE Sim",
    color='lightblue',
)
plt.plot(
    beta_dense,
    D_theory,
    "k--",
    color='navy',
    label=r"$\gamma + \lambda^2 / (2\cosh^3(\beta m_\beta))$",
)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$D_{\mathrm{eff}}$")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("PDE_D_eff_vs_beta.png", dpi=200)
plt.close()

