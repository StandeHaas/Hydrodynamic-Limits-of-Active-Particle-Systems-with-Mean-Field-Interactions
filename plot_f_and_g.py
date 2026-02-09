import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fixed_point, curve_fit

lst_f_fit = [1.2674132914385021, 1.132521977941153, 1.2420524107838409, 1.1296458815534764, 1.1979197594043645, 1.1206747508447639, 1.0180689042684508, 1.0307328322400515, 0.9650095664861454, 0.9452818450712913, 0.9238960678087177, 0.8949960386818892, 0.8409280455152771, 0.8142731770524829, 0.7971705090490088, 0.7679387053195614, 0.7303637160033035, 0.7208187588637358, 0.7036396523323505]
lst_f_err = [0.18323615786041292, 0.06127426836109709, 0.04347979344583143, 0.03737012433281702, 0.015920866661987804, 0.025648432721351422, 0.024683910514322678, 0.016551902973365108, 0.008645300446906189, 0.01861280630225769, 0.015037471812334403, 0.013672149713545233, 0.008892119225363877, 0.00919099023881873, 0.006843599528549221, 0.008849969258122517, 0.009633666346095575, 0.006873981731875445, 0.0063602729055100655]
lst_g_fit = [4.301977590397233, 3.128546847383045, 1.9904506408846678, 1.7043821544884974, 1.2480333641415329, 1.0002918774707261, 0.9635467345124697, 0.7506281368430943, 0.6788244123004412, 0.5134422197273695, 0.4498881603836516, 0.4023543211373238, 0.3682173915422443, 0.3224593139637179, 0.27679667248782425, 0.2486795809948861, 0.22703966916050528, 0.19686079110695956, 0.18976491360090697]
lst_g_err = [0.44053609330317783, 0.08270356117050541, 0.09357566598703401, 0.06136648077945498, 0.022382647561986237, 0.032561398964612076, 0.03413130628434911, 0.024629302776538728, 0.012057275130467915, 0.019854116707707595, 0.01738299359240982, 0.014647303305680554, 0.00953581908708521, 0.011827717925468282, 0.007861918667140231, 0.010326940055011427, 0.010624747712418768, 0.007087646239968091, 0.00652166073374481]




def f_model(x, C0, C1):
    return C0 - C1 * x

list_N_part = np.linspace(50, 950, 19)

x_vals = np.asarray(list_N_part/1000)
f_vals = np.asarray(lst_f_fit)
f_errs = np.asarray(lst_f_err)

popt_f, pcov_f = curve_fit(f_model, x_vals, f_vals, sigma=f_errs, absolute_sigma=True)

C0, C1 = popt_f
print(C0,C1)

x_dense = np.linspace(x_vals.min(), x_vals.max(), 300)

plt.figure(figsize=(6, 4))

plt.errorbar(
    x_vals,
    f_vals,
    yerr=f_errs,
    fmt='o',
    capsize=3,
    label='f data',
    color='lightblue'
)

plt.plot(
    x_dense,
    f_model(x_dense, C0, C1),
    linestyle='--',
    label=r'$C_0 - C_1 (\bar\rho / k)$', 
    color = 'cadetblue'
)

######### g
def g_model(x, C2):
    return C2 / (x ** (3/2))

g_vals = np.asarray(lst_g_fit)
g_errs = np.asarray(lst_g_err)

popt_g, pcov_g = curve_fit(
    g_model,
    x_vals,
    g_vals,
    sigma=g_errs,
    absolute_sigma=True,
)

C2 = popt_g[0]
print(C2)

plt.errorbar(
    x_vals,
    g_vals,
    yerr=g_errs,
    fmt='o',
    capsize=3,
    label='g data', 
    color='blue'
)

plt.plot(
    x_dense,
    g_model(x_dense, C2),
    linestyle='--',
    label=r'$C_2 / (\bar\rho / k)^{3/2}$',
    color = 'navy'
)

plt.xlabel(r'$\bar\rho / k$')
plt.ylabel(r'$\cdot(\bar\rho / k)$')
plt.ylim(0,5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("f_and_g_fit.png", dpi=200)
plt.close()
