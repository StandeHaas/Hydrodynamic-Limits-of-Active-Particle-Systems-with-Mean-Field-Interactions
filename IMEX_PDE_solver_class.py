########### IMEX PDE SOLVER ############################################
#### IMPORTS ###################
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.fft import rfft, irfft
from pathlib import Path
from scipy.optimize import fixed_point

class IMEXPDE:
    #### INIT ###########################
    def __init__(
        self,
        L=1000,
        xlim=1.0,
        T=10.0,
        dt=5e-4,
        gamma=2.33e-4,
        lam=0.6,
        beta=2.0,
        bc="periodic",            # "periodic" or "neumann"
        active_model="bidirectional",  # "bidirectional" or "anchored_minus"
        gaussian_kernel=False,
        kernel_sigma=0.02,
        snapshot_interval=50,
        outdir="IMEX_output",
        seed=None,
    ):
        self.L = L
        self.xlim = xlim
        self.dx = xlim / L
        self.x = np.linspace(0, xlim, L, endpoint=False)

        self.T = T
        self.dt = dt
        self.nsteps = int(T / dt)

        self.gamma = gamma
        self.lam = lam
        self.beta = beta

        self.bc = bc
        self.active_model = active_model

        self.gaussian_kernel = gaussian_kernel
        self.kernel_sigma = kernel_sigma

        self.snapshot_interval = snapshot_interval
        self.seed = seed

        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            np.random.seed(seed)

        self.rho_mean = 1.0 / self.xlim

        self._build_diffusion_operator()
        self._build_kernel()

    #### HELPER FUNCTIONS #################
    def cw_rate(self, sigma, m):
        r = np.exp(-self.beta * sigma * m)
        return np.clip(r, 1e-8, 1e8)

    def _build_diffusion_operator(self):
        L = self.L
        dx = self.dx

        D = diags([1, -2, 1], [-1, 0, 1], shape=(L, L)).tolil()

        if self.bc == "periodic":
            D[0, -1] = D[-1, 0] = 1
        elif self.bc == "neumann":
            D[0, 1] = 2
            D[-1, -2] = 2

        self.A_diff = (
            diags(np.ones(L), 0) - self.gamma * self.dt * D / dx**2
        ).tocsr()

    def _build_kernel(self):
        if not self.gaussian_kernel:
            self.kernel_hat = None
            return

        i = np.arange(self.L)
        dist = np.minimum(i, self.L - i) * self.dx
        kernel = np.exp(-0.5 * (dist / self.kernel_sigma) ** 2)
        kernel /= kernel.sum()
        self.kernel_hat = rfft(kernel)

    #### IC ##############################
    def initialize(self, mode="poisson", rho0=1.0, noise=0.2, n_tracers=1000):
        if mode == "homogeneous":
            rho_p = rho0 + noise * np.random.randn(self.L)
            rho_m = rho0 + noise * np.random.randn(self.L)
            rho_p = np.clip(rho_p, 0, None)
            rho_m = np.clip(rho_m, 0, None)


        elif mode == "poisson":
            x = self.x
            rho_p = np.exp(-np.abs(x - 0.5)/0.05)
            rho_m = np.exp(-np.abs(x - 0.5)/0.05)

            rho_p += noise*np.random.randn(self.L)
            rho_m += noise*np.random.randn(self.L)
            rho_p = np.clip(rho_p, 0, None)
            rho_m = np.clip(rho_m, 0, None)
        else:
            raise ValueError("Unknown init mode.")

        # normalize
        tot = (rho_p + rho_m).sum()
        rho_p /= tot
        rho_m /= tot

        self.rho_p = rho_p
        self.rho_m = rho_m

        self._allocate_storage()

        # tracers
        self.n_tracers = n_tracers
        self.tracers = np.random.choice(self.L, size=n_tracers) * self.dx
        self.tracers_unwrapped = self.tracers.copy()   
        self.tracer_history = []
        self.tracer_state = np.random.choice([-1, 1], size=self.n_tracers)


    def _allocate_storage(self):
        n = self.nsteps + 1
        kmax = self.L // 2 + 1

        self.m_series = np.zeros(n)
        self.var_series = np.zeros(n)
        self.fft_amp = np.zeros((n, kmax))
        self.fft_phase = np.zeros((n, kmax), dtype=complex)

        self.snapshots = []
        self.m_snapshots = []
        self.times = []

        # effective velocity
        self.v_eff_series = np.full(n, np.nan)

        # diffusion
        self.D_eff_series = np.full(n, np.nan)

    #### CORE NUMERICS #################
    def magnetization(self):
        if self.kernel_hat is None:
            num = self.rho_p - self.rho_m
            den = self.rho_p + self.rho_m + 1e-12
            return num / den
        elif self.kernel_sigma > 100000:
            num = np.sum(self.rho_p - self.rho_m)
            den = np.sum(self.rho_p + self.rho_m)
            return num / (den + 1e-12)
        else:
            num = irfft(rfft(self.rho_p - self.rho_m) * self.kernel_hat, n=self.L)
            den = irfft(rfft(self.rho_p + self.rho_m) * self.kernel_hat, n=self.L)
            return num / (den + 1e-12)

    def advective_derivative(self, rho, direction):
            d = np.zeros_like(rho)

            if direction > 0:  # right-moving
                d[1:] = (rho[1:] - rho[:-1]) / self.dx
                if self.bc == "neumann":
                    d[0] = 0.0
                else:  # periodic
                    d[0] = (rho[0] - rho[-1]) / self.dx

            else:  # left-moving
                d[:-1] = (rho[1:] - rho[:-1]) / self.dx
                if self.bc == "neumann":
                    d[-1] = 0.0
                else:
                    d[-1] = (rho[0] - rho[-1]) / self.dx

            return d

    def step(self):
        # implicit diffusion
        rho_p = spsolve(self.A_diff, self.rho_p)
        rho_m = spsolve(self.A_diff, self.rho_m)

        if self.active_model == "bidirectional":
            # diffusion
            adv_p = -self.lam * self.advective_derivative(rho_p, +1)
            adv_m = +self.lam * self.advective_derivative(rho_m, -1)

            # reaction
            m = self.magnetization()
            R_p = self.cw_rate(-1, m) * rho_m - self.cw_rate(+1, m) * rho_p
            R_m = -R_p

            # update
            self.rho_p = np.clip(rho_p + self.dt * (adv_p + R_p), 0, None)
            self.rho_m = np.clip(rho_m + self.dt * (adv_m + R_m), 0, None)
        else: 
            # implicit diffusion
            rho_p = spsolve(self.A_diff, self.rho_p)
            rho_m = spsolve(self.A_diff, self.rho_m)

            m = self.magnetization()
            R_p = self.cw_rate(-1, m) * rho_m - self.cw_rate(+1, m) * rho_p
            R_m = -R_p

            rho_p_star = np.clip(rho_p + self.dt * R_p, 0, None)
            rho_m_star = np.clip(rho_m + self.dt * R_m, 0, None)

            # advection
            if self.active_model == "bidirectional":
                adv_p = -self.lam * self.advective_derivative(rho_p_star, +1)
                adv_m = +self.lam * self.advective_derivative(rho_m_star, -1)

            elif self.active_model == "anchored_minus":
                adv_p = -self.lam * self.advective_derivative(rho_p_star, +1)
                adv_m = 0.0

            self.rho_p = np.clip(rho_p_star + self.dt * adv_p, 0, None)
            self.rho_m = rho_m_star
            
        # renormalize
        M0 = (rho_p + rho_m).sum()
        M1 = (self.rho_p + self.rho_m).sum()
        self.rho_p *= M0 / M1
        self.rho_m *= M0 / M1

    #### Solver ####################
    def solve(self):
        # tracers
        window_time = 0.05             # physical time window
        window = int(window_time / self.dt)


        for n in range(self.nsteps + 1):
            total = self.rho_p + self.rho_m

            self.m_series[n] = np.mean(self.magnetization())
            self.var_series[n] = np.var(total)

            fft = rfft(total) / self.L
            self.fft_amp[n] = np.abs(fft)
            self.fft_phase[n] = fft

            if n % self.snapshot_interval == 0:
                self.snapshots.append(total.copy())
                self.m_snapshots.append(self.rho_p - self.rho_m)
                self.times.append(n * self.dt)
    
            # tracer magnetization
            m_field = self.magnetization()
            idx = (self.tracers / self.dx).astype(int) % self.L
            m_loc = m_field[idx]

            # CW rates with base rate kappa
            r_plus  = self.cw_rate(+1, m_loc)
            r_minus = self.cw_rate(-1, m_loc)

            # single Poisson clock
            rate = np.where(self.tracer_state == +1, r_plus, r_minus)
            flip = np.random.rand(self.n_tracers) < rate * self.dt
            self.tracer_state[flip] *= -1

            # velocity and diffusion
            v_loc = self.lam * self.tracer_state
            noise = np.sqrt(2 * self.gamma * self.dt) * np.random.randn(self.n_tracers)

            self.tracers_unwrapped += v_loc * self.dt + noise
            self.tracers = self.tracers_unwrapped % self.xlim

            self.tracer_history.append(self.tracers_unwrapped.copy())

            if len(self.tracer_history) > window:
                dr = self.tracers_unwrapped - self.tracer_history[- window]

                mean_dr = np.mean(dr)
                var_dr = np.mean((dr - mean_dr)**2)

                self.v_eff_series[n] = mean_dr / (window * self.dt)
                self.D_eff_series[n] = var_dr / (2 * window * self.dt)

            if n < self.nsteps: 
                self.step()

    #### Output ####################
    def get_output(self):
        return dict(
            rho_p=self.rho_p,
            rho_m=self.rho_m,
            m_series=self.m_series,
            var_series=self.var_series,
            fft_amp=self.fft_amp,
            fft_phase=self.fft_phase,
            snapshots=np.array(self.snapshots),
            m_snapshots=np.array(self.m_snapshots),
            times=np.array(self.times),
            v_eff_series=self.v_eff_series,
            D_eff_series=self.D_eff_series,
        )

    #### Plotting ###################
    def plot_all(self):
        t = np.linspace(0, self.T, self.nsteps + 1)

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        axs[0, 0].plot(t, self.m_series)
        axs[0, 0].set_title("Global magnetization")

        k_vals = range(1, 7)
        cmap = plt.cm.Blues
        colors = cmap(np.linspace(0.4, 0.9, len(k_vals)))

        for k, color in zip(k_vals, colors):
            axs[0, 1].plot(t, self.fft_amp[:, k], color=color, label=f"k={k}")
        axs[0, 1].legend()
        axs[0, 1].set_title("Fourier amplitudes")

        for k, color in zip(k_vals, colors):
            axs[1, 0].plot(t, np.unwrap(np.angle(self.fft_phase[:, k])), color=color, label=f"k={k}")
        axs[1, 0].set_title("Unwrapped phase")
        axs[1, 0].legend()
        axs[1, 1].plot(t, self.var_series)
        axs[1, 1].set_title("Variance")

        im0 = axs[2, 0].imshow(
            self.snapshots, aspect="auto", origin="lower",
            extent=[0, self.xlim, 0, self.times[-1]]
        )
        plt.colorbar(im0, ax=axs[2, 0])

        im1 = axs[2, 1].imshow(
            self.m_snapshots, aspect="auto", origin="lower",
            extent=[0, self.xlim, 0, self.times[-1]]
        )
        plt.colorbar(im1, ax=axs[2, 1])

        plt.savefig(self.outdir / "summary.png", dpi=200)
        plt.close(fig)
    
    def plot_individual(self, k_max=6):
        t = np.linspace(0, self.T, self.nsteps + 1)

        # Global magnetization
        plt.figure(figsize=(6, 4))
        plt.plot(t, self.m_series)
        plt.xlabel("t")
        plt.ylabel("m(t)")
        plt.grid()
        plt.savefig(self.outdir / "m_global.png", dpi=200)
        plt.close()

        # Variance
        plt.figure(figsize=(6, 4))
        plt.plot(t, self.var_series)
        plt.xlabel("t")
        plt.ylabel("Var(t)")
        plt.grid()
        plt.savefig(self.outdir / "variance.png", dpi=200)
        plt.close()

        # Fourier amplitudes
        k_vals = range(1, min(k_max + 1, self.fft_amp.shape[1]))
        cmap = plt.cm.Blues
        colors = cmap(np.linspace(0.4, 0.9, len(k_vals)))

        plt.figure(figsize=(6, 4))
        for k, color in zip(k_vals, colors):
            plt.plot(t, self.fft_amp[:, k], color=color, label=f"k={k}", alpha=0.75)
        plt.xlabel("t")
        plt.ylabel(r"$|A_k(t)|$")
        plt.xlim(0, self.times[-1])
        plt.legend()
        plt.grid()
        plt.savefig(self.outdir / "fft_amplitudes.png", dpi=200)
        plt.close()

        # Unwrapped Fourier phases
        plt.figure(figsize=(6, 4))
        for k, color in zip(k_vals, colors):
            plt.plot(t, np.unwrap(np.angle(self.fft_phase[:, k])), color=color, label=f"k={k}")
        plt.xlabel("t")
        plt.ylabel(r"unwrap Arg$(A_k)$")
        plt.legend()
        plt.grid()
        plt.savefig(self.outdir / "fft_phase_unwrapped.png", dpi=200)
        plt.close()

        # Space–time plot: total density
        plt.figure(figsize=(8, 5))
        plt.imshow(
            self.snapshots,
            aspect="auto",
            origin="lower",
            extent=[0, self.xlim, 0, self.times[-1]],
            cmap="viridis",
        )
        plt.colorbar(label=r"$\rho_+ + \rho_-$")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()
        plt.savefig(self.outdir / "spacetime_total.png", dpi=200)
        plt.close()

        # Space–time plot: magnetization
        plt.figure(figsize=(8, 5))
        plt.imshow(
            self.m_snapshots,
            aspect="auto",
            origin="lower",
            extent=[0, self.xlim, 0, self.times[-1]],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        plt.colorbar(label=r"$\rho_+ - \rho_-$")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()
        plt.savefig(self.outdir / "spacetime_magnetization.png", dpi=200)
        plt.close()

        # Effective velocity
        m_beta = fixed_point(lambda m: np.tanh(self.beta * m), 0.5)
        v_theory = self.lam * np.tanh(self.beta * m_beta)

        plt.figure(figsize=(6,4))
        plt.plot(t, self.v_eff_series, label=r"$v_{\mathrm{eff}}(t)$")
        plt.axhline(v_theory, linestyle="--", color="k",
                    label=r"$\lambda\tanh(\beta m_\beta)$")
        plt.axhline(-v_theory, linestyle="--", color="k",
            label=r"$\lambda\tanh(\beta m_\beta)$")
        plt.xlabel("t")
        plt.ylabel("velocity")
        plt.xlim(0, self.T)
        plt.ylim(-1, 1)
        plt.legend()
        plt.grid()
        plt.savefig(self.outdir / "v_eff.png", dpi=200)
        plt.close()

        # Diffusion
        D_theory = self.gamma + self.lam**2 / (2 *  (np.cosh(self.beta * m_beta)**3))

        plt.figure(figsize=(6,4))
        plt.plot(t, self.D_eff_series, label=r"$D_{\mathrm{eff}}(t)$")
        plt.axhline(D_theory, linestyle="--", color="k",
                    label=r"$\gamma + \lambda^2/(2\cosh^3(\beta m_\beta))$")
        plt.xlabel("t")
        plt.ylabel("diffusion")
        plt.xlim(0, self.T)
        plt.legend()
        plt.grid()
        plt.savefig(self.outdir / "D_eff.png", dpi=200)
        plt.close()