#Gillepsie simulation for biological system
#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Optional, Tuple, Dict, Any
from vispy import app, scene, io
from scipy.ndimage import gaussian_filter1d

# full system
class ParticleSystem:
    def __init__(
    self,
    L: int,
    xlim: float,
    rate_diffusion: float,  
    rate_active: float,
    beta: float,
    flip_rate_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,               
    init: str = 'fixed',
    N: int = 1000,
    rho0_plus: Optional[Callable[[float], float]] = None,                   
    rho0_minus:  Optional[Callable[[float], float]] = None,                 
    rng = None,
    scale_rates: bool = True,
    local_kernel_sigma: float = 0.005,
    periodic: bool = False,
    minus_anchor: bool = True,
    immobilize_when_anchored: bool = True,
    anchor_positions: list = None,
    anchor_radius: float = 0.005,
    ):   
        self.L = L
        self.xlim = xlim
        self.dx = self.xlim / self.L
        if scale_rates:
            self.rate_diffusion = rate_diffusion / (self.dx **2)
            self.rate_active = rate_active  / (self.dx)
        else: 
            self.rate_diffusion = float(rate_diffusion)
            self.rate_active = float(rate_active)
        self.beta = beta

        # define flip rates, usually just Curie-Weiss
        if flip_rate_fn is None:
            self.flip_rate_fn = lambda sigma, m: np.exp(-self.beta * sigma * m)
        else:
            self.flip_rate_fn = flip_rate_fn
        
        # initializiation mode
        assert init in ('fixed', 'poisson')
        self.init_mode = init

        if self.init_mode == 'fixed':
            self.N_fixed = N
        else: #Poisson
            self.rho0_plus = np.array([rho0_plus(i / self.L) for i in range(self.L)], dtype=float)
            self.rho0_minus = np.array([rho0_minus(i / self.L) for i in range(self.L)], dtype=float)

        # rng
        if rng is None:
            self.rng = np.random.default_rng()
        else: 
            self.rng = rng 

        self.local_kernel_sigma = local_kernel_sigma 
        self.periodic = periodic
        self.immobilize_when_anchored = immobilize_when_anchored
        self.minus_anchor = minus_anchor
        self._sigma_grid = self.local_kernel_sigma / self.dx
        self.anchor_radius = anchor_radius

        # convert anchor_positions to the lattice indexes, and build a boolean mask
        if anchor_positions is None:
            self.anchor_idxs = np.array([], dtype=int)
            self.is_anchor_site = np.zeros(self.L, dype=bool)
        else: 
            anchor_positions = np.asarray(anchor_positions, dtype=float)
            anchor_idxs = np.unique(np.round((anchor_positions / self.xlim) * (self.L - 1)).astype(int))
            self.anchor_idxs = anchor_idxs
            r_idx = int(np.ceil(anchor_radius / self.dx))
            anchor_set = set()
            for a in anchor_idxs:
                lo = max(0, a - r_idx)
                hi = min(self.L - 1, a + r_idx)
                anchor_set.update(range(lo, hi+1))
            self.anchor_idx_array = np.array(sorted(anchor_set), dtype=int)
            self.is_anchor_site = np.zeros(self.L, dtype=bool)
            self.is_anchor_site[self.anchor_idx_array] = True

        # define the kernel for easy usen (sigma == 0: global)
        # CHANGE THE NON-PERIODIC IF WE START BRAKING SYMMETR / SPATIAL VARYING INTERACTIONS/DIFFUSION
        if self.local_kernel_sigma > 0:
            xs = np.arange(self.L) * self.dx
            s = self.local_kernel_sigma
            if self.periodic:
                # this for now only works on a periodic Torus, where the center is (arbitrarly) choosen at zero
                # then the distance from 0 to any other spot is
                j = np.arange(self.L)
                dist = np.minimum(j, self.L - j) * self.dx
                kernel = np.exp(-0.5 * (dist / s) ** 2)

                #normalize and store FFT
                kernel = kernel.astype(float)
                kernel /= kernel.sum()
                self._kernel = kernel
                self._fft_kernel = np.fft.fft(self._kernel)
            else: # non-periodic with Neumann condtions
                """
                j = np.arange(self.L)
                dist_left  = j * self.dx
                dist_right = (self.L - 1 - j) * self.dx
                dist = np.minimum(dist_left, dist_right)

                kernel = np.exp(-(dist**2) / (2 * s**2))
                kernel /= kernel.sum()
                """
                # For non-periodic domains we DO NOT use FFT convolution
                self._kernel = None
                self._fft_kernel = None
        else:
            self._kernel = None
            self._fft_kernel = None
        
    #distribute the particles for the different possible start distrubutions
    def _init_fixed(self):
        N = self.N_fixed
        pos = self.rng.integers(0, self.L, size=N)
        sigma = self.rng.choice([1,-1], size=N)
        return pos.astype(np.int64), sigma.astype(np.int8)
    
    def _init_poisson(self):
        counts_p = self.rng.poisson(self.rho0_plus)   
        counts_m = self.rng.poisson(self.rho0_minus)
        total = int(counts_p.sum() + counts_m.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

        pos_list = []
        sigma_list = []
        for x in range(self.L):
            nplus = int(counts_p[x])
            if nplus > 0:
                pos_list.append(np.full(nplus, x, dtype=np.int64))
                sigma_list.append(np.ones(nplus, dtype=np.int8))
            nminus = int(counts_m[x])
            if nminus > 0:
                pos_list.append(np.full(nminus, x, dtype=np.int64))
                sigma_list.append(-np.ones(nminus, dtype=np.int8))
        pos = np.concatenate(pos_list) if pos_list else np.empty(0, dtype=np.int64)
        sigma = np.concatenate(sigma_list) if sigma_list else np.empty(0, dtype=np.int8)
        return pos, sigma
    
    def init_particles(self):
        if self.init_mode == 'fixed':
            return self._init_fixed()
        else:
            return self._init_poisson()
        
    @staticmethod
    def empirical_densities_from_particles(
        pos: np.ndarray, 
        sigma: np.ndarray, 
        L: int, 
        dx: float, 
        total_norm=None,
        ):
        counts_p = np.bincount(pos[sigma == 1], minlength=L)
        counts_m = np.bincount(pos[sigma == -1], minlength=L)
        if total_norm is None:
            Np = pos.size
            denom = float(max(1, Np)) * dx
        else:
            denom = float(total_norm) * dx
        rho_p = counts_p / denom
        rho_m = counts_m / denom
        return rho_p.astype(float), rho_m.astype(float)

    def compute_local_m_field(self, counts_p, counts_m):
        s_counts = counts_p.astype(float) - counts_m.astype(float)
        total_counts = counts_p.astype(float) + counts_m.astype(float)
        if self.local_kernel_sigma <= 0:
            m_global = float(s_counts.sum()) / float(total_counts.sum())
            return np.full(self.L, m_global, dtype=float)
        
        if self.periodic:
            fft = np.fft.fft
            ifft = np.fft.ifft
            s_conv = np.real(ifft(fft(s_counts) * self._fft_kernel))
            tot_conv = np.real(ifft(fft(total_counts) * self._fft_kernel))
        else:
            s_conv = gaussian_filter1d(
                s_counts,
                sigma=self._sigma_grid,
                mode='reflect'
            )
            tot_conv = gaussian_filter1d(
                total_counts,
                sigma=self._sigma_grid,
                mode='reflect'
            )


        m_field = np.zeros_like(s_conv)
        mask = tot_conv > 0
        m_field[mask] = s_conv[mask] / tot_conv[mask]
        # ensure values in [-1,1]
        m_field = np.clip(m_field, -1.0, 1.0)
        return m_field

    def step_gillespie(self, pos, sigma, m_field, counts_p, counts_m):
        n = sigma.size
        
        # magnetization, flip rates and      
        m_at_particles = m_field[pos]
        cvec = self.flip_rate_fn(sigma, m_at_particles)
        r_diff_vec = np.full(n, self.rate_diffusion, dtype=float)
        if self.minus_anchor == True:
            r_act_vec = np.where(sigma == 1, self.rate_active, 0.0)
        else: 
            r_act_vec = np.full(n, self.rate_active, dtype=float)

        if self.immobilize_when_anchored:
            anchored_mask = (sigma == -1) & self.is_anchor_site[pos] # so we only set the diffision to zero at these anchored locations
            r_diff_vec[anchored_mask] = 0.0
            r_act_vec[anchored_mask] = 0.0

        rates = r_diff_vec + r_act_vec + cvec
        R = float(rates.sum())

        # choose particle and its event
        tau = self.rng.exponential(1.0 / R) # waiting time
        probs = rates / R
        i = self.rng.choice(n, p=probs)

        v = self.rng.random() * rates[i]
        diff_thresh  = r_diff_vec[i]
        act_thresh = diff_thresh + r_act_vec[i]

        old_pos = pos[i]
        if v < diff_thresh:
            if self.rng.random() < 0.5:
                step = 1
            else:
                step = -1
            new_pos = int(pos[i] + step)
            if self.periodic:
                new_pos %= self.L
            else:
                if new_pos < 0:
                    new_pos = 0
                elif new_pos >= self.L:
                    new_pos = self.L - 1
            pos[i] = new_pos

            if sigma[i] == +1:
                counts_p[old_pos] -= 1
                counts_p[new_pos] += 1
            else:
                counts_m[old_pos] -= 1
                counts_m[new_pos] += 1

        elif v < act_thresh:
            new_pos = int(pos[i] + int(sigma[i])) 
            if self.periodic:
                new_pos = new_pos % self.L
            else: 
                if new_pos < 0:
                    new_pos = 0
                elif new_pos >= self.L:
                    new_pos = self.L - 1   
            pos[i] = new_pos

            if sigma[i] == +1:
                counts_p[old_pos] -= 1
                counts_p[new_pos] += 1
            else:
                counts_m[old_pos] -= 1
                counts_m[new_pos] += 1

        else:
            if sigma[i] == +1:
                sigma[i] = -1
                counts_p[old_pos] -= 1
                counts_m[old_pos] += 1
            else:
                sigma[i] = +1
                counts_m[old_pos] -= 1
                counts_p[old_pos] += 1

        return pos, sigma, tau, counts_p, counts_m
    
    def run(
        self,
        T: float = 10.0,
        obs_dt: float = 0.01,
        record_fft: bool = False,
        record_var: bool = False,
    ):
        # returns a dictonairy with keys of all the observed variables
        pos, sigma = self.init_particles()
        times_obs = np.arange(0.0, T, obs_dt)
        M = len(times_obs)

        # preallocate storage
        rho_p_list = np.zeros((M, self.L), dtype=float)
        rho_m_list = np.zeros((M, self.L), dtype=float)
        total_list = np.zeros((M, self.L), dtype=float)
        m_local_list = np.zeros((M, self.L), dtype=float)
        m_global = np.zeros(M, dtype=float)
        rho_hat_complex = np.zeros((M, self.L), dtype=complex) if record_fft else None
        fft_amp_list = np.zeros((M, self.L), dtype=float) if record_fft else None
        var_list = np.zeros(M, dtype=float) if record_var else None       

        # initial storage
        t = 0.0
        obs_idx = 0 
        counts_p = np.bincount(pos[sigma == 1], minlength=self.L)
        counts_m = np.bincount(pos[sigma == -1], minlength=self.L)

        rho_p, rho_m = self.empirical_densities_from_particles(pos, sigma, self.L, self.dx)
        rho_p_list[obs_idx, :] = rho_p
        rho_m_list[obs_idx, :] = rho_m
        total_list[obs_idx, :] = rho_p + rho_m
        m_field = self.compute_local_m_field(counts_p, counts_m)
        m_local_list[obs_idx, :] = m_field
        m_global[obs_idx] = np.mean(sigma)
        if record_fft:
            u = total_list[obs_idx, :]
            var = float(np.var(u))
            u_hat = np.fft.fft(u)
            amp = np.abs(u_hat)
            rho_hat_complex[obs_idx, :] = u_hat
            fft_amp_list[obs_idx, :] = amp
            if record_var:
                var_list[obs_idx] = var
        obs_idx += 1

        # Gillespie loop
        while t < T:
            m_field = self.compute_local_m_field(counts_p, counts_m)
            pos, sigma, tau, counts_p, counts_m = self.step_gillespie(pos, sigma, m_field, counts_p, counts_m)
            t += tau
            if t > T:
                break
            while obs_idx < M and times_obs[obs_idx] <= t:
                rho_p, rho_m = self.empirical_densities_from_particles(pos, sigma, self.L, self.dx)
                rho_p_list[obs_idx, :] = rho_p
                rho_m_list[obs_idx, :] = rho_m
                total_list[obs_idx, :] = rho_p + rho_m
                m_local_list[obs_idx, :] = m_field
                m_global[obs_idx] = np.mean(sigma)
                if record_fft:
                    u = total_list[obs_idx, :]
                    var = float(np.var(u))
                    u_hat = np.fft.fft(u)
                    amp = np.abs(u_hat)
                    rho_hat_complex[obs_idx, :] = u_hat
                    fft_amp_list[obs_idx, :] = amp
                    if record_var:
                        var_list[obs_idx] = var
                obs_idx += 1
            
            if obs_idx >= M:
                break
        
        #define output dictionary
        out = {
            'times_obs': times_obs,
            'rho_p_list': rho_p_list,
            'rho_m_list': rho_m_list,
            'total_list': total_list,
            'm_local_list': m_local_list,
            'm_global': m_global,
            'rho_hat_complex': rho_hat_complex,
            'fft_amp_list': fft_amp_list,
            'var_list': var_list,
        }
        return out
    
    ### Visualizations
    def visualize_all(
        self,
        out: dict,
        show_k_max: int = 6,
        cmap_name: str = 'viridis',
        xlim: float = 1, 
        fig_size=(10,6),        
        save_path: Optional[str] = None,
    ):
        times = out["times_obs"]
        T = times[-1]
        rho_p = out["rho_p_list"]
        rho_m = out["rho_m_list"]
        total = out["total_list"]
        m_local = out["m_local_list"]
        m_global = out["m_global"]
        rho_hat_complex = out.get("rho_hat_complex")
        fft_amp=  out.get("fft_amp_list")
        var = out.get("var_list")       
        #get colors
        colors = cm.get_cmap(cmap_name, show_k_max)

        fig, axes = plt.subplots(3, 2, figsize=fig_size, constrained_layout=True)
        ax00, ax01, ax10, ax11, ax20, ax21 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]

        # (0,0) magnetizatio
        ax00.plot(times, m_global, label="m_global")
        ax00.set_xlabel("t")
        ax00.set_ylabel("m^N(t)")
        ax00.set_xlim(0, T)
        ax00.grid(True)
        ax00.legend(loc="upper left")

        # (0,1) FFT amplitudes
        if fft_amp is not None:
            for k in range(1, min(show_k_max + 1, fft_amp.shape[1])):
                ax01.plot(times, fft_amp[:, k] / self.L, label=f"k={k}", color=colors(k - 1), alpha=0.8)
            ax01.set_xlabel("t")
            ax01.set_ylabel("|A_k(t)| / L")
            ax01.set_xlim(0, T)
            ax01.legend()
            ax01.grid(True)
        else:
            ax01.text(0.5, 0.5, "FFT not recorded", ha="center", va="center")
            ax01.axis("off")

        # (1,0) unwrapped Arg(A_k(t))
        if rho_hat_complex is not None:
            for k in range(1, min(show_k_max + 1, rho_hat_complex.shape[1])):
                ax10.plot(times, np.unwrap(np.angle(rho_hat_complex[:, k])), label=f"k={k}", color=colors(k - 1), alpha=0.8)
            ax10.set_xlabel("t")
            ax10.set_ylabel("unwrapped Arg(A_k)")
            ax10.set_xlim(0, T)
            ax10.legend()
            ax10.grid(True)
        else:
            ax10.text(0.5, 0.5, "FFT not recorded", ha="center", va="center")
            ax10.axis("off")

        # (1,1) Arg(A_k(t))
        if rho_hat_complex is not None:
            for k in range(1, min(show_k_max + 1, rho_hat_complex.shape[1])):
                ax11.plot(times, np.angle(rho_hat_complex[:, k]), label=f"k={k}", color=colors(k - 1), alpha=0.8)
            ax11.set_xlabel("t")
            ax11.set_ylabel("Arg(A_k)")
            ax11.set_xlim(0, T)
            ax11.legend()
            ax11.grid(True)
        else:
            ax11.text(0.5, 0.5, "FFT not recorded", ha="center", va="center")
            ax11.axis("off")

        # (2,0) heatmap local magnetization m_local(x,t)
        im0 = ax20.imshow(m_local, aspect="auto", origin="upper", extent=[0, xlim, times[-1], 0], cmap=cmap_name, vmin=-1, vmax=1)
        ax20.set_xlabel("x")
        ax20.set_ylabel("t")
        ax20.set_ylim([0, times[-1]])
        ax20.set_title("Local magnetization (rho_+ - rho_-) / total")
        fig.colorbar(im0, ax=ax20, label="m_local")

        # (2,1) heatmap total density
        im1 = ax21.imshow(total, aspect="auto", origin="upper", extent=[0, xlim, times[-1], 0], cmap=cmap_name, vmin=0, vmax=7)
        ax21.set_xlabel("x")
        ax21.set_ylabel("t")
        ax21.set_ylim([0, times[-1]])
        ax21.set_title("Total density")
        fig.colorbar(im1, ax=ax21, label="rho_total")

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        plt.close()

    def plot_individuals(
        self,
        out: dict,
        show_k_max: int = 6,
        cmap_name: str = 'viridis',
        xlim: float = 1, 
        fig_size=(10,6),  
    ):
        # get values
        times_obs = out["times_obs"]
        T = times_obs[-1] if times_obs.size > 0 else 0.0
        rho_p = out["rho_p_list"]
        rho_m = out["rho_m_list"]
        total = out["total_list"]
        m_local = out["m_local_list"]
        m_global = out["m_global"]
        rho_hat_complex = out.get("rho_hat_complex")
        fft_amp=  out.get("fft_amp_list")
        var = out.get("var_list")       
        #get colors
        colors = cm.get_cmap(cmap_name, show_k_max)

        # t / m^N(t)
        plt.figure(figsize=fig_size)
        plt.plot(times_obs, out["m_global"])
        plt.xlabel("t")
        plt.ylabel(r"$m^N(t)$")
        plt.xlim(0, T)
        plt.grid()
        plt.savefig('plot_t_m.png', format='png', dpi = 200)
        plt.close()

        # t / A_K(t)
        if fft_amp is not None:
            plt.figure(figsize=fig_size)
            for k in range(1, min(show_k_max + 1, fft_amp.shape[1])):
                plt.plot(times_obs, fft_amp[:, k] / self.L, label=str(k), color=colors(k - 1), alpha=0.6)
            plt.legend()
            plt.xlabel("t")
            plt.ylabel(r"$|A_k(t)|$")
            plt.xlim(0, T)
            plt.grid()
            plt.savefig('plot_t_A_K.png', format='png', dpi = 200)
            plt.close()

        # unwarppand angle Arg(A_k(t))
        if rho_hat_complex is not None:
            plt.figure(figsize=fig_size)
            for k in range(1, min(show_k_max + 1, rho_hat_complex.shape[1])):
                plt.plot(times_obs, np.unwrap(np.angle(rho_hat_complex[:, k])), label=str(k), color=colors(k - 1), alpha=0.6)
            plt.xlabel("t")
            plt.ylabel("unwrapped Arg(A_k(t))")
            plt.xlim(0, T)
            plt.legend()
            plt.grid()
            plt.savefig('plot_t_unwrap_Arg_A_K.png', format='png', dpi = 200)
            plt.close()

            plt.figure(figsize=fig_size)
            for k in range(1, min(show_k_max + 1, rho_hat_complex.shape[1])):
                plt.plot(times_obs, np.angle(rho_hat_complex[:, k]), label=str(k), color=colors(k - 1), alpha=0.6)
            plt.xlabel("t")
            plt.ylabel("Arg(A_k(t))")
            plt.xlim(0, T)
            plt.legend()
            plt.grid()
            plt.savefig('plot_t_Arg_A_K.png', format='png', dpi = 200)
            plt.close()
        
        # t / Var 
        if var is not None:
            plt.figure(figsize=fig_size)
            plt.plot(times_obs, var)
            plt.xlabel("t")
            plt.ylabel("Var(t)")
            plt.xlim(0, T)
            plt.grid()
            plt.savefig('plot_t_Var.png', format='png', dpi = 200)
            plt.close()
    
        # space time of local magnetization
        plt.figure(figsize=(10, 6))
        plt.imshow(m_local, aspect="auto", origin="upper", extent=[0, xlim, times_obs[-1], 0], cmap=cmap_name, vmin=-1, vmax=1)
        plt.colorbar(label=r"$\rho_+ - \rho_-$")
        plt.xlabel("x")
        plt.ylabel("time")
        plt.ylim(0, times_obs[-1])
        plt.tight_layout()
        plt.savefig('2D_plot_t_x_mlocal.png', format='png', dpi = 200)
        plt.close()

        # space time of local density 
        plt.figure(figsize=(10, 6))
        plt.imshow(total, aspect="auto", origin="upper", extent=[0, xlim, times_obs[-1], 0], cmap=cmap_name, vmin=0, vmax=10)
        plt.colorbar(label=r"$\rho_+ - \rho_-$")
        plt.xlabel("x")
        plt.ylabel("time")
        plt.ylim(0, times_obs[-1])
        plt.tight_layout()
        plt.savefig('2D_plot_t_x_total.png', format='png', dpi = 200)
        plt.close()

    def animate_profiles(
        self,
        out: dict,
        xlim: float = 1.0,
        fps: float = 30,
        smoothing_sigma: float = 1.0,
        save_path: str = None,
    ):
        times = out["times_obs"]
        rho_p_raw = out["rho_p_list"]
        rho_m_raw = out["rho_m_list"]
        total_raw = out["total_list"]
        m_local_raw = out["m_local_list"]    

        if smoothing_sigma == 0:
            def smooth(arr):
                return arr
        else: 
            def smooth(arr):
                return gaussian_filter1d(arr, sigma=smoothing_sigma, mode='nearest')

        rho_p = [smooth(arr) for arr in rho_p_raw]
        rho_m = [smooth(arr) for arr in rho_m_raw]
        total = [smooth(arr) for arr in total_raw]
        m_local = m_local_raw

        spread_factor = 10
        M = len(times)
        L = self.L
        x = np.linspace(0, xlim*spread_factor, L)
        xlim *= spread_factor
        canvas = scene.SceneCanvas(
                keys='interactive',
                show=True,
                bgcolor='white',      
                size=(1200, 700)
            )
        view = canvas.central_widget.add_view()

        # Pan/zoom camera with fixed limits
        cam = scene.PanZoomCamera(aspect=1)
        view.camera = cam 
        cam.set_range(x=(0, xlim), y=(-1, 3))   # visible range
        cam._limits = {
            'x': (0, xlim),
            'y': (-0.1, 1.1)
        }
        cam.zoom_factor = 1.0

        #axis
        scene.Line(np.array([[0, 0], [xlim, 0]]), color='black', width=2, parent=view.scene)  # x-axis
        scene.Line(np.array([[0, -1], [0, 3]]), color='black', width=2, parent=view.scene)    # y-axis
        for tx in np.linspace(0, xlim, 51):  # 6 ticks from 0 to 1
            scene.Line(np.array([[tx, 0], [tx, -0.05]]), color='black', width=2, parent=view.scene)
        for ty in np.linspace(-1, 3, 21):
            scene.Line(np.array([[0, ty], [-0.05, ty]]), color='black', width=2, parent=view.scene)

        line_rho_p = scene.Line(
            pos=np.column_stack([x, rho_p[0]]),
            parent=view.scene,
            color=(0, 0.6, 1, 0.6),   # cyan-ish, alpha 0.6
            width=2
        )
        line_rho_m = scene.Line(
            pos=np.column_stack([x, rho_m[0]]),
            parent=view.scene,
            color=(1, 0, 1, 0.6),     # magenta, alpha 0.6
            width=2
        )
        line_total = scene.Line(
            pos=np.column_stack([x, total[0]]),
            parent=view.scene,
            color=(1, 0.75, 0, 0.6),  # yellow/orange, alpha 0.6
            width=2
        )
        line_m_local = scene.Line(
            pos=np.column_stack([x, m_local[0]]),
            parent=view.scene,
            color=(0, 0, 0, 0.6),     # black, alpha 0.6
            width=2
        )

        # save 
        writer = None
        if save_path is not None:
            writer = io.write_movie(save_path, fps=fps)


        frame_duration = 1.0 / fps
        index = 0

        def update(ev):
            nonlocal index
            if index >= M:
                index = 0

            line_rho_p.set_data(np.column_stack([x, rho_p[index]]))
            line_rho_m.set_data(np.column_stack([x, rho_m[index]]))
            line_total.set_data(np.column_stack([x, total[index]]))
            line_m_local.set_data(np.column_stack([x, m_local[index]]))

            if writer is not None:
                frame = canvas.render()
                writer.add_frame(frame)

            index += 1

        timer = app.Timer(interval=frame_duration, connect=update, start=True)

        canvas.show()
        app.run()

        if writer is not None:
            writer.close()

ps = ParticleSystem(
    L = 500,
    xlim = 1,
    rate_diffusion = 0.0005,
    rate_active = 0.1,
    beta = 1.1,
    flip_rate_fn = None,
    init = 'fixed',
    N = 1000, 
    scale_rates = True,
    local_kernel_sigma = 0.005,
    minus_anchor = True,
    periodic = False,
    immobilize_when_anchored = True,
    anchor_radius= 0.003,
    anchor_positions=[0.25, 0.60]
)

T = 20
obs_dt = 0.01
out = ps.run(T=T, obs_dt=obs_dt, record_fft=True, record_var=True)

show_k_max = 5
cmap_map = 'viridis'
save_image_path = 'PARTICLE_overview'
fps = 10
smoothing_sigma = 2
save_video_path = None #"local_evolution.mp4"
ps.animate_profiles(out, fps=fps, smoothing_sigma=smoothing_sigma, save_path=save_video_path)
ps.visualize_all(out, show_k_max=show_k_max, cmap_name=cmap_map, xlim=1, save_path=save_image_path)
ps.plot_individuals(out, show_k_max=show_k_max, cmap_name=cmap_map, xlim=1)
