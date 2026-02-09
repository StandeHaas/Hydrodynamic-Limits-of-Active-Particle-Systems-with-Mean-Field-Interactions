6#Gillepsie simulation for biological system
#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Callable, Optional, Tuple, Dict, Any
from vispy import app, scene, io
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.optimize import fixed_point, curve_fit
from scipy.stats import nbinom, poisson

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
    site_capacity: int = 1,
    crowding_suppresses_rates: bool = False,
    k_on: float = 0.1,
    k_off: float = 0.01,     
    suppress_flip_when_bound: bool = True,
    k_exit: float = 0,
    ):   
        self.L = L
        self.xlim = xlim
        self.K = site_capacity
        self.dx = self.xlim / self.L
        if scale_rates:
            self.rate_diffusion = rate_diffusion / (self.dx **2)
            self.rate_active = rate_active  / (self.dx)
        else: 
            self.rate_diffusion = float(rate_diffusion)
            self.rate_active = float(rate_active)
        self.beta = beta
        self.k_on = k_on
        self.k_off = k_off
        self.suppress_flip_when_bound = suppress_flip_when_bound
        self.crowding_suppresses_rates = crowding_suppresses_rates
        self.k_exit = k_exit

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
            self.anchor_positions = anchor_positions
            self.anchor_idxs = np.array([], dtype=int)
            self.is_anchor_site = np.zeros(self.L, dtype=bool)
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

        if self.K == 1:
            pos = self.rng.choice(self.L, size=N, replace=False)
            sigma = self.rng.choice([1,-1], size=N)
            return pos.astype(np.int64), sigma.astype(np.int8)
        else: 
            pos = np.empty(N, dtype=np.int64)
            counts = np.zeros(self.L, dtype=int)
            # simple fill, that is, we chooise uniformly among sites with available capacity
            for i in range(N):
                avail = np.where(counts < self.K)[0]
                j = self.rng.choice(avail)
                pos[i] = j
                counts[j] += 1
            sigma = self.rng.choice([1,-1], size=N)
            return pos.astype(np.int64), sigma.astype(np.int8)

    def _init_poisson(self):
        counts_p = self.rng.poisson(self.rho0_plus)   
        counts_m = self.rng.poisson(self.rho0_minus)

        pos_list = []
        sigma_list = []
        for x in range(self.L):
            cp = int(counts_p[x])
            cm = int(counts_m[x])
            # if cp+cm > K randomly choose which are kept
            total_here = cp + cm
            if total_here == 0:
                continue
            labels = np.array([1]*cp + [-1]*cm, dtype=int)
            if total_here > self.K:
                keep_idx = self.rng.choice(total_here, size=self.K, replace=False)
                labels = labels[keep_idx]
            
            for lab in labels:
                pos_list.append(np.array([x], dtype=np.int64))
                sigma_list.append(np.array([lab], dtype=np.int8))
        
        if pos_list:
            pos = np.concatenate(pos_list)
            sigma = np.concatenate(sigma_list)
        else:
            pos = np.empty(0, dtype=np.int64)
            sigma = np.empty(0, dtype=np.int8)

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

    def _build_occupancy(self, pos, sigma):
        counts_p = np.bincount(pos[sigma == 1], minlength=self.L).astype(int)
        counts_m = np.bincount(pos[sigma == -1], minlength=self.L).astype(int)
        occ_total = counts_p + counts_m
        return occ_total, counts_p, counts_m      
 
    def step_gillespie(self, pos, sigma, bound, m_field, counts_p, counts_m, init_bin, exit_times, exit_positions, exit_init_bin, t):
        n = sigma.size
        if n == 0: # no particle
            return pos, sigma, bound, np.inf, counts_p, counts_m

        occ_total = counts_p + counts_m
        # magnetization, flip rates and      
        m_at_particles = m_field[pos]
        cvec = self.flip_rate_fn(sigma, m_at_particles)
        r_exit_vec = np.zeros(n, dtype=float)

        bound_mask = bound.copy()
        if self.suppress_flip_when_bound:
            cvec[bound_mask] = 0.0

        if self.minus_anchor:
            r_act_vec = np.where(sigma == 1, self.rate_active, 0.0).astype(float)
        else: 
            r_act_vec = np.full(n, self.rate_active, dtype=float)

        # we now check for each particle if its target sites are available (exclusion)
        # Active target (forward) depends on sigma sign
        active_step = (sigma == 1).astype(int)
        forward_targets = pos + active_step
        if self.periodic:
            forward_targets %= self.L
        else:
            forward_targets = np.clip(forward_targets, 0, self.L - 1)

        # for diffusion we have hopping to each neighbor: set an effective availability = average of neighbor availabilities.
        left_targets = pos - 1 
        right_targets = pos + 1
        if self.periodic:
            left_targets %= self.L
            right_targets %= self.L
        else:
            left_targets = np.clip(left_targets, 0, self.L - 1)
            right_targets = np.clip(right_targets, 0, self.L - 1)

        # detect if the move on a boundary and make a move that is not allowed
        same_forward = (forward_targets == pos)
        same_left = (left_targets == pos)
        same_right = (right_targets == pos)

        occ_forward = occ_total[forward_targets] - (forward_targets == pos)
        forward_free = (occ_total[forward_targets] < self.K) & (~same_forward)
        left_free = (occ_total[left_targets] < self.K) & (~same_left)
        right_free = (occ_total[right_targets] < self.K)  & (~same_right)
        
        # update the rate vectors
        r_left_vec  = self.rate_diffusion * left_free  
        r_right_vec = self.rate_diffusion * right_free 
    
        if self.immobilize_when_anchored:
            anchored_mask = (sigma == -1) & self.is_anchor_site[pos] & bound # so we only set the diffision to zero at these anchored locations
            r_act_vec[anchored_mask] = 0.0
            r_left_vec[anchored_mask] = 0.0
            r_right_vec[anchored_mask] = 0.0
            r_exit_vec[anchored_mask] = self.k_exit
            
        
        r_diff_vec = r_left_vec + r_right_vec

        active_mask = (sigma == 1)
        act_possible = active_mask & forward_free
        r_act_vec[~act_possible] = 0.0

        # crowding_suppresses_rates (1 - occ_target / K) for active and for diffusion average of left/right fractions
        if self.crowding_suppresses_rates:
            # active
            forward_free_frac = 1.0 - (occ_total[forward_targets].astype(float) / float(self.K))
            forward_free_frac = np.clip(forward_free_frac, 0.0, 1.0)
            r_act_vec *= forward_free_frac

            # diffusion
            left_frac = 1.0 - (occ_total[left_targets].astype(float) / float(self.K))
            right_frac = 1.0 - (occ_total[right_targets].astype(float) / float(self.K))
            left_frac = np.clip(left_frac, 0.0, 1.0)
            right_frac = np.clip(right_frac, 0.0, 1.0)

            r_left_vec  = self.rate_diffusion * left_free  * left_frac
            r_right_vec = self.rate_diffusion * right_free * right_frac
            r_diff_vec = r_left_vec + r_right_vec

        if self.immobilize_when_anchored: # just a backup (check)
            r_diff_vec[anchored_mask] = 0.0
            r_act_vec[anchored_mask] = 0.0

        #binding & unbinding 
        bind_eligible = (~bound_mask) & (sigma == -1) & self.is_anchor_site[pos] & (occ_total[pos] < self.K)
        r_bind_vec = np.zeros(n, dtype=float)
        r_bind_vec[bind_eligible] = self.k_on

        r_unbind_vec = np.zeros(n, dtype=float)
        r_unbind_vec[bound_mask] = self.k_off

        # calculate the total rates
        rates = r_diff_vec + r_act_vec + cvec + r_bind_vec + r_unbind_vec + r_exit_vec
        R = float(rates.sum())
        if R <= 0:
            # no allowed events; return infinite waiting time (or you could break simulation)
            return pos, sigma, bound, np.inf, counts_p, counts_m

        # choose particle and its event
        tau = self.rng.exponential(1.0 / R) # waiting time
        probs = rates / R
        i = self.rng.choice(n, p=probs)

        v = self.rng.random() * rates[i]
        diff_thresh  = r_diff_vec[i]
        act_thresh = diff_thresh + r_act_vec[i]
        bind_thresh = act_thresh + r_bind_vec[i]
        unbind_thresh = bind_thresh + r_unbind_vec[i]
        exit_thresh = unbind_thresh + r_exit_vec[i]

        old_pos = pos[i]

        if v < diff_thresh:
            rL = r_left_vec[i]
            rR = r_right_vec[i]

            if rL + rR <= 0: # no move possible 
                return pos, sigma, bound, tau, counts_p, counts_m 
            # we must pick a neighbor proportional to availability
            if self.rng.random() < rL / (rL + rR):
                new_pos = left_targets[i]
            else:
                new_pos = right_targets[i]

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
            new_pos = forward_targets[i]
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

        elif v < bind_thresh:
            bound[i] = True
        
        elif v < unbind_thresh:
            bound[i] = False

        elif v < exit_thresh:
            exit_times.append(t)
            exit_positions.append(pos[i])
            exit_init_bin.append(int(init_bin[i]))

            if sigma[i] == +1:
                counts_p[pos[i]] -= 1
            else: 
                counts_m[pos[i]] -= 1

            pos   = np.delete(pos,   i)
            sigma = np.delete(sigma, i)
            bound = np.delete(bound, i)

        else:
            if sigma[i] == +1:
                sigma[i] = -1
                counts_p[old_pos] -= 1
                counts_m[old_pos] += 1
            else:
                sigma[i] = +1
                counts_m[old_pos] -= 1
                counts_p[old_pos] += 1

        return pos, sigma, bound, tau, counts_p, counts_m, exit_times, exit_positions, exit_init_bin
    
    def run(
        self,
        T: float = 10.0,
        obs_dt: float = 0.01,
        record_fft: bool = False,
        record_var: bool = False,
    ):
        # returns a dictonairy with keys of all the observed variables
        pos, sigma = self.init_particles()
        bound = np.zeros_like(sigma, dtype=bool)

        times_obs = np.arange(0.0, T, obs_dt)
        M = len(times_obs)

        # preallocate storage
        pos_list = [None] * M
        rho_p_list = np.zeros((M, self.L), dtype=float)
        rho_m_list = np.zeros((M, self.L), dtype=float)
        total_list = np.zeros((M, self.L), dtype=float)
        particle_count_list = [None] * M
        bound_list = [None] * M
        m_local_list = np.zeros((M, self.L), dtype=float)
        m_global = np.zeros(M, dtype=float)
        rho_hat_complex = np.zeros((M, self.L), dtype=complex) if record_fft else None
        fft_amp_list = np.zeros((M, self.L), dtype=float) if record_fft else None
        var_list = np.zeros(M, dtype=float) if record_var else None     

        exit_times = []
        exit_positions = []
        exit_init_bin = []
        
        init_bin = np.floor(pos / self.L * self.L).astype(int) 

        # initial storage
        t = 0.0
        obs_idx = 0 
        counts_p = np.bincount(pos[sigma == 1], minlength=self.L)
        counts_m = np.bincount(pos[sigma == -1], minlength=self.L)
        
        pos_list[obs_idx] = pos.copy()
        rho_p, rho_m = self.empirical_densities_from_particles(pos, sigma, self.L, self.dx)
        rho_p_list[obs_idx, :] = rho_p
        rho_m_list[obs_idx, :] = rho_m
        total_list[obs_idx, :] = rho_p + rho_m
        particle_count_list[obs_idx] = pos.size
        bound_list[obs_idx] = bound.copy()
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
            pos, sigma, bound, tau, counts_p, counts_m, exit_times, exit_positions, exit_init_bin = self.step_gillespie(pos, sigma, bound, m_field, counts_p, counts_m, init_bin, exit_times, exit_positions, exit_init_bin, t)
            t += tau
            if t > T:
                break
            while obs_idx < M and times_obs[obs_idx] <= t:
                pos_list[obs_idx] = pos.copy()
                rho_p, rho_m = self.empirical_densities_from_particles(pos, sigma, self.L, self.dx)
                rho_p_list[obs_idx, :] = rho_p
                rho_m_list[obs_idx, :] = rho_m
                total_list[obs_idx, :] = rho_p + rho_m
                particle_count_list[obs_idx] = pos.size
                bound_list[obs_idx] = bound.copy()
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
            'pos_list': pos_list,
            'rho_p_list': rho_p_list,
            'rho_m_list': rho_m_list,
            'total_list': total_list,
            'particle_count_list': particle_count_list,
            'bound_list': bound_list,
            'm_local_list': m_local_list,
            'm_global': m_global,
            'rho_hat_complex': rho_hat_complex,
            'fft_amp_list': fft_amp_list,
            'var_list': var_list,
            'exit_times': exit_times,
            'exit_positions': exit_positions,
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
        plot_fft: bool = True, 
    ):
        times = out["times_obs"]
        T = times[-1]
        rho_p = out["rho_p_list"]
        rho_m = out["rho_m_list"]
        total = out["total_list"]
        count = out['particle_count_list']
        bounds = out["bound_list"]
        m_local = out["m_local_list"]
        m_global = out["m_global"]
        rho_hat_complex = out.get("rho_hat_complex")
        fft_amp=  out.get("fft_amp_list")
        var = out.get("var_list")       
        #get colors
        colors = cm.get_cmap(cmap_name, show_k_max)

        fig, axes = plt.subplots(3, 2, figsize=fig_size, constrained_layout=True)
        ax00, ax01, ax10, ax11, ax20, ax21 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]

        # (0,0) magnetization
        ax00.plot(times, m_global, label="m_global")
        ax00.set_xlabel("t")
        ax00.set_ylabel("m^N(t)")
        ax00.set_xlim(0, T)
        ax00.grid(True)
        ax00.legend(loc="upper left")

        # (0,1) FFT amplitudes
        if plot_fft:
            for k in range(1, min(show_k_max + 1, fft_amp.shape[1])):
                ax01.plot(times, fft_amp[:, k] / self.L, label=f"k={k}", color=colors(k - 1), alpha=0.8)
            ax01.set_xlabel("t")
            ax01.set_ylabel("|A_k(t)| / L")
            ax01.set_xlim(0, T)
            ax01.grid(True)
        else:
            exits = count[0] - np.array(count, dtype = float)
            ax01.plot(times, exits)
            ax01.set_xlabel("t")
            ax01.set_ylabel("# of exits")
            ax01.set_xlim(0, T)
            ax01.grid(True)

        # (1,0) unwrapped Arg(A_k(t))
        if plot_fft:
            for k in range(1, min(show_k_max + 1, rho_hat_complex.shape[1])):
                ax10.plot(times, np.unwrap(np.angle(rho_hat_complex[:, k])), label=f"k={k}", color=colors(k - 1), alpha=0.8)
            ax10.set_xlabel("t")
            ax10.set_ylabel("unwrapped Arg(A_k)")
            ax10.set_xlim(0, T)
            ax10.legend()
            ax10.grid(True)
        else:
            boundss = np.array([b.sum() for b in bounds])
            ax10.plot(times, boundss)
            ax10.set_xlabel("t")
            ax10.set_ylabel("# of bounds")
            ax10.set_xlim(0, T)
            ax10.grid(True)

        # (1,1) Arg(A_k(t))
        if plot_fft:
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
        bounds = out["bound_list"]
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

        # Histogram of cluster sizes
        total_final = total[-1]

        def get_cluster_sizes(arr):
            sizes = []
            current = 0
            for v in arr:
                if v > 1e-12:   # non-empty site
                    current += 1
                else:
                    if current > 0:
                        sizes.append(current)
                        current = 0
            if current > 0:
                sizes.append(current)
            return sizes

        cluster_sizes = get_cluster_sizes(total_final)

        plt.figure(figsize=fig_size)
        plt.hist(cluster_sizes, bins=6, edgecolor='black')
        plt.xlabel("Cluster size")
        plt.ylabel("Frequency")
        plt.title("Histogram of cluster sizes (final)")
        plt.grid()
        plt.savefig("cluster_size_histogram.png", dpi=200)
        plt.close()

        # Distributioon of bound-state lifetimes
        lifetimes = []
        active_bounds = {}  # particle_id → time_entered

        for ti in range(len(times_obs)):
            bound = bounds[ti]
            t = times_obs[ti]

            n = len(bound)
            remove_ids = [pid for pid in active_bounds.keys() if pid >= n]
            for pid in remove_ids:
                del active_bounds[pid]

            for pid in range(n):
                if bound[pid]:  # currently bound
                    if pid not in active_bounds:
                        active_bounds[pid] = t   # record entry time
                else:  # currently unbound
                    if pid in active_bounds:
                        lifetimes.append(t - active_bounds[pid])
                        del active_bounds[pid]

        # Plot the distribution
        if len(lifetimes) > 0:
            plt.figure(figsize=fig_size)
            plt.hist(lifetimes, bins=40, edgecolor='black')
            plt.xlabel("Bound-state lifetime")
            plt.ylabel("Count")
            plt.title("Distribution of bound-state lifetimes")
            plt.grid()
            plt.savefig("lifetime_distribution.png", dpi=200)
            plt.close()

        # flux profile
        flux = (np.diff(rho_p, axis=1) + np.diff(rho_m, axis=1))

        plt.figure(figsize=(10,6))
        plt.imshow(
            flux,
            aspect="auto",
            origin="upper",
            extent=[0, xlim, times_obs[-1], 0],
            cmap="viridis",
            vmin=-3.5, vmax=3.5,

        )
        plt.colorbar(label="Flux (Δρ_p + Δρ_m)")
        plt.xlabel("x")
        plt.ylabel("time")
        plt.title("Flux profile over space-time")
        plt.savefig("flux_profile.png", dpi=200)
        plt.close()

        # 1. Survival curve & flux-based FPT
        N_t = np.array(out['particle_count_list'], dtype=float)
        N0 = N_t[0]
        S = N_t / N0
        flux = -np.gradient(N_t, times_obs)
        flux = np.clip(flux, 0, None)
        fpt_pdf = flux / N0
        total_exited = N0 - N_t[-1]
        fpt_pdf_cond = flux / total_exited if total_exited > 0 else fpt_pdf*0.0

        # plot survival curve
        plt.figure(figsize=fig_size)
        plt.plot(times_obs, S, label='Survival fraction S(t)')
        plt.xlabel('t')
        plt.ylabel('S(t)')
        plt.title('Survival curve')
        plt.grid(True)
        plt.savefig('FPT_survival_curve.png', dpi=200)
        plt.close()

        # plot flux-based FPT PDF
        plt.figure(figsize=fig_size)
        plt.plot(times_obs, fpt_pdf_cond, label='Flux-based FPT PDF')
        plt.xlabel('t')
        plt.ylabel('f(t)')
        plt.title('First-passage time PDF (flux)')
        plt.grid(True)
        plt.savefig('FPT_flux_based.png', dpi=200)
        plt.close()

        # 2. Histogram from exit events
        exit_times = out.get('exit_times', [])
        if exit_times:
            plt.figure(figsize=fig_size)
            plt.hist(exit_times, bins=80, density=True, alpha=0.8, edgecolor='k')
            plt.xlabel('Exit time')
            plt.ylabel('PDF')
            plt.title('First-passage times (event list)')
            plt.grid(True)
            plt.savefig('FPT_event_histogram.png', dpi=200)
            plt.close()

        # 3. Exit position distribution
        exit_pos = out.get('exit_positions', [])
        if exit_pos:
            plt.figure(figsize=fig_size)
            plt.hist(np.array(exit_pos)/self.L, bins=50, alpha=0.8, edgecolor='k')
            plt.xlabel('Exit position (normalized x)')
            plt.ylabel('Count')
            plt.title('Exit-position distribution')
            plt.grid(True)
            plt.savefig('Exit_position_histogram.png', dpi=200)
            plt.close()

        # 4. Effective drift velocity (center-of-mass)
        total_density = out['total_list']
        x_grid = np.linspace(0, 1.0, self.L)
        mean_x = (total_density * x_grid).sum(axis=1) / (total_density.sum(axis=1)+1e-12)
        v_eff = np.gradient(mean_x, times_obs)
        mean_v_eff = np.mean(v_eff[int(len(v_eff)*0.6):])
        plt.figure(figsize=fig_size)
        plt.plot(times_obs, v_eff)
        plt.xlabel('t')
        plt.ylabel('v_eff(t)')
        plt.xlim(0, times_obs[-1])
        plt.title('Effective drift velocity (COM method)')
        plt.grid(True)
        plt.savefig('Effective_drift_velocity.png', dpi=200)
        plt.close()
  
        # cumulative exits by anchor
        if self.anchor_positions is not None:
            times = out['times_obs']
            exit_t = np.array(out['exit_times'])
            exit_x = np.array(out['exit_positions'])

            # Build site → anchor_id lookup, fully using anchor_radius
            site_to_gid = np.full(self.L, -1, dtype=int)
            centers = np.array(self.anchor_idxs, dtype=int)
            for s in np.array(self.anchor_idx_array, dtype=int):
                nearest = int(np.argmin(np.abs(centers - s)))
                site_to_gid[s] = nearest


            # Map exit positions to anchor group id
            gid = np.array([site_to_gid[x] if 0 <= x < self.L else -1 for x in exit_x])

            # Bin exit times using observation times
            if len(times) > 1:
                dt = times[1] - times[0]
            else:
                dt = 1.0
            edges = np.concatenate([times, [times[-1] + dt]])
            centers_t = edges[:-1] + 0.5 * np.diff(edges)

            nA = len(self.anchor_idxs)
            counts = np.zeros((len(centers_t), nA), dtype=int)

            # Fill histogram for each anchor
            for t, g in zip(exit_t, gid):
                if g >= 0:
                    b = np.searchsorted(edges, t, side="right") - 1
                    if 0 <= b < len(centers_t):
                        counts[b, g] += 1

            # Cumulative counts
            cumA = np.cumsum(counts, axis=0)
            total = cumA.sum(axis=1)

            # ---- Plot ----
            plt.figure(figsize=(7,4))
            blues = plt.get_cmap("Blues")
            cols = [blues(0.55), blues(0.65), blues(0.75), blues(0.9)]

            # anchor lines
            for a in range(nA):
                plt.plot(centers_t, cumA[:,a], color=cols[a % 4], lw=2, label=f"anchor {a}")

            # total line
            plt.plot(centers_t, total, color=cols[3], lw=2, ls="--", label="total exits")

            plt.xlabel("t")
            plt.ylabel("Cumulative exits")
            plt.title("Cumulative exits per anchor")
            plt.xlim(0, times_obs[-1])
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig("cumulative_exits_by_anchor.png", dpi=200)
            plt.close()

        return mean_v_eff

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
