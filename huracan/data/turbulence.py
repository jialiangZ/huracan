"""Idealized atmospheric dynamics for OSSE-style assimilation experiments.

Nature run = forced-dissipative 2D turbulence on the rotating sphere, i.e.
the barotropic vorticity equation (2D Navier-Stokes in vorticity form) with
the planetary beta term:

    dz/dt = -J(psi, zeta) + (2*Omega/a^2) dpsi/dlam + F - r*zeta - nu*lap^2(zeta)
    zeta = lap(psi)

plus per-layer advected tracers: a temperature-like scalar relaxed to a
meridional profile and a moisture-like scalar with a tropical source.  The
inverse energy cascade of 2D turbulence + beta organize the flow into jets
and Rossby waves -- the canonical idealization of the large-scale atmosphere
and a genuinely chaotic system: unlike a solid-body-advection toy
(extrapolable from a single frame), the trajectory here can only be
recovered by *fusing* sparse noisy observations
with the learned dynamics, which is exactly what an assimilation system
must demonstrate.

Spectral method: real spherical harmonics (``core.sht.build_real_basis``)
on the repo's exact Gauss grid, at a working truncation ``L <= lmax // 2``
chosen so that quadratic products are *alias-free*: products of two fields
band-limited at L contain only degrees <= 2L, which the grid quadrature
still integrates exactly (Gauss latitude nodes are exact to degree
2*nlat-1 >= 2*lmax, the longitude sum is exact for m <= nlon-1).  Time
stepping: RK4.  Output layout (files, normalizers, units of each record)
follows the standard dataset contract of ``data/dataset.py`` so every
downstream stage (training phases, EDA cycling, CRPS/ESSR evaluation)
runs unchanged.

Recorded physical units per channel family:
    z  geopotential [dam]  (z = z_base(level) + alpha * f * psi, geostrophic)
    t  temperature [K]     (advected tracer + meridional relaxation)
    u,v wind [m/s]         (from the streamfunction, exactly non-divergent)
    q  specific humidity-like [g/kg] (advected, tropical source, clipped >= 0)
    msl [hPa], 2t/2d [K], 10u/10v [m/s], tcc [0,1], tcwv [kg/m^2]
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from ..config.schema import HuracanConfig, TurbCfg
from ..core.grid import grid_from_cfg, insolation
from ..core.sht import build_real_basis
from .norm import Normalizer, obs_norm_mapping

EARTH_RADIUS = 6.371e6  # m
EARTH_OMEGA = 7.292e-5  # s^-1
GRAVITY = 9.81  # m s^-2
GAS_CONSTANT = 287.0  # J kg^-1 K^-1
FRAME_SECONDS = 6 * 3600  # one recorded frame = 6 h


# ---------------------------------------------------------------------------
# spectral machinery
# ---------------------------------------------------------------------------


class SpectralSphere:
    """Real spherical-harmonic ops at an alias-free working truncation L.

    B    (npix, n_real)  basis, columns orthonormal under int g g' dOmega
    Fwd  (n_real, npix)  exact quadrature forward transform
    Bt   (npix, n_real)  d/dtheta basis (4th-order FD of the analytic basis)
    Bl   (npix, n_real)  d/dlambda basis (exact cos/sin column exchange)

    Coefficient arrays are (n_real, C) with C batch channels (layers or
    tracers); grid arrays are (npix, C).
    """

    def __init__(self, grid, lmax_dyn: int | None = None):
        lmax_grid = min(grid.nlat - 1, (grid.nlon - 1) // 2)
        L = lmax_dyn if lmax_dyn is not None else lmax_grid // 2
        if not (0 < 2 * L <= lmax_grid):
            raise ValueError(
                f"need 0 < 2*lmax_dyn <= min(nlat-1, (nlon-1)//2) for "
                f"alias-free products; got lmax_dyn={L}, grid cap {lmax_grid}"
            )
        self.L = L
        self.a = EARTH_RADIUS
        colat, lon = grid.colat, grid.lon_flat
        B, (lidx, midx, kind) = build_real_basis(L, colat, lon)
        w = np.repeat(grid.w_lat, grid.nlon) * (2 * np.pi / grid.nlon)
        self.B = B
        self.Fwd = (w[:, None] * B).T  # exact quadrature, not pinv
        self.lidx, self.midx, self.kind = lidx, midx, kind

        # d/dtheta of the basis: 4th-order central FD of the analytic
        # functions (smooth trig polynomials; error ~1e-11 relative).
        h = 1e-5
        Bp2 = build_real_basis(L, colat + 2 * h, lon)[0]
        Bp1 = build_real_basis(L, colat + h, lon)[0]
        Bm1 = build_real_basis(L, colat - h, lon)[0]
        Bm2 = build_real_basis(L, colat - 2 * h, lon)[0]
        self.Bt = (-Bp2 + 8.0 * Bp1 - 8.0 * Bm1 + Bm2) / (12.0 * h)

        # d/dlambda of the basis: exact, d cos(m lon) = -m sin, d sin = m cos
        cos_col: dict[tuple[int, int], int] = {}
        sin_col: dict[tuple[int, int], int] = {}
        for i in range(len(lidx)):
            key = (int(lidx[i]), int(midx[i]))
            if kind[i] == 1:
                cos_col[key] = i
            elif kind[i] == 2:
                sin_col[key] = i
        Bl = np.zeros_like(B)
        for i in range(len(lidx)):
            key = (int(lidx[i]), int(midx[i]))
            if kind[i] == 1:
                Bl[:, sin_col[key]] += -midx[i] * B[:, i]
            elif kind[i] == 2:
                Bl[:, cos_col[key]] += midx[i] * B[:, i]
        self.Bl = Bl
        # paired column indices for the beta operator
        self._pair_c = np.array([cos_col[k] for k in sorted(cos_col)])
        self._pair_s = np.array([sin_col[k] for k in sorted(cos_col)])
        self._pair_m = np.array([k[1] for k in sorted(cos_col)], dtype=np.float64)

        self.sin_t = np.sin(colat)  # (npix,)
        self.cos_t = np.cos(colat)
        self.ell = lidx.astype(np.float64)
        self._lap = -self.ell * (self.ell + 1.0)  # per column, 1/a^2 at call

    # -- transforms ------------------------------------------------------
    def synth(self, u):
        """coeffs (n_real, C) -> grid (npix, C)."""
        return self.B @ u

    def analysis(self, x):
        """grid (npix, C) -> coeffs (n_real, C), exact quadrature."""
        return self.Fwd @ x

    # -- spectral operators ------------------------------------------------
    def laplacian(self, u):
        return u * (self._lap / self.a**2)[:, None]

    def inverse_laplacian(self, u):
        out = np.zeros_like(u)
        nz = self.ell > 0
        fac = (self._lap / self.a**2)[nz]
        out[nz] = u[nz] / fac[:, None]
        return out

    def jacobian(self, upsi, uq):
        """J(psi, q) = (1/(a^2 sin t)) (dlam psi * dtheta q - dtheta psi
        * dlam q), coefficient space in and out (alias-free)."""
        psi_l = self.Bl @ upsi
        psi_t = self.Bt @ upsi
        q_l = self.Bl @ uq
        q_t = self.Bt @ uq
        jac = (psi_l * q_t - psi_t * q_l) / (self.a**2 * self.sin_t)[:, None]
        return self.Fwd @ jac

    def beta_op(self, upsi):
        """(2*Omega/a^2) dlam(psi): exact cos/sin coefficient exchange."""
        out = np.zeros_like(upsi)
        m = self._pair_m[:, None]
        out[self._pair_c] += m * upsi[self._pair_s]
        out[self._pair_s] -= m * upsi[self._pair_c]
        return (2.0 * EARTH_OMEGA / self.a**2) * out

    def wind(self, upsi):
        """Non-divergent wind (u_east, v_north) on the grid from psi coeffs."""
        u_e = -(1.0 / self.a) * (self.Bt @ upsi)
        v_n = -(1.0 / (self.a * self.sin_t))[:, None] * (self.Bl @ upsi)
        return u_e, v_n

    # -- diagnostics ---------------------------------------------------------
    def energy(self, upsi):
        """Total kinetic energy sum over channels (normalized units)."""
        return 0.5 * np.sum((self.ell * (self.ell + 1.0))[:, None] * upsi**2, axis=0)


# ---------------------------------------------------------------------------
# nature run dynamics
# ---------------------------------------------------------------------------


class TurbulentAtmosphere:
    """Multi-layer forced-dissipative barotropic spheres + advected tracers.

    Layers are independent barotropic realizations (different stirring
    realizations, jet amplitudes increasing aloft) coupled only through an
    optional linear relaxation toward the layer mean.  Each layer carries
    a temperature-like tracer (relaxed to T_eq(lat)) and a moisture-like
    tracer (tropical source, linear sink), both advected by that layer's
    flow -- so z/t/u/v/q channels are dynamically consistent and observing
    one channel constrains the others, as in real DA.

    The vorticity forcing is stochastic (Markovian stirring, Scott &
    Polvani-style): an OU process in coefficient space with correlation
    time ``forcing_tau_days``.  A deterministic fixed pattern locks the
    low-resolution system onto a laminar steady state; stochastic forcing
    keeps the flow turbulent and puts an irreducible source of forecast
    uncertainty into the nature run, which is exactly what an
    assimilation experiment needs (obs are then *necessary*, not
    optional, for tracking the trajectory)."""

    def __init__(self, grid, tcfg: TurbCfg, rng):
        self.ops = SpectralSphere(grid, tcfg.lmax_dyn)
        s = self.ops
        self.rng = rng
        n = int(tcfg.n_layers)
        self.n_layers = n

        # per-layer target winds: jets stronger aloft
        ratios = np.linspace(0.55, 1.15, n)
        winds = tcfg.wind_rms * ratios
        self.wind_targets = winds
        # stationary OU std from the equilibrium energy balance estimate;
        # refined by the warm-up controller (calibrate_forcing)
        l_ref = 0.5 * (tcfg.forcing_lmin + tcfg.forcing_lmax + 1)
        drag = 1.0 / (tcfg.drag_days * 86400.0)
        psi_rms = winds * s.a / l_ref
        self.sigma = 2.5 * drag * (winds**2 / 2.0) / psi_rms
        self.drag = drag
        # nu such that nu * (L(L+1)/a^2)^2 = 1/hyper_days at the truncation
        self.hyp = (
            1.0 / (tcfg.hyper_days * 86400.0) * (s.a**2 / (s.L * (s.L + 1.0))) ** 2
        )
        self.tau_c = (
            1.0 / (tcfg.coupling_days * 86400.0) if tcfg.coupling_days > 0 else 0.0
        )

        # stochastic stirring: band-limited OU process in coefficient space
        self.band = (s.lidx >= tcfg.forcing_lmin) & (s.lidx <= tcfg.forcing_lmax)
        if tcfg.forcing_tau_days > 0:
            self.rho = np.exp(-FRAME_SECONDS / (tcfg.forcing_tau_days * 86400.0))
        else:
            self.rho = 1.0  # frozen pattern (deterministic forcing)
        self.forcing = self._draw_forcing()

        # temperature-like tracer: relaxation to T_eq(lat) per layer
        lat = np.pi / 2 - grid.colat
        self.t_eq = np.concatenate(
            [
                s.analysis(((288.0 - 22.0 * ell) - 45.0 * np.sin(lat) ** 2)[:, None])
                for ell in range(n)
            ],
            axis=1,
        )  # (n_real, n)
        self.tau_T = 1.0 / (tcfg.temp_tau_days * 86400.0)
        # moisture-like tracer: tropical source, linear sink
        self.tau_q = 1.0 / (tcfg.tracer_tau_days * 86400.0)
        src = 12.0 * np.exp(-((lat / np.deg2rad(25.0)) ** 2))[:, None]
        self.q_src = np.repeat(s.analysis(src), n, axis=1) * self.tau_q

        # initial state: random band-limited vorticity near equilibrium scale
        zeta_rms = winds * l_ref / s.a
        self.zeta = self._band_limited(
            rng, tcfg.forcing_lmin, tcfg.forcing_lmax + 2, zeta_rms
        )
        self.ttr = self.t_eq.copy()
        self.qtr = np.zeros_like(self.t_eq)

    def _band_limited(self, rng, lmin, lmax, rms):
        """Random coefficients band-limited to l in [lmin, lmax] with field
        RMS (over the sphere) equal to ``rms`` per channel."""
        s = self.ops
        u = rng.standard_normal((s.B.shape[1], self.n_layers))
        band = (s.lidx >= lmin) & (s.lidx <= lmax)
        u[~band] = 0.0
        u *= (
            np.asarray(rms)[None, :]
            * np.sqrt(4.0 * np.pi / np.maximum(np.sum(u**2, axis=0), 1e-30))[None, :]
        )
        return u

    def _draw_forcing(self):
        """Fresh band-limited draw with per-layer RMS = sigma."""
        s = self.ops
        u = self.rng.standard_normal((s.B.shape[1], self.n_layers))
        u[~self.band] = 0.0
        u *= (
            self.sigma[None, :]
            * np.sqrt(4.0 * np.pi / np.maximum(np.sum(u**2, axis=0), 1e-30))[None, :]
        )
        return u

    # -- rhs and stepping ------------------------------------------------
    def rhs(self, zeta, ttr, qtr):
        s = self.ops
        psi = s.inverse_laplacian(zeta)
        dz = (
            -s.jacobian(psi, zeta)
            + s.beta_op(psi)
            + self.forcing
            - self.drag * zeta
            - self.hyp * s.laplacian(s.laplacian(zeta))
        )
        if self.tau_c > 0:
            dz = dz - self.tau_c * (zeta - zeta.mean(axis=1, keepdims=True))
        dt_ = -s.jacobian(psi, ttr) + self.tau_T * (self.t_eq - ttr)
        dq = -s.jacobian(psi, qtr) + self.q_src - self.tau_q * qtr
        return dz, dt_, dq

    def step(self, dt):
        z, t, q = self.zeta, self.ttr, self.qtr
        k1z, k1t, k1q = self.rhs(z, t, q)
        k2z, k2t, k2q = self.rhs(
            z + 0.5 * dt * k1z, t + 0.5 * dt * k1t, q + 0.5 * dt * k1q
        )
        k3z, k3t, k3q = self.rhs(
            z + 0.5 * dt * k2z, t + 0.5 * dt * k2t, q + 0.5 * dt * k2q
        )
        k4z, k4t, k4q = self.rhs(z + dt * k3z, t + dt * k3t, q + dt * k3q)
        self.zeta = z + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
        self.ttr = t + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        self.qtr = q + dt / 6.0 * (k1q + 2 * k2q + 2 * k3q + k4q)

    def psi(self):
        return self.ops.inverse_laplacian(self.zeta)

    def advance_frame(self, dt_seconds: float):
        n = int(round(FRAME_SECONDS / dt_seconds))
        if abs(n * dt_seconds - FRAME_SECONDS) > 1e-6:
            raise ValueError(
                f"frame {FRAME_SECONDS}s not divisible by dt {dt_seconds}s"
            )
        for _ in range(n):
            self.step(dt_seconds)
        # OU refresh of the stochastic stirring at frame boundaries
        if self.rho < 1.0:
            self.forcing = (
                self.rho * self.forcing
                + np.sqrt(1.0 - self.rho**2) * self._draw_forcing()
            )

    def calibrate_forcing(self, wind_rms_targets):
        """Adapt the stirring amplitude toward target rms winds.

        Called during warm-up only; sigma is frozen afterwards so the nature
        run stays a fixed stochastic process.  At equilibrium E ~ sigma^2
        (input ~ sigma * psi_rms balances 2 r E), so a bounded proportional
        controller on U converges quickly."""
        psi = self.ops.inverse_laplacian(self.zeta)
        u_e, v_n = self.ops.wind(psi)
        realized = np.sqrt((u_e**2 + v_n**2).mean(axis=0))  # (n,)
        target = np.asarray(wind_rms_targets, dtype=np.float64)
        ratio = np.clip(target / np.maximum(realized, 1e-3), 0.85, 1.18)
        self.sigma = self.sigma * ratio
        self.forcing = self.forcing * ratio[None, :]
        return realized


# ---------------------------------------------------------------------------
# nature-run generator (standard dataset contract, see data/dataset.py)
# ---------------------------------------------------------------------------


def generate_turbulence(cfg: HuracanConfig, out_dir: str | Path, seed: int = 0):
    """Run the nature run and write the standard dataset files.

    Writes state.npy / obs.npy / forcing.npy / times.json /
    norm_state.json / norm_obs.json with the standard layout
    (see data/dataset.py)."""
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = cfg.data.synthetic
    tcfg = s.turbulence
    grid = grid_from_cfg(cfg.grid)
    npix = grid.npix
    names = cfg.var_names()
    C = s.channels
    if C != len(names):
        raise ValueError(
            f"data.synthetic.channels ({C}) must equal the prognostic "
            f"channel count ({len(names)}) for the turbulence source"
        )
    levels = list(cfg.data.variables.levels)
    nlev = len(levels)
    T = s.steps
    rng = np.random.default_rng(seed)

    atm = TurbulentAtmosphere(grid, tcfg, rng)
    n = atm.n_layers

    # level -> layer: contiguous altitude blocks, layer 0 = near-surface
    # (levels are ascending pressure: index 0 = top of the atmosphere)
    rank = nlev - 1 - np.arange(nlev)  # 0 = highest pressure
    layer_of = np.clip((rank * n) // nlev, 0, n - 1)
    # per-level amplitude: layer jet ratio * mild within-block variation
    layer_ratio = np.linspace(0.55, 1.15, n)
    alpha = np.empty(nlev)
    for j in range(nlev):
        block = np.where(layer_of == layer_of[j])[0]
        span = max(rank[block].max() - rank[block].min(), 1)
        frac = (rank[block].max() - rank[j]) / span  # 0 = block bottom
        alpha[j] = layer_ratio[layer_of[j]] * (0.9 + 0.2 * frac)

    f_cor = 2.0 * EARTH_OMEGA * np.cos(grid.colat)  # (npix,)
    z_base = GAS_CONSTANT * 255.0 * np.log(1013.0 / np.array(levels))
    t_off = -3.0 * rank / max(nlev - 1, 1)  # K, mild aloft
    q_fac = 0.5 + 0.7 * (rank / max(nlev - 1, 1))  # moister near sfc
    # crude column weights [hPa] for tcwv
    dp = np.array(
        [levels[i + 1] - levels[i] if i + 1 < nlev else levels[-1] for i in range(nlev)]
    )

    state = np.empty((T, npix, C), dtype=np.float16)
    ssum = np.zeros(C)
    ssq = np.zeros(C)
    upper = cfg.data.variables.upper
    surf = cfg.data.variables.surface
    idx = {nm: i for i, nm in enumerate(names)}

    print(
        f"turbulence nature run: grid {grid.nlat}x{grid.nlon}, spectral "
        f"L={atm.ops.L}, layers={n}, levels={nlev}, frames={T} "
        f"(+{tcfg.warmup_frames} warmup), dt={tcfg.dt_seconds:.0f}s"
    )
    t0 = time.time()
    for _ in range(tcfg.warmup_frames):
        atm.advance_frame(tcfg.dt_seconds)
        if _ % 3 == 0:
            atm.calibrate_forcing(atm.wind_targets)
    winds = atm.calibrate_forcing(atm.wind_targets)
    print(
        f"  warmup done in {time.time() - t0:.0f}s; winds "
        + "/".join(f"{u:.1f}" for u in winds)
        + f" (target "
        + "/".join(f"{u:.1f}" for u in atm.wind_targets)
        + "), forcing frozen"
    )

    t0 = time.time()
    for t in range(T):
        psi = atm.psi()
        psi_g = atm.ops.synth(psi)  # (npix, n)
        u_e, v_n = atm.ops.wind(psi)  # (npix, n)
        t_g = atm.ops.synth(atm.ttr)
        q_g = np.clip(atm.ops.synth(atm.qtr), 0.0, None)

        frame = np.empty((npix, C), dtype=np.float32)
        for j, lev in enumerate(levels):
            ell = layer_of[j]
            aj = alpha[j]
            frame[:, idx[f"z{lev}"]] = (
                z_base[j] + aj * f_cor * psi_g[:, ell]
            ) / 10.0  # dam
            frame[:, idx[f"t{lev}"]] = t_g[:, ell] + t_off[j]
            frame[:, idx[f"u{lev}"]] = aj * u_e[:, ell]
            frame[:, idx[f"v{lev}"]] = aj * v_n[:, ell]
            frame[:, idx[f"q{lev}"]] = q_fac[j] * q_g[:, ell]
        # surface channels from the near-surface layer/level
        j0 = nlev - 1
        e0 = layer_of[j0]
        u0, v0 = 0.75 * u_e[:, e0], 0.75 * v_n[:, e0]
        t2m = t_g[:, e0] + t_off[j0] + 2.0
        q_low = q_fac[j0] * q_g[:, e0]
        frame[:, idx["10u"]] = u0
        frame[:, idx["10v"]] = v0
        frame[:, idx["2t"]] = t2m
        frame[:, idx["2d"]] = np.clip(t2m - 8.0 * q_low / 12.0, None, t2m - 0.3)
        frame[:, idx["tcc"]] = 0.9 * np.clip(1.4 * q_low / 12.0, 0.0, 1.0)
        frame[:, idx["tcwv"]] = (
            np.sum(
                [q_fac[j] * q_g[:, layer_of[j]] * dp[j] for j in range(nlev)],
                axis=0,
            )
            / (1000.0 * GRAVITY)
            * 100.0
        )
        frame[:, idx["msl"]] = 1013.25 + 1.2e-6 * psi_g[:, e0]  # hPa
        # any extra surface channels not covered above: fill from t2m
        for nm in surf:
            if nm not in ("10u", "10v", "2t", "2d", "tcc", "tcwv", "msl"):
                frame[:, idx[nm]] = t2m

        state[t] = frame
        ssum += frame.sum(axis=0)
        ssq += (frame.astype(np.float64) ** 2).sum(axis=0)
        atm.advance_frame(tcfg.dt_seconds)
        if (t + 1) % 200 == 0:
            print(f"  frame {t + 1}/{T}  ({time.time() - t0:.0f}s)")

    mean = ssum / (T * npix)
    std = np.sqrt(np.maximum(ssq / (T * npix) - mean**2, 0.0))
    norm = Normalizer(names, mean, std + 1e-6)
    norm.save(out / "norm_state.json")
    obs_cfg_norm = obs_norm_mapping(cfg, norm)
    obs_cfg_norm.save(out / "norm_obs.json")

    np.save(out / "state.npy", state)

    # --- forcing channels (insolation cycle + static fields) -------------
    start = dt.datetime(2010, 1, 1)
    times = [(start + dt.timedelta(hours=6 * i)) for i in range(T)]
    colat, lon = grid.colat, grid.lon_flat
    insol = np.stack(
        [
            insolation(colat, lon, t_.hour + t_.minute / 60, t_.timetuple().tm_yday)
            for t_ in times
        ]
    ).astype(np.float32)
    ops = atm.ops

    def low_degree_field(rng, lmax_f, scale=None, binary=False):
        u = rng.standard_normal((ops.B.shape[1], 1))
        u[ops.lidx > lmax_f] = 0.0
        if binary:
            return (ops.synth(u)[:, 0] > 0).astype(np.float32)
        u *= scale / np.sqrt(np.sum(u**2) / (4 * np.pi))
        return ops.synth(u)[:, 0].astype(np.float32)

    zsfc = low_degree_field(rng, 12, scale=3000.0)
    lsm = low_degree_field(rng, 8, binary=True)
    forcing = np.zeros((T, npix, 3), dtype=np.float32)
    forcing[:, :, 0] = insol
    forcing[:, :, 1] = zsfc[None, :]
    forcing[:, :, 2] = lsm[None, :]
    np.save(
        out / "forcing.npy",
        (forcing / np.array([500.0, 10000.0, 1.0], dtype=np.float32)).astype(
            np.float16
        ),
    )
    with open(out / "times.json", "w") as f:
        json.dump([t_.isoformat() for t_ in times], f)

    # --- observations: fixed station network, noisy samples --------------
    write_observations(cfg, out, state, rng)

    print(
        f"turbulence data written to {out}  T={T} npix={npix} C={C} "
        f"({time.time() - t_start:.0f}s)"
    )
    return out


def observation_network(cfg: HuracanConfig, grid, rng):
    """Fixed station pixel indices, sampled area-uniformly."""
    tcfg = cfg.data.synthetic.turbulence
    n_st = max(1, round(cfg.data.synthetic.obs_fraction * grid.npix))
    p = grid.pixel_weights / grid.pixel_weights.sum()
    return rng.choice(grid.npix, size=n_st, replace=False, p=p)


def write_observations(cfg: HuracanConfig, out: Path, state, rng):
    """Generate obs.npy over an existing state.npy (chunked; reuses the
    saved normalizers).  ``state`` may be a memmap."""
    s = cfg.data.synthetic
    tcfg = s.turbulence
    T, npix, _ = state.shape
    names = cfg.var_names()
    upper = cfg.data.variables.upper
    obs_ch = [
        names.index(f"{u}{lev}") for lev in cfg.data.variables.levels for u in upper
    ] + [names.index(sv) for sv in cfg.data.obs.surface_vars]
    norm_obs = Normalizer.load(out / "norm_obs.json")

    if tcfg.obs_fixed_stations:
        grid = grid_from_cfg(cfg.grid)
        stations = observation_network(cfg, grid, rng)
        active = np.zeros((T, npix), dtype=bool)
        rep = rng.random((T, len(stations))) < tcfg.obs_report_prob
        active[:, stations] = rep
    else:
        active = rng.random((T, npix)) < s.obs_fraction

    # per-channel truth std for the noise scale
    std = np.array(norm_obs.std, dtype=np.float64)

    obs = np.empty((T, npix, len(obs_ch)), dtype=np.float16)
    chunk = 128
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        raw = state[a:b, :, :][:, :, obs_ch].astype(np.float32)
        noise = s.obs_noise * std[None, None, :] * rng.standard_normal(raw.shape)
        o = norm_obs.encode(raw + noise.astype(np.float32))
        o[~active[a:b]] = 0.0
        obs[a:b] = o.astype(np.float16)
    np.save(out / "obs.npy", obs)
    return out


def make_obs_variant(
    cfg: HuracanConfig,
    src_root: str | Path,
    dst_root: str | Path,
    *,
    obs_fraction: float,
    obs_noise: float,
    seed: int = 0,
):
    """Clone a turbulence dataset with a different observation network.

    Symlinks the heavy files (state/forcing/times/norms) and regenerates
    only obs.npy -- for sweeping observation density/noise without
    re-running the nature run."""
    import copy as _copy

    src, dst = Path(src_root), Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)
    for f in (
        "state.npy",
        "forcing.npy",
        "times.json",
        "norm_state.json",
        "norm_obs.json",
    ):
        link = dst / f
        if not link.exists():
            link.symlink_to(src.resolve() / f)
    vcfg = _copy.deepcopy(cfg)
    vcfg.data.synthetic.obs_fraction = obs_fraction
    vcfg.data.synthetic.obs_noise = obs_noise
    state = np.load(dst / "state.npy", mmap_mode="r")
    rng = np.random.default_rng(seed + 17)
    write_observations(vcfg, dst, state, rng)
    return dst
