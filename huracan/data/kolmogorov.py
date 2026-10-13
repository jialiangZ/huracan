"""Kolmogorov-flow nature run: 2D Navier-Stokes with sinusoidal forcing on
the periodic square -- the community-standard turbulence benchmark of the
ADDA/LEVDA line, here as the second OSSE data source.

Physics (nondimensional, domain [0, 2pi)^2):

    d(omega)/dt = -u . grad(omega) + nu lap(omega) + A sin(k_f y)
    omega = -lap(psi),   u = (psi_y, -psi_x)

Spectral method: pseudo-spectral (numpy float64) with the 2/3 dealiasing
rule on the quadratic term, RK4 time stepping.  The forcing amplitude is
calibrated during warm-up toward a target rms speed and frozen afterwards,
giving a statistically steady turbulent nature run.  The exact laminar
steady state psi* = A/(nu k_f^4) sin(k_f y) is used as a solver test.

Output contract is identical to ``data/turbulence.py`` / ``synthetic.py``
(state/obs/forcing/norms files, fp16), so every downstream stage --
training phases, EDA cycling, the OSSE controls of ``run_osse.py`` --
runs unchanged.  Recorded channels: [w, psi, u, v] (this source's
``variables.upper``) on a single level.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from ..config.schema import HuracanConfig
from ..core.grid import grid_from_cfg
from .norm import Normalizer, obs_norm_mapping
from .turbulence import write_observations

FRAME_FAKE_HOURS = 6  # calendar label spacing only (nondimensional physics)


class KolmogorovFlow:
    """Pseudo-spectral 2D NS solver on the periodic square (float64).

    Fields are laid out [y, x]; spectra use rfft2 over both axes
    (shape (N, N//2+1), kx >= 0), norm='ortho' so Parseval holds."""

    def __init__(self, N: int, nu: float, kf: int, rng):
        self.N, self.nu, self.kf, self.rng = N, nu, kf, rng
        ky = np.fft.fftfreq(N) * N  # signed y wavenumbers
        kx = np.fft.rfftfreq(N) * N  # non-negative x
        self.k2 = ky[:, None] ** 2 + kx[None, :] ** 2  # (N, N//2+1)
        self.iKY = 1j * ky[:, None] * np.ones((1, N // 2 + 1))
        self.iKX = 1j * np.ones((N, 1)) * kx[None, :]
        # 2/3 rule: project inputs to |k| <= N//3 so quadratic products
        # (bandwidth <= 2N//3) are alias-free on the N-point grid; the same
        # mask truncates the nonlinear output (spectral cutoff)
        self.dealias = (np.abs(ky[:, None]) <= N // 3) & (np.abs(kx[None, :]) <= N // 3)
        y = 2 * np.pi * np.arange(N) / N
        self.forcing_pattern = np.sin(kf * y)[:, None].repeat(N, axis=1)
        self.forcing_amp = 1.0  # calibrated in warm-up
        # random band-limited initial condition around the forced band
        wh = rng.standard_normal(self.k2.shape)
        band = (np.abs(ky[:, None]) <= kf + 2) & (np.abs(kx[None, :]) <= kf + 2)
        self.omega_hat = np.where(band, wh, 0.0).astype(np.complex128)

    # -- spectral helpers -------------------------------------------------
    def _inv(self, c):
        return np.fft.irfft2(c, s=(self.N, self.N), norm="ortho")

    def psi_hat(self, omega_hat):
        """omega = -lap(psi): lap has spectrum -k^2, so omega_hat = k^2
        psi_hat and psi_hat = omega_hat / k^2."""
        out = np.zeros_like(omega_hat)
        nz = self.k2 > 0
        out[nz] = omega_hat[nz] / self.k2[nz]
        return out

    def velocity(self, omega_hat=None):
        ph = self.psi_hat(omega_hat if omega_hat is not None else self.omega_hat)
        return self._inv(ph * self.iKY), self._inv(-ph * self.iKX)

    def u_rms(self) -> float:
        u, v = self.velocity()
        return float(np.sqrt((u**2 + v**2).mean()))

    def rhs(self, omega_hat):
        w = omega_hat * self.dealias  # projection: keep inputs band-limited
        ph = self.psi_hat(w)
        u = self._inv(ph * self.iKY)  # psi_y
        v = self._inv(-ph * self.iKX)  # -psi_x
        wx = self._inv(w * self.iKX)
        wy = self._inv(w * self.iKY)
        adv = np.fft.rfft2(u * wx + v * wy, norm="ortho") * self.dealias
        visc = -self.nu * self.k2 * w
        f = np.fft.rfft2(self.forcing_amp * self.forcing_pattern, norm="ortho")
        return -adv + visc + f

    def step(self, h: float):
        w = self.omega_hat
        k1 = self.rhs(w)
        k2 = self.rhs(w + 0.5 * h * k1)
        k3 = self.rhs(w + 0.5 * h * k2)
        k4 = self.rhs(w + h * k3)
        self.omega_hat = w + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def advance(self, dt: float, t_frame: float):
        n = int(round(t_frame / dt))
        if abs(n * dt - t_frame) > 1e-9:
            raise ValueError(f"frame {t_frame} not divisible by dt {dt}")
        for _ in range(n):
            self.step(dt)

    def calibrate(self, target_u: float):
        """Proportional controller on the forcing amplitude (warm-up only)."""
        ratio = np.clip(target_u / max(self.u_rms(), 1e-6), 0.85, 1.18)
        self.forcing_amp *= float(ratio)
        return self.u_rms()

    def state(self):
        """Physical fields (w, psi, u, v), each (N, N)."""
        ph = self.psi_hat(self.omega_hat)
        w = self._inv(self.omega_hat)
        psi = self._inv(ph)
        u = self._inv(ph * self.iKY)
        v = self._inv(-ph * self.iKX)
        return w, psi, u, v


def kolmogorov_steady_state(N: int, nu: float, kf: int, A: float = 1.0):
    """Exact laminar steady state: psi* = A/(nu kf^4) sin(kf y).

    At steady state nu*lap(omega) + f = 0 and u.grad(omega) = 0."""
    y = 2 * np.pi * np.arange(N) / N
    psi = (A / (nu * kf**4)) * np.sin(kf * y)[:, None].repeat(N, axis=1)
    omega = (kf**2) * psi  # omega = -lap(psi) = kf^2 psi
    return psi, omega


# ---------------------------------------------------------------------------
# nature-run generator (output contract identical to turbulence.py)
# ---------------------------------------------------------------------------


def generate_kolmogorov(cfg: HuracanConfig, out_dir: str | Path, seed: int = 0):
    """Run the Kolmogorov nature run and write the standard dataset files."""
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = cfg.data.synthetic
    kcfg = s.kolmogorov
    grid = grid_from_cfg(cfg.grid)
    npix = grid.npix
    names = cfg.var_names()
    C = s.channels
    if C != len(names):
        raise ValueError(
            f"data.synthetic.channels ({C}) must equal the prognostic "
            f"channel count ({len(names)}) for the kolmogorov source"
        )
    upper = cfg.data.variables.upper
    if upper != ["w", "psi", "u", "v"] or len(cfg.data.variables.levels) != 1:
        raise ValueError(
            "kolmogorov source expects variables.upper = [w, psi, u, v] "
            f"with a single level; got {upper} x "
            f"{len(cfg.data.variables.levels)}"
        )
    T = s.steps
    rng = np.random.default_rng(seed)

    sim = KolmogorovFlow(grid.nlat, kcfg.nu, kcfg.kf, rng)
    nsub = int(round(kcfg.dt_frame / kcfg.dt))

    print(
        f"kolmogorov nature run: grid {grid.nlat}x{grid.nlon}, "
        f"nu={kcfg.nu}, kf={kcfg.kf}, frames={T} "
        f"(+{kcfg.warmup_frames} warmup), dt={kcfg.dt} x {nsub}/frame"
    )
    t0 = time.time()
    for i in range(kcfg.warmup_frames):
        sim.advance(kcfg.dt, kcfg.dt_frame)
        if i % 4 == 0:
            sim.calibrate(kcfg.u_rms)
    urms = sim.calibrate(kcfg.u_rms)
    print(
        f"  warmup done in {time.time() - t0:.0f}s; u_rms {urms:.3f} "
        f"(target {kcfg.u_rms}), Re ~ {urms / (kcfg.nu * kcfg.kf):.0f}, "
        f"forcing frozen"
    )

    state = np.empty((T, npix, C), dtype=np.float16)
    ssum = np.zeros(C)
    ssq = np.zeros(C)

    t0 = time.time()
    for t in range(T):
        frame = np.stack(sim.state(), axis=-1).reshape(npix, C)
        state[t] = frame
        ssum += frame.sum(axis=0)
        ssq += (frame.astype(np.float64) ** 2).sum(axis=0)
        sim.advance(kcfg.dt, kcfg.dt_frame)
        if (t + 1) % 200 == 0:
            print(f"  frame {t + 1}/{T}  ({time.time() - t0:.0f}s)")

    mean = ssum / (T * npix)
    std = np.sqrt(np.maximum(ssq / (T * npix) - mean**2, 0.0))
    norm = Normalizer(names, mean, std + 1e-6)
    norm.save(out / "norm_state.json")
    obs_norm_mapping(cfg, norm).save(out / "norm_obs.json")
    np.save(out / "state.npy", state)

    # no physical forcings: empty forcing block (calendar semantics only)
    np.save(out / "forcing.npy", np.zeros((T, npix, 0), dtype=np.float16))
    start = dt.datetime(2010, 1, 1)
    times = [(start + dt.timedelta(hours=FRAME_FAKE_HOURS * i)) for i in range(T)]
    with open(out / "times.json", "w") as f:
        json.dump([t_.isoformat() for t_ in times], f)

    write_observations(cfg, out, state, rng)

    print(
        f"kolmogorov data written to {out}  T={T} npix={npix} C={C} "
        f"({time.time() - t_start:.0f}s)"
    )
    return out
