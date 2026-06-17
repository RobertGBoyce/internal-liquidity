"""FX Market Making with Internal Liquidity.

Numerical solver and Monte-Carlo simulator for the model in

    Barzykin, Boyce & Neuman, "FX Market Making with Internal Liquidity" (2025).

The market maker continuously streams a bid/ask ladder of sizes ``z`` to
external OTC clients, and these are filled with a Poisson rate depending
on the market maker's quotes.  The internal order size process ``L`` is driven
by client cancellations ``C`` (rate ``nu``), arrivals ``A`` (rate ``mu``), market maker's
trades ``M`` (unit size) and replenishment ``R`` (probability ``p`` when the last unit
is consumed).

Units follow the paper: half-spreads / ``rho`` in bps, ``kappa`` in bps^-1,
intensities in s^-1, sizes in notional units (1 = 1mm), time in seconds.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Back up half-spread for an unavailable quote (boundary of the inventory
# grid).  Finite so that ``exp(-kappa*NO_QUOTE)`` underflows to 0 (no fill) and
# arithmetic such as ``size * count * (S + depth)`` stays NaN-free.
NO_QUOTE = 1.0e6

# Cap on the Hamiltonian exponent ``(kappa/z)*(h(q±z)-h(q))`` in the fill terms.
# One-sided differences blow up at the inventory boundary. Clipping keeps the
# explicit scheme from overflowing without affecting the interior solution.
_EXP_CAP = 30.0


# ---------------------------------------------------------------------------
# Model parameters and presets
# ---------------------------------------------------------------------------
@dataclass
class ModelParams:
    """All parameters of the internal-liquidity market-making model."""

    # OTC pricing ladder ------------------------------------------------------
    sizes: Tuple[int, ...] = (1, 5, 10)
    kappa: Dict[int, float] = field(
        default_factory=lambda: {1: 1.5, 5: 1.0, 10: 0.5}
    )  # bps^-1
    lam_a: Dict[int, float] = field(
        default_factory=lambda: {1: 0.2, 5: 0.005, 10: 0.001}
    )  # s^-1, ask side
    lam_b: Dict[int, float] = field(
        default_factory=lambda: {1: 0.2, 5: 0.005, 10: 0.001}
    )  # s^-1, bid side

    # Risk penalties ----------------------------------------------------------
    alpha: float = 0.001  # terminal inventory penalty
    phi: float = 0.001    # running inventory penalty
    psi: float = 0.01     # running penalty on unfilled internal order, ``psi*(L)^+``

    # Internal exchange -------------------------------------------------------
    rho: float = 0.0      # net price offset = ``tilde_rho - xi`` (client offset minus fee)
    ell: int = 1          # block/order size unit
    p: float = 0.0        # replenishment probability when last unit is taken
    mu: float = 0.0       # arrival intensity (acts only when ``L <= 0``)
    nu: float = 0.001     # cancellation intensity (acts only when ``L > 0``)

    # Market / horizon --------------------------------------------------------
    sigma: float = 0.3         # mid-price volatility in bps/√s 
    T: float = 300.0      # trading horizon in seconds

    # Inventory grid ----------------------------------------------------------
    q_max: int = 50       # inventory grid is ``{-q_max, ..., q_max}``

    def __post_init__(self) -> None:
        self.sizes = tuple(int(z) for z in self.sizes)
        self.ell = int(self.ell)
        self.q_max = int(self.q_max)

    # Grid helpers ------------------------------------------------------------
    @property
    def q_values(self) -> np.ndarray:
        return np.arange(-self.q_max, self.q_max + 1)

    @property
    def l_values(self) -> np.ndarray:
        return np.arange(1 - self.ell, self.ell + 1)

    @property
    def n_q(self) -> int:
        return 2 * self.q_max + 1

    @property
    def n_l(self) -> int:
        return 2 * self.ell

    def q_index(self, q: np.ndarray) -> np.ndarray:
        """Inventory -> grid index, clipped to the grid (boundary lookup)."""
        return np.clip(np.asarray(q) + self.q_max, 0, self.n_q - 1)

    def l_index(self, l: np.ndarray) -> np.ndarray:
        """Internal order size -> grid index (values live in ``[1-ell, ell]``)."""
        return np.clip(np.asarray(l) - (1 - self.ell), 0, self.n_l - 1)


def iceberg(rho: float = 0.0, **overrides) -> ModelParams:
    """Iceberg client: immediate replenishment (``p=0.9``), rare cancellations."""
    return ModelParams(ell=1, p=0.9, mu=0.0, nu=0.001, rho=rho, **overrides)


def twap(rho: float = 0.0, **overrides) -> ModelParams:
    """TWAP client: no instant replenishment, paused renewals (``mu=0.05``)."""
    return ModelParams(ell=1, p=0.0, mu=0.05, nu=0.001, rho=rho, **overrides)


def full_amount(rho: float = 0.0, **overrides) -> ModelParams:
    """Full-amount client: one large order (``ell=10``), no renewal/replenishment."""
    return ModelParams(ell=10, p=0.0, mu=0.0, nu=0.001, rho=rho, **overrides)


# ---------------------------------------------------------------------------
# HJBQVI operators
# ---------------------------------------------------------------------------
def _generator(H: np.ndarray, p: ModelParams, q2: np.ndarray, lpos: np.ndarray) -> np.ndarray:
    """Continuation generator ``L h`` of eq. (2.7) evaluated on the grid.

    Returns an array with the same shape as ``H`` holding, at each ``(q, l)``,

        -phi q^2 - psi (l)^+
        + sum_{z,i} (z lambda^{i,z}/kappa_z) e^{-1} exp((kappa_z/z)(h(q±z,l)-h(q,l)))
        + nu (h(q,l-ell)-h(q,l)) 1_{l>0}  +  mu (h(q,l+ell)-h(q,l)) 1_{l<=0}.
    """
    nq, _ = H.shape
    ell = p.ell

    G = -p.phi * q2[:, None] - p.psi * lpos[None, :]

    for z in p.sizes:
        kz = p.kappa[z]
        kzz = kz / z
        coef = (z / kz) * np.exp(-1.0)

        # Bid fills
        hb = np.zeros_like(H)
        hb[: nq - z] = H[z:]
        bterm = (p.lam_b[z] * coef) * np.exp(np.minimum(kzz * (hb - H), _EXP_CAP))
        G[: nq - z] += bterm[: nq - z]

        # Ask fills
        ha = np.zeros_like(H)
        ha[z:] = H[: nq - z]
        aterm = (p.lam_a[z] * coef) * np.exp(np.minimum(kzz * (ha - H), _EXP_CAP))
        G[z:] += aterm[z:]

    # Cancellations
    G[:, ell:] += p.nu * (H[:, :ell] - H[:, ell:])
    # Arrivals
    G[:, :ell] += p.mu * (H[:, ell:] - H[:, :ell])

    return G


def _intervention(H: np.ndarray, p: ModelParams) -> np.ndarray:
    """Value of taking one internal unit

    Returns ``M`` with ``M[iq,il]`` equal to the post-trade continuation value

        h(q+1, l-1)                              if l > 1
        p h(q+1, l-1+ell) + (1-p) h(q+1, l-1)    if l = 1   (replenishment)

    minus the trade cost ``rho``.  Entries where no trade is admissible
    (``l <= 0`` or ``q+1`` off-grid) are ``-inf``.
    """
    nq, nl = H.shape
    ell = p.ell

    M = np.full_like(H, -np.inf)

    hq1 = np.zeros_like(H)
    hq1[: nq - 1] = H[1:]

    if nl > ell + 1:
        M[:, ell + 1 :] = hq1[:, ell : nl - 1] - p.rho
    M[:, ell] = p.p * hq1[:, nl - 1] + (1.0 - p.p) * hq1[:, ell - 1] - p.rho

    M[nq - 1, :] = -np.inf
    return M


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class HJBSolution:
    """Result of :func:`solve_hjbqvi`."""

    params: ModelParams
    snapshots: np.ndarray      # (n_snap, n_q, n_l), ascending in time, [0]=t0, [-1]=T
    snap_times: np.ndarray     # (n_snap,)
    dt: float                  # PDE time step actually used

    def h_at(self, t: float) -> np.ndarray:
        """Return the ``h(t, .,.)`` slice nearest to time ``t``."""
        idx = int(np.clip(np.round(t / (self.snap_times[1] - self.snap_times[0])),
                          0, len(self.snap_times) - 1))
        return self.snapshots[idx]


def solve_hjbqvi(
    p: ModelParams,
    dt: float = 0.02,
    snap_dt: float = 0.3,
    obstacle_tol: float = 1e-9,
    max_obstacle_iter: Optional[int] = None,
) -> HJBSolution:
    """Solve the reduced HJBQVI (2.7) for h(t, q, l) backward in time.

    Parameters
    ----------
    p:
        Model parameters.
    dt:
        Explicit-Euler time step for the PDE (seconds).  Must divide snap_dt.
    snap_dt:
        Spacing at which h snapshots are retained (used by the simulator,
        in particular for the time-dependent policy).  Default 0.3s = sim step.
    obstacle_tol, max_obstacle_iter:
        Tolerance and cap for the per-step fixed-point that applies the
        intervention obstacle (resolves simultaneous multi-unit takes).
    """
    q2 = (p.q_values ** 2).astype(float)
    lpos = np.maximum(p.l_values, 0).astype(float)

    n_steps = int(round(p.T / dt))
    dt = p.T / n_steps  # exact divisor
    snap_step = max(1, int(round(snap_dt / dt)))
    n_snap = n_steps // snap_step + 1
    if max_obstacle_iter is None:
        max_obstacle_iter = 4 * p.q_max + 4

    # Terminal condition ``h(T, q, l) = -alpha q^2``.
    H = (-p.alpha * q2)[:, None] * np.ones((1, p.n_l))

    snaps = [None] * n_snap
    snaps[n_snap - 1] = H.copy()  # t = T

    for n in range(n_steps, 0, -1):
        # Explicit backward Euler on the continuation equation.
        Hc = H + dt * _generator(H, p, q2, lpos)

        # Obstacle / intervention: ``H = max(continuation, take-one-unit)``, iterated
        # to a fixed point so several units can be consumed at one instant.
        Hn = Hc
        for _ in range(max_obstacle_iter):
            updated = np.maximum(Hn, _intervention(Hn, p))
            if np.max(np.abs(updated - Hn)) < obstacle_tol:
                Hn = updated
                break
            Hn = updated
        H = Hn

        step_from_end = n - 1  # this ``H`` is ``h`` at time t_{n-1}
        if step_from_end % snap_step == 0:
            snaps[step_from_end // snap_step] = H.copy()

    snap_times = np.arange(n_snap) * snap_step * dt
    return HJBSolution(params=p, snapshots=np.array(snaps), snap_times=snap_times, dt=dt)


def solve_avellaneda_stoikov(p: ModelParams, dt: float = 0.02) -> np.ndarray:
    """Solve the no-internal-exchange (``L == 0``) benchmark for ``h^{AS}(0, q)``.

    Returns the ``t=0`` slice ``h^{AS}(0, .)`` as a 1-D array over the inventory grid.
    """
    q2 = (p.q_values ** 2).astype(float)
    nq = p.n_q
    n_steps = int(round(p.T / dt))
    dt = p.T / n_steps

    h = -p.alpha * q2
    for _ in range(n_steps):
        G = -p.phi * q2
        for z in p.sizes:
            kz = p.kappa[z]
            kzz = kz / z
            coef = (z / kz) * np.exp(-1.0)
            hb = np.zeros_like(h)
            hb[: nq - z] = h[z:]
            bterm = (p.lam_b[z] * coef) * np.exp(np.minimum(kzz * (hb - h), _EXP_CAP))
            G[: nq - z] += bterm[: nq - z]
            ha = np.zeros_like(h)
            ha[z:] = h[: nq - z]
            aterm = (p.lam_a[z] * coef) * np.exp(np.minimum(kzz * (ha - h), _EXP_CAP))
            G[z:] += aterm[z:]
        h = h + dt * G
    return h


# ---------------------------------------------------------------------------
# Depth / policy helpers
# ---------------------------------------------------------------------------
def _depths_from_slice(Hs: np.ndarray, p: ModelParams):
    """Bid/ask half-spread tables from an ``h`` slice.

    Returns ``(db, da)`` dicts mapping ``z`` to array ``(n_q, n_l)``; unavailable
    quotes at the inventory boundary are set to :data:`NO_QUOTE`.
    """
    nq = p.n_q
    db: Dict[int, np.ndarray] = {}
    da: Dict[int, np.ndarray] = {}
    for z in p.sizes:
        kz = p.kappa[z]
        # Bid: delta = 1/kappa + (h(q,l) - h(q+z,l))/z.
        b = np.full_like(Hs, NO_QUOTE)
        b[: nq - z] = 1.0 / kz + (Hs[: nq - z] - Hs[z:]) / z
        db[z] = b
        # Ask: delta = 1/kappa + (h(q,l) - h(q-z,l))/z.
        a = np.full_like(Hs, NO_QUOTE)
        a[z:] = 1.0 / kz + (Hs[z:] - Hs[: nq - z]) / z
        da[z] = a
    return db, da


def _avellaneda_stoikov_depths(h_as: np.ndarray, p: ModelParams):
    """AS half-spreads (1-D in ``q``) from h^{AS}(0, .); boundary to ``NO_QUOTE``."""
    nq = p.n_q
    db: Dict[int, np.ndarray] = {}
    da: Dict[int, np.ndarray] = {}
    for z in p.sizes:
        kz = p.kappa[z]
        b = np.full(nq, NO_QUOTE)
        b[: nq - z] = 1.0 / kz + (h_as[: nq - z] - h_as[z:]) / z
        db[z] = b
        a = np.full(nq, NO_QUOTE)
        a[z:] = 1.0 / kz + (h_as[z:] - h_as[: nq - z]) / z
        da[z] = a
    return db, da


def _naive_ask_depths(da_as_at_q: Dict[int, float], sizes, l: int, client_price: float):
    """Naive benchmark ask half-spreads at one inventory level.

    Builds the marginal-price ladder implied by the AS ask depths, inserts the
    client sell order (size l at client_price = tilde_rho + iota), then
    returns the volume-weighted average half-spread for each cumulative size.
    """
    # Marginal ask price for each native slice (z_{k-1}, z_k].
    slices = []
    prev_z = 0
    prev_pd = 0.0
    for z in sizes:
        pd = z * da_as_at_q[z]
        slices.append([(pd - prev_pd) / (z - prev_z), float(z - prev_z)])
        prev_z, prev_pd = z, pd
    # Client order slice.
    slices.append([float(client_price), float(l)])
    slices.sort(key=lambda s: s[0])

    out: Dict[int, float] = {}
    for z in sizes:
        need = float(z)
        tot = 0.0
        for price, size in slices:
            take = min(size, need)
            tot += price * take
            need -= take
            if need <= 1e-12:
                break
        out[z] = tot / z
    return out


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class OptimalPolicy:
    """Optimal strategy: HJBQVI half-spreads + take inside the execution region.

    ``stationary=True`` (default, matching Section 4 of the paper) freezes the
    ``t=0`` policy across the horizon; ``stationary=False`` uses the full
    time-dependent solution.
    """

    kind = "optimal"

    def __init__(self, solution: HJBSolution, stationary: bool = True, exec_tol: float = 1e-7):
        self.sol = solution
        self.p = solution.params
        self.stationary = stationary
        self.exec_tol = exec_tol
        self._snap_dt = solution.snap_times[1] - solution.snap_times[0]

    def _slice(self, t: float) -> np.ndarray:
        if self.stationary:
            return self.sol.snapshots[0]
        idx = int(np.clip(round(t / self._snap_dt), 0, len(self.sol.snap_times) - 1))
        return self.sol.snapshots[idx]

    def quote_depths(self, t, Q, L):
        """Return (db, da) dicts of per-path half-spread arrays."""
        p = self.p
        Hs = self._slice(t)
        iq = p.q_index(Q)
        il = p.l_index(L)
        nq = p.n_q
        db: Dict[int, np.ndarray] = {}
        da: Dict[int, np.ndarray] = {}
        h0 = Hs[iq, il]
        for z in p.sizes:
            valid_b = iq + z < nq
            hb = Hs[np.clip(iq + z, 0, nq - 1), il]
            b = np.where(valid_b, 1.0 / p.kappa[z] + (h0 - hb) / z, NO_QUOTE)
            db[z] = b
            valid_a = iq - z >= 0
            ha = Hs[np.clip(iq - z, 0, nq - 1), il]
            a = np.where(valid_a, 1.0 / p.kappa[z] + (h0 - ha) / z, NO_QUOTE)
            da[z] = a
        return db, da

    def _in_execution_region(self, t, Q, L):
        """Boolean per-path mask to check if it is optimal to take an internal unit"""
        p = self.p
        Hs = self._slice(t)
        nq, nl = Hs.shape
        ell = p.ell
        iq = p.q_index(Q)
        il = p.l_index(L)

        present = L > 0
        iq1 = iq + 1
        valid = iq1 < nq
        iq1c = np.clip(iq1, 0, nq - 1)

        is_one = L == 1
        hnext = np.where(
            is_one,
            p.p * Hs[iq1c, nl - 1] + (1.0 - p.p) * Hs[iq1c, ell - 1],
            Hs[iq1c, np.clip(il - 1, 0, nl - 1)],
        )
        m_val = hnext - p.rho
        return present & valid & (m_val >= Hs[iq, il] - self.exec_tol)

    def internal_trade(self, t, S, Q, X, L, rng, cnt):
        """Take internal units (in place) while inside the execution region.

        Updates ``Q``, ``X``, ``L`` and accumulates per-path.
        """
        p = self.p
        n = Q.shape[0]
        max_iter = 4 * p.q_max + 4
        for _ in range(max_iter):
            go = self._in_execution_region(t, Q, L)
            if not go.any():
                break
            X[go] -= S[go] + p.rho
            Q[go] += 1
            cnt[go] += 1
            last = go & (L == 1)
            not_last = go & (L > 1)
            L[not_last] -= 1
            if last.any():
                repl = last & (rng.random(n) < p.p)
                L[last] = 0
                L[repl] = p.ell
        return cnt


class NaivePolicy:
    """Naive benchmark

    Quotes AS bid depths unchanged and AS ask depths VWAP-adjusted for the
    inserted client order; takes one internal unit per step whenever
    inventory is negative and an internal order is present.
    """

    kind = "naive"

    def __init__(self, p: ModelParams, h_as: np.ndarray, iota: float = 0.1,
                 tilde_rho: Optional[float] = None):
        self.p = p
        self.iota = iota
        self.tilde_rho = p.rho if tilde_rho is None else tilde_rho

        db_as, da_as = _avellaneda_stoikov_depths(h_as, p)
        nq, nl = p.n_q, p.n_l
        l_vals = p.l_values

        # Bid depths: AS, broadcast across ``l``.
        self.db = {z: np.repeat(db_as[z][:, None], nl, axis=1) for z in p.sizes}
        # Ask depths: AS when absent (``l<=0``); VWAP-adjusted when present (``l>0``).
        self.da = {z: np.full((nq, nl), NO_QUOTE) for z in p.sizes}
        client_price = self.tilde_rho + iota
        for iq in range(nq):
            dvec = {z: float(da_as[z][iq]) for z in p.sizes}
            for il in range(nl):
                lval = int(l_vals[il])
                if lval <= 0:
                    for z in p.sizes:
                        self.da[z][iq, il] = da_as[z][iq]
                else:
                    adj = _naive_ask_depths(dvec, p.sizes, lval, client_price)
                    for z in p.sizes:
                        self.da[z][iq, il] = adj[z]

    def quote_depths(self, t, Q, L):
        p = self.p
        iq = p.q_index(Q)
        il = p.l_index(L)
        db = {z: self.db[z][iq, il] for z in p.sizes}
        da = {z: self.da[z][iq, il] for z in p.sizes}
        return db, da

    def internal_trade(self, t, S, Q, X, L, rng, cnt):
        p = self.p
        n = Q.shape[0]
        go = (L > 0) & (Q < 0)
        if go.any():
            X[go] -= S[go] + p.rho
            Q[go] += 1
            cnt[go] += 1
            last = go & (L == 1)
            not_last = go & (L > 1)
            L[not_last] -= 1
            if last.any():
                repl = last & (rng.random(n) < p.p)
                L[last] = 0
                L[repl] = p.ell
        return cnt


class ASPolicy:
    """Pure Avellaneda–Stoikov strategy.

    Quotes the classical AS half-spreads (ignoring the internal order entirely)
    and never takes liquidity from the internal exchange.  
    """

    kind = "avellaneda_stoikov"

    def __init__(self, p: ModelParams, h_as: np.ndarray):
        self.p = p
        db_as, da_as = _avellaneda_stoikov_depths(h_as, p)
        nl = p.n_l
        # Broadcast AS depths uniformly across all ``l`` columns.
        self.db = {z: np.repeat(db_as[z][:, None], nl, axis=1) for z in p.sizes}
        self.da = {z: np.repeat(da_as[z][:, None], nl, axis=1) for z in p.sizes}

    def quote_depths(self, t, Q, L):
        p = self.p
        iq = p.q_index(Q)
        il = p.l_index(L)
        return ({z: self.db[z][iq, il] for z in p.sizes},
                {z: self.da[z][iq, il] for z in p.sizes})

    def internal_trade(self, t, S, Q, X, L, rng, cnt):
        return cnt   # never interact with the internal exchange


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    """Output of :func:`simulate`.

    Arrays are shaped ``(n_paths, n_steps+1)`` for state variables recorded on
    the time grid ``times``, and ``(n_paths, n_steps)`` for per-step event
    counts recorded at ``trade_times = times[:-1]`` (the event occurs at the
    start of the step, before that step's price move).
    """

    params: ModelParams
    times: np.ndarray              # (n_steps+1,)
    S: np.ndarray                  # mid price
    Q: np.ndarray                  # inventory
    X: np.ndarray                  # cash
    L: np.ndarray                  # internal order size
    internal_trades: np.ndarray    # (n_paths, n_steps) int, internal takes per step
    M: np.ndarray                  # (n_paths, n_steps+1) cumulative internal takes
    external_dq: np.ndarray        # (n_paths, n_steps) inventory change from OTC fills
    first_fill_time: np.ndarray    # (n_paths,) time of first internal take (inf if none)

    @property
    def trade_times(self) -> np.ndarray:
        return self.times[:-1]

    @property
    def pnl(self) -> np.ndarray:
        """Terminal mark-to-market PnL, ``X_T + Q_T S_T``."""
        return self.X[:, -1] + self.Q[:, -1] * self.S[:, -1]

    @property
    def internal_volume(self) -> np.ndarray:
        """Total internal volume per path, ``M_T``."""
        return self.M[:, -1]

    def internal_trade_times(self, path: int) -> np.ndarray:
        """Times at which a given path took internal liquidity."""
        idx = np.nonzero(self.internal_trades[path] > 0)[0]
        return self.trade_times[idx]

    def summary(self) -> dict:
        ff = self.first_fill_time
        filled = np.isfinite(ff)
        return {
            "pnl_mean": float(self.pnl.mean()),
            "pnl_std": float(self.pnl.std()),
            "internal_volume_mean": float(self.internal_volume.mean()),
            "internal_volume_per_sec": float(self.internal_volume.mean() / self.params.T),
            "first_fill_mean": float(ff[filled].mean()) if filled.any() else float("inf"),
            "first_fill_std": float(ff[filled].std()) if filled.any() else 0.0,
            "fill_fraction": float(filled.mean()),
        }


def simulate(
    p: ModelParams,
    policy,
    n_paths: int = 5000,
    dt: float = 0.3,
    S0: float = 10000.0,
    Q0: int = 0,
    L0: Optional[int] = None,
    seed: int = 0,
) -> SimResult:
    """Vectorised fixed-step Euler simulation of the model under ``policy``.

    Within each step of length ``dt`` the order of events is:
      1. internal trading (instantaneous; policy-defined);
      2. OTC fills over ``[t, t+dt]`` (per size/side Poisson counts);
      3. internal-order exogenous events (cancellation if present, else arrival);
      4. mid-price increment ``sigma sqrt(dt) Z``.

    The RNG is split into two independent streams (via ``SeedSequence``):
    ``event_rng`` for all OTC fills, cancellations, arrivals and replenishment;
    ``price_rng`` for the Brownian mid-price increments.  Two simulations with
    the same ``seed`` therefore share identical mid-price paths regardless of
    which policy is used.
    """
    ss = np.random.SeedSequence(seed)
    event_rng, price_rng = [np.random.default_rng(s) for s in ss.spawn(2)]
    ell = p.ell
    if L0 is None:
        L0 = ell

    n_steps = int(round(p.T / dt))
    dt = p.T / n_steps
    sqdt = np.sqrt(dt)
    times = np.arange(n_steps + 1) * dt

    S = np.full(n_paths, float(S0))
    Q = np.full(n_paths, int(Q0))
    X = np.zeros(n_paths)
    L = np.full(n_paths, int(L0))

    S_path = np.empty((n_paths, n_steps + 1))
    Q_path = np.empty((n_paths, n_steps + 1), dtype=np.int64)
    X_path = np.empty((n_paths, n_steps + 1))
    L_path = np.empty((n_paths, n_steps + 1), dtype=np.int64)
    internal_trades = np.zeros((n_paths, n_steps), dtype=np.int64)
    external_dq = np.zeros((n_paths, n_steps), dtype=np.int64)
    M_path = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    first_fill = np.full(n_paths, np.inf)

    S_path[:, 0], Q_path[:, 0], X_path[:, 0], L_path[:, 0] = S, Q, X, L

    p_cancel = 1.0 - np.exp(-p.nu * dt)
    p_arrive = 1.0 - np.exp(-p.mu * dt)

    for n in range(n_steps):
        t = times[n]

        # 1. Internal trading (in place).
        cnt = np.zeros(n_paths, dtype=np.int64)
        policy.internal_trade(t, S, Q, X, L, event_rng, cnt)
        internal_trades[:, n] = cnt
        newly = (cnt > 0) & ~np.isfinite(first_fill)
        first_fill[newly] = t

        # 2. OTC fills over the step.
        db, da = policy.quote_depths(t, Q, L)
        dq = np.zeros(n_paths, dtype=np.int64)
        for z in p.sizes:
            # Bid: maker buys z, q += z, cash -= z (S - delta_b).
            inten_b = np.where(db[z] < NO_QUOTE, p.lam_b[z] * np.exp(-p.kappa[z] * db[z]), 0.0)
            nb = event_rng.poisson(inten_b * dt)
            depth_b = np.where(db[z] < NO_QUOTE, db[z], 0.0)
            Q += z * nb
            dq += z * nb
            X -= z * nb * (S - depth_b)
            # Ask: maker sells z, q -= z, cash += z (S + delta_a).
            inten_a = np.where(da[z] < NO_QUOTE, p.lam_a[z] * np.exp(-p.kappa[z] * da[z]), 0.0)
            na = event_rng.poisson(inten_a * dt)
            depth_a = np.where(da[z] < NO_QUOTE, da[z], 0.0)
            Q -= z * na
            dq -= z * na
            X += z * na * (S + depth_a)
        external_dq[:, n] = dq

        # 3. Internal-order exogenous dynamics.
        present = L > 0
        cancel = present & (event_rng.random(n_paths) < p_cancel)
        L[cancel] -= ell
        absent = L <= 0
        arrive = absent & (event_rng.random(n_paths) < p_arrive)
        L[arrive] += ell

        # 4. Mid-price increment.
        S = S + p.sigma * sqdt * price_rng.standard_normal(n_paths)

        S_path[:, n + 1] = S
        Q_path[:, n + 1] = Q
        X_path[:, n + 1] = X
        L_path[:, n + 1] = L
        M_path[:, n + 1] = M_path[:, n] + cnt

    return SimResult(
        params=p,
        times=times,
        S=S_path,
        Q=Q_path,
        X=X_path,
        L=L_path,
        internal_trades=internal_trades,
        M=M_path,
        external_dq=external_dq,
        first_fill_time=first_fill,
    )


# ---------------------------------------------------------------------------
# Execution region helpers
# ---------------------------------------------------------------------------

def execution_region_at(solution: HJBSolution, time: float = 0.0) -> np.ndarray:
    """Boolean ``(n_q, n_l)`` array: ``True`` where the maker immediately takes the internal order.

    The execution region at time ``t`` is the set of ``(q, l)`` pairs where the
    intervention value weakly dominates the continuation value.
    """
    p = solution.params
    snap_dt = solution.snap_times[1] - solution.snap_times[0]
    idx = int(np.clip(round(time / snap_dt), 0, len(solution.snap_times) - 1))
    H = solution.snapshots[idx]
    M = _intervention(H, p)
    return np.isfinite(M) & (M >= H - 1e-7)


def execution_boundary_at(
    solution: HJBSolution,
    time: float = 0.0,
    liquidity: Optional[int] = None,
) -> float:
    """Largest inventory ``q`` at which the maker takes the internal order.

    Returns ``-inf`` when the execution region is empty for the given ``l`` level.
    ``liquidity`` defaults to ``params.ell`` (one full unit present).
    """
    p = solution.params
    if liquidity is None:
        liquidity = p.ell
    l_idx = int(p.l_index(np.array([liquidity]))[0])
    region = execution_region_at(solution, time)
    col = region[:, l_idx]
    return float(p.q_values[col].max()) if col.any() else -np.inf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    plt.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["grid.color"] = "lightgrey"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.bottom"] = False
    plt.rcParams["axes.titlepad"] = 12
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18


def plot_deltas_time(
    solution: HJBSolution,
    size: int = 1,
    plot_to_q: int = 7,
    colbar: bool = True,
    save_pdf: bool = False,
) -> None:
    """2 × 2 grid: bid/ask half-spreads vs time for ``l = 0`` and ``l = ell``.

    Each curve is one inventory level ``q``, coloured by ``q`` value (indigo →
    yellow).  Rows correspond to no-liquidity (``l = 0``) and with-liquidity
    (``l = ell``); columns are bid and ask depths.  Reflects the time evolution
    of the policy that is approximately stationary away from ``T``.
    """

    p = solution.params
    h = solution.snapshots           # (n_snap, n_q, n_l)
    t_vals = solution.snap_times
    q_vals = p.q_values
    kz = p.kappa[size]

    l_absent, l_present = 0, p.ell
    j_lo, j_hi = p.q_max - plot_to_q, p.q_max + plot_to_q + 1

    # Global y limits across both l levels and both sides.
    ymin, ymax = np.inf, -np.inf
    for l_show in (l_absent, l_present):
        k = int(p.l_index(np.array([l_show]))[0])
        for j in range(j_lo, j_hi):
            if j + size < p.n_q:
                db = 1.0 / kz + (h[:, j, k] - h[:, j + size, k]) / size
                ymin = min(ymin, float(db.min()))
                ymax = max(ymax, float(db.max()))
            if j - size >= 0:
                da = 1.0 / kz + (h[:, j, k] - h[:, j - size, k]) / size
                ymin = min(ymin, float(da.min()))
                ymax = max(ymax, float(da.max()))

    q_slice = q_vals[j_lo:j_hi]
    cmap = LinearSegmentedColormap.from_list(
        'q_cmap', ['indigo', 'darkblue', 'green', 'greenyellow', 'yellow']
    )
    norm = Normalize(vmin=q_slice.min(), vmax=q_slice.max())
    xticks = np.linspace(0, p.T, int(p.T / 100) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    axs = axs.flatten()
    plot_idx = 0

    for l_show, lbl in ((l_absent, 'no liquidity'), (l_present, 'with liquidity')):
        k = int(p.l_index(np.array([l_show]))[0])

        # Bid depths.
        for j in range(j_lo, j_hi):
            if j + size >= p.n_q:
                continue
            db = 1.0 / kz + (h[:, j, k] - h[:, j + size, k]) / size
            axs[plot_idx].plot(t_vals, db, color=cmap(norm(q_vals[j])))
        axs[plot_idx].set_title(rf'$\delta^b$ vs time, {lbl}')
        axs[plot_idx].set_xlabel('t')
        axs[plot_idx].set_ylabel(r'$\delta^b$ (bps)')
        axs[plot_idx].set_ylim(ymin, ymax)
        axs[plot_idx].set_xticks(xticks)
        plot_idx += 1

        # Ask depths.
        for j in range(j_lo, j_hi):
            if j - size < 0:
                continue
            da = 1.0 / kz + (h[:, j, k] - h[:, j - size, k]) / size
            axs[plot_idx].plot(t_vals, da, color=cmap(norm(q_vals[j])))
        axs[plot_idx].set_title(rf'$\delta^a$ vs time, {lbl}')
        axs[plot_idx].set_xlabel('t')
        axs[plot_idx].set_ylabel(r'$\delta^a$ (bps)')
        axs[plot_idx].set_ylim(ymin, ymax)
        axs[plot_idx].set_xticks(xticks)
        plot_idx += 1

    if colbar:
        plt.tight_layout(rect=[0, 0, 0.9, 1.0])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cbar_ax, label=r'$Q_t$')
    else:
        plt.tight_layout(rect=[0, 0, 1, 1.0])

    plt.savefig('deltas_vs_t.pdf' if save_pdf else 'deltas_vs_t.png', bbox_inches='tight')
    plt.show()


def plot_deltas_q(
    solution: HJBSolution,
    time: float = 0.0,
    liquidity: Optional[int] = None,
    plot_to_q: int = 15,
    show_without_liquidity: bool = True,
    show_with_liquidity: bool = True,
    colbar: bool = True,
    save_pdf: bool = False,
    title: str = "",
) -> None:
    """Bid/ask half-spreads vs inventory ``q`` at a fixed time snapshot.

    Solid lines: no internal order (``l = 0``).  Dashed lines: with the internal
    order present (``l = liquidity``).  Ask depths shown positive (red shades),
    bid depths shown negative (blue shades), all sizes ``z ∈ Z`` on one axis.
    The green dashed vertical line marks the execution-region boundary; to its
    left the maker takes the internal order directly and no depths are quoted.
    """

    p = solution.params
    if liquidity is None:
        liquidity = p.ell

    snap_dt = solution.snap_times[1] - solution.snap_times[0]
    t_idx = int(np.clip(round(time / snap_dt), 0, len(solution.snap_times) - 1))
    h = solution.snapshots[t_idx]   # (n_q, n_l)

    q_vals = p.q_values
    l_idx = int(p.l_index(np.array([liquidity]))[0])
    l_absent = 0

    region = execution_region_at(solution, time)
    boundary = execution_boundary_at(solution, time, liquidity)

    ask_cmap = LinearSegmentedColormap.from_list('z_cmap', ['xkcd:tomato red', 'xkcd:orangeish'])
    bid_cmap = LinearSegmentedColormap.from_list('z_cmap', ['xkcd:true blue', 'xkcd:azure'])
    norm = Normalize(vmin=min(p.sizes), vmax=max(p.sizes))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    max_delta = -np.inf
    nq = p.n_q
    q_min_vis, q_max_vis = float(-plot_to_q), float(plot_to_q)

    l_levels: list = []
    if show_without_liquidity:
        l_levels.append(l_absent)
    if show_with_liquidity:
        l_levels.append(liquidity)
    both = show_without_liquidity and show_with_liquidity

    for z in p.sizes:
        for l_show in l_levels:
            k = int(p.l_index(np.array([l_show]))[0])
            ls = '--' if (l_show == liquidity and both) else '-'

            # delta_a plotted at q_vals[z:]; delta_b plotted at q_vals[:-z].
            delta_a = 1.0 / p.kappa[z] + (h[z:, k] - h[:-z, k]) / z
            delta_b = 1.0 / p.kappa[z] + (h[:-z, k] - h[z:, k]) / z

            # Max depth within the visible q window (before NaN-masking).
            vis_a = (q_vals[z:] >= q_min_vis) & (q_vals[z:] <= q_max_vis)
            vis_b = (q_vals[:-z] >= q_min_vis) & (q_vals[:-z] <= q_max_vis)
            if vis_a.any():
                max_delta = max(float(np.nanmax(delta_a[vis_a])), max_delta)
            if vis_b.any():
                max_delta = max(float(np.nanmax(delta_b[vis_b])), max_delta)

            # Blank depths inside the execution region (with-liquidity case only).
            if l_show == liquidity:
                for iq in range(nq):
                    if region[iq, l_idx]:
                        if iq >= z:
                            delta_a[iq - z] = np.nan
                        if iq < nq - z:
                            delta_b[iq] = np.nan

            ax.plot(q_vals[z:], delta_a, color=ask_cmap(norm(z)), linestyle=ls,
                    label=f'Ask z={z}')
            ax.plot(q_vals[:-z], -delta_b, color=bid_cmap(norm(z)), linestyle=ls,
                    label=f'Bid z={z}')

    # Execution-region boundary: the vertical line sits at the first q where the
    # maker quotes rather than takes (one step right of the last take-inventory).
    if np.isfinite(boundary):
        ax.axvline(x=boundary + 1, color='xkcd:apple green', linestyle='--', linewidth=1.2)

    if title:
        ax.set_title(title, fontsize=23)
    ax.set_xlabel('q', fontsize=20)
    ax.set_ylabel(r'$-\delta^b$ (blue), $\delta^a$ (red), in bps', fontsize=20)
    if max_delta > 0:
        ax.set_ylim(-max_delta, max_delta)
    ax.set_xlim(q_min_vis, q_max_vis)

    if colbar:
        bid_sm = plt.cm.ScalarMappable(cmap=bid_cmap, norm=norm)
        bid_sm.set_array([])
        ask_sm = plt.cm.ScalarMappable(cmap=ask_cmap, norm=norm)
        ask_sm.set_array([])
        bid_cbar_ax = fig.add_axes([0.905, 0.15, 0.01, 0.7])
        ask_cbar_ax = fig.add_axes([0.920, 0.15, 0.01, 0.7])
        cbar_a = fig.colorbar(ask_sm, cax=ask_cbar_ax)
        fig.colorbar(bid_sm, cax=bid_cbar_ax, ticks=[])
        cbar_a.set_label(r'Size ($z$)', fontsize=20)

    fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.9)
    plt.savefig('deltas_vs_q.pdf' if save_pdf else 'deltas_vs_q.png', bbox_inches='tight')
    plt.show()


def plot_pnl_hist(
    sim_opt: SimResult,
    sim_nai: SimResult,
    bins: int = 50,
    pnl_scale: float = 1.0,
    opt_colour: str = 'xkcd:azure',
    naive_colour: str = 'xkcd:orange',
    opacity: float = 0.25,
    lw: float = 1.5,
    title: str = "P&L distribution",
    save_pdf: bool = False,
) -> None:
    """Overlaid terminal PnL histograms: optimal vs naive benchmark.

    Parameters
    ----------
    pnl_scale:
        Multiply all PnL values before plotting (e.g. 1000 for K$).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    pnl_opt = sim_opt.pnl * pnl_scale
    pnl_nai = sim_nai.pnl * pnl_scale

    combined = np.concatenate([pnl_opt, pnl_nai])
    bin_edges = np.linspace(combined.min(), combined.max(), bins + 1)

    ax.hist(pnl_nai, bins=bin_edges, color=naive_colour, label='Naive', histtype='step', linewidth=lw)
    ax.hist(pnl_opt, bins=bin_edges, color=opt_colour,   label='Optimal', histtype='step', linewidth=lw)
    ax.hist(pnl_nai, bins=bin_edges, color=naive_colour, alpha=opacity)
    ax.hist(pnl_opt, bins=bin_edges, color=opt_colour,   alpha=opacity)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('PNLs.pdf' if save_pdf else 'PNLs.png', bbox_inches='tight')
    plt.show()


def plot_sample_paths(
    sim_opt: SimResult,
    sim_nai: SimResult,
    plot_start: int = 0,
    plot_num: int = 1,
    pnl_scale: float = 1.0,
    vline_opacity: float = 0.3,
    save_pdf: bool = False,
) -> None:
    """2 × 3 grid of sample paths: PnL, inventory ``Q``, and internal-order size ``L``.

    Rows: optimal strategy (top) and naive benchmark (bottom).
    Columns: running mark-to-market PnL, inventory ``Q``, and internal-order size ``L``.
    Green dashed vertical lines mark times of internal-exchange trades.
    For ``plot_num = 1``, each variable gets a fixed colour; for multiple paths
    each path shares one automatic colour across all three columns.

    Parameters
    ----------
    pnl_scale:
        Multiply PnL values before plotting (e.g. 1000 for K$).
    """
    p = sim_opt.params
    times = sim_opt.times

    pnl_opt = (sim_opt.X + sim_opt.Q * sim_opt.S) * pnl_scale
    pnl_nai = (sim_nai.X + sim_nai.Q * sim_nai.S) * pnl_scale

    sel = slice(plot_start, plot_start + plot_num)
    ymin_pnl = min(float(pnl_opt[sel].min()), float(pnl_nai[sel].min()))
    ymax_pnl = max(float(pnl_opt[sel].max()), float(pnl_nai[sel].max()))
    ymin_q   = min(int(sim_opt.Q[sel].min()),  int(sim_nai.Q[sel].min()))
    ymax_q   = max(int(sim_opt.Q[sel].max()),  int(sim_nai.Q[sel].max()))

    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharey='col')
    single = (plot_num == 1)

    for n in range(plot_start, plot_start + plot_num):
        exec_opt = sim_opt.internal_trade_times(n)
        exec_nai = sim_nai.internal_trade_times(n)

        if single:
            ax[0, 0].plot(times, pnl_opt[n], color='xkcd:blue',        label='P&L')
            ax[0, 1].plot(times, sim_opt.Q[n], color='xkcd:red',       label='Position')
            ax[0, 2].plot(times, sim_opt.L[n], color='xkcd:apple green', label='IX Liquidity')
            ax[1, 0].plot(times, pnl_nai[n], color='xkcd:blue',        label='P&L')
            ax[1, 1].plot(times, sim_nai.Q[n], color='xkcd:red',       label='Position')
            ax[1, 2].plot(times, sim_nai.L[n], color='xkcd:apple green', label='IX Liquidity')
            vline_col = 'xkcd:apple green'
        else:
            (pl,) = ax[0, 0].plot(times, pnl_opt[n])
            color = pl.get_color()
            ax[0, 1].plot(times, sim_opt.Q[n], color=color)
            ax[0, 2].plot(times, sim_opt.L[n], color=color)
            ax[1, 0].plot(times, pnl_nai[n],   color=color)
            ax[1, 1].plot(times, sim_nai.Q[n], color=color)
            ax[1, 2].plot(times, sim_nai.L[n], color=color)
            vline_col = color

        for col, (ym, yM) in enumerate([(ymin_pnl, ymax_pnl), (ymin_q, ymax_q), (0, p.ell)]):
            ax[0, col].vlines(exec_opt, ymin=ym, ymax=yM, colors=vline_col, linestyles='--', alpha=vline_opacity, linewidth=1.0)
            ax[1, col].vlines(exec_nai, ymin=ym, ymax=yM, colors=vline_col, linestyles='--', alpha=vline_opacity, linewidth=1.0)

    for row in range(2):
        for col in range(3):
            ax[row, col].tick_params(axis='both', labelsize=13)

    ax[0, 0].set_ylabel('Optimal')
    ax[1, 0].set_ylabel('Naive Benchmark')
    ax[0, 0].set_title('P&L')
    ax[0, 1].set_title(r'Position $Q$')
    ax[0, 2].set_title(r'Liquidity $L$')
    ax[0, 2].set_ylim(ymin=-0.01, ymax=p.ell * 1.1)
    ax[1, 2].set_ylim(ymin=-0.01, ymax=p.ell * 1.1)
    ax[0, 2].yaxis.set_ticks(list(range(p.ell + 1)))
    ax[1, 2].yaxis.set_ticks(list(range(p.ell + 1)))
    ax[1, 1].set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig('example.pdf' if save_pdf else 'example.png', bbox_inches='tight')
    plt.show()


def plot_price_paths(
    sim_opt: SimResult,
    sim_nai: SimResult,
    policy_opt,
    policy_nai,
    path: int = 0,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    depth_scale: float = 1.0,
    price_scale: float = 1e-4,
    opacity_factor: float = 0.4,
    vline_opacity: float = 0.4,
    save_pdf: bool = False,
) -> None:
    """4-panel plot: ask/bid price ladder + ``L`` indicator strip, for one sample path.

    Layout (height ratios 20:0.5:20:0.5):
      row 0 — optimal: mid price S plus S ± δ^{a/b} for each size z
      row 1 — ``L`` strip (optimal): green line when internal order is present
      row 2 — naive benchmark: same layout
      row 3 — ``L`` strip (naive)

    Green dashed vertical lines mark times of internal-exchange trades.

    Parameters
    ----------
    price_scale:
        Multiply all prices before plotting (default 1e-4 converts bps → $).
    depth_scale:
        Divide depths by this before adding to ``S`` (default 1.0 assumes ``S`` and δ
        share the same units, i.e. both in bps).
    opacity_factor:
        Larger-size price curves are more transparent: ``alpha = z^{-opacity_factor}``.
    """
    p = sim_opt.params
    times = sim_opt.times
    if t_end is None:
        t_end = p.T

    start = int(np.searchsorted(times, t_start))
    end   = int(np.searchsorted(times, t_end, side='right'))
    t_slice = times[start:end]

    Q_opt = sim_opt.Q[path]; L_opt = sim_opt.L[path]; S_opt = sim_opt.S[path]
    Q_nai = sim_nai.Q[path]; L_nai = sim_nai.L[path]; S_nai = sim_nai.S[path]

    db_opt, da_opt = policy_opt.quote_depths(0.0, Q_opt, L_opt)
    db_nai, da_nai = policy_nai.quote_depths(0.0, Q_nai, L_nai)

    def _price(S, delta, sign):
        valid = delta < NO_QUOTE / 2
        return np.where(valid, (S + sign * delta / depth_scale) * price_scale, np.nan)

    # Global y limits over the time window.
    ymin, ymax = np.inf, -np.inf
    for z in p.sizes:
        for S, da, db in [(S_opt, da_opt, db_opt), (S_nai, da_nai, db_nai)]:
            Sa = _price(S, da[z], +1)[start:end]
            Sb = _price(S, db[z], -1)[start:end]
            if np.any(np.isfinite(Sa)): ymin = min(ymin, float(np.nanmin(Sb))); ymax = max(ymax, float(np.nanmax(Sa)))

    fig, ax = plt.subplots(4, 1, figsize=(15, 9),
                           height_ratios=[20, 0.5, 20, 0.5], sharex=True)

    for z in p.sizes:
        alpha = float(z ** (-opacity_factor))
        a_lbl = r'$S^a$' if z == p.sizes[0] else None
        b_lbl = r'$S^b$' if z == p.sizes[0] else None
        ax[0].plot(t_slice, _price(S_opt, da_opt[z], +1)[start:end], color='red',       linewidth=0.5, alpha=alpha, label=a_lbl)
        ax[0].plot(t_slice, _price(S_opt, db_opt[z], -1)[start:end], color='royalblue', linewidth=0.5, alpha=alpha, label=b_lbl)
        ax[2].plot(t_slice, _price(S_nai, da_nai[z], +1)[start:end], color='red',       linewidth=0.5, alpha=alpha, label=a_lbl)
        ax[2].plot(t_slice, _price(S_nai, db_nai[z], -1)[start:end], color='royalblue', linewidth=0.5, alpha=alpha, label=b_lbl)

    ax[0].plot(t_slice, S_opt[start:end] * price_scale, color='black', linewidth=0.5, label=r'$S$')
    ax[2].plot(t_slice, S_nai[start:end] * price_scale, color='black', linewidth=0.5, label=r'$S$')

    L_na_opt = np.where(L_opt == 0, np.nan, L_opt.astype(float))
    L_na_nai = np.where(L_nai == 0, np.nan, L_nai.astype(float))
    ax[1].plot(t_slice, L_na_opt[start:end], linewidth=1.75, color='xkcd:apple green', alpha=0.6)
    ax[3].plot(t_slice, L_na_nai[start:end], linewidth=1.75, color='xkcd:apple green', alpha=0.6)

    exec_opt = sim_opt.internal_trade_times(path)
    exec_nai = sim_nai.internal_trade_times(path)
    win_opt = exec_opt[(exec_opt >= times[start]) & (exec_opt <= times[end - 1])]
    win_nai = exec_nai[(exec_nai >= times[start]) & (exec_nai <= times[end - 1])]
    if win_opt.size:
        ax[0].vlines(win_opt, ymin=ymin, ymax=ymax, colors='xkcd:apple green',
                     linestyles='--', linewidth=1.0, alpha=vline_opacity)
    if win_nai.size:
        ax[2].vlines(win_nai, ymin=ymin, ymax=ymax, colors='xkcd:apple green',
                     linestyles='--', linewidth=1.0, alpha=vline_opacity)

    ax[0].set_title('Prices ($)')
    ax[0].set_ylabel('Optimal')
    ax[0].legend(loc='upper left', fontsize=9)
    ax[0].set_ylim(ymin, ymax)
    ax[0].ticklabel_format(useOffset=False, style='plain', axis='y')
    ax[0].xaxis.set_ticklabels([])

    ax[1].yaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[1].grid(False)
    ax[1].set_ylabel(r'$L$', rotation=0, labelpad=12)

    ax[2].set_ylabel('Naive Benchmark')
    ax[2].legend(loc='upper left', fontsize=9)
    ax[2].set_ylim(ymin, ymax)
    ax[2].ticklabel_format(useOffset=False, style='plain', axis='y')
    ax[2].xaxis.set_ticklabels([])

    ax[3].yaxis.set_visible(False)
    ax[3].grid(False)
    ax[3].set_ylabel(r'$L$', rotation=0, labelpad=12)
    ax[3].set_xlabel('Time (seconds)')

    plt.tight_layout()
    plt.savefig('example_prices.pdf' if save_pdf else 'example_prices.png', bbox_inches='tight')
    plt.show()


def plot_execution_boundary(
    solutions: list,
    param_values=None,
    param_name: str = "",
    label: str = "",
    solutions_2: list = None,
    label_2: str = "",
    liquidity: Optional[int] = None,
    time: float = 0.0,
    title: str = "",
    save_pdf: bool = False,
) -> None:
    """Execution-region boundary (highest q to take internal order) vs a varying parameter.

    Parameters
    ----------
    solutions:
        List of :class:`HJBSolution` objects, one per x-axis point.
    param_values:
        x-axis values; defaults to ``[1, 2, ...]``.
    param_name:
        x-axis label.  Special values ``'rho'`` / ``'rho_tilde'`` and ``'psi'``
        get LaTeX labels and an auto title.
    label, label_2:
        Legend labels for the first and (optional) second series.
    solutions_2:
        Optional second list of solutions plotted in saffron.
    liquidity:
        Internal order level passed to :func:`execution_boundary_at`; defaults
        to ``params.ell`` of each solution.
    time:
        Time snapshot at which to evaluate the boundary (default ``0.0``).
    """
    boundaries = [execution_boundary_at(s, time=time, liquidity=liquidity) for s in solutions]

    if param_values is None:
        param_values = list(range(1, len(solutions) + 1))

    if not title:
        if param_name in ('rho', 'rho_tilde'):
            title = r'Position to take liquidity vs client price'
        elif param_name == 'psi':
            title = r'Position to take liquidity vs $\psi$'
        elif param_name:
            title = f'Position to take liquidity vs {param_name}'
        else:
            title = 'Position to take liquidity'

    if param_name in ('rho', 'rho_tilde'):
        x_label = r'Client price offset $\tilde{\rho}$ (bps)'
    elif param_name == 'psi':
        x_label = r'$\psi$'
    elif param_name:
        x_label = param_name
    else:
        x_label = 'Parameter set index'

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6))
    ax.plot(param_values, boundaries, label=label or None, color='xkcd:deep sky blue')

    if solutions_2:
        boundaries_2 = [execution_boundary_at(s, time=time, liquidity=liquidity) for s in solutions_2]
        ax.plot(param_values, boundaries_2, label=label_2 or None, color='xkcd:saffron')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'Boundary inventory $q^*$')

    if label or label_2:
        ax.legend()

    plt.tight_layout()
    plt.savefig('execution_boundaries.pdf' if save_pdf else 'execution_boundaries.png', bbox_inches='tight')
    plt.show()


def plot_skew_diff(
    solutions: list,
    size: int,
    liquidity: Optional[int] = None,
    param_values=None,
    param_name: str = "",
    plot_to_q: int = 8,
    time: float = 0.0,
    save_pdf: bool = False,
) -> None:
    """Pricing adjustment due to internal liquidity: δ(q,l=ℓ) − δ(q,l=0).

    Plots the ask-side difference (red shades) and bid-side difference (blue
    shades) vs inventory ``q`` for each solution, coloured by ``param_values``.

    Parameters
    ----------
    solutions:
        List of :class:`HJBSolution` objects, one per x-axis value.
    size:
        OTC size ``z`` at which to compute the depth adjustment.
    liquidity:
        Internal order level for the "present" column; defaults to ``ell``.
    param_values:
        Values used to colour the curves and label the colour bar.
    param_name:
        Used to select LaTeX labels.  ``'rho'`` / ``'rho_tilde'`` and
        ``'psi'`` get special treatment; anything else is used verbatim.
    """
    p0 = solutions[0].params
    if liquidity is None:
        liquidity = p0.ell
    q_vals = p0.q_values
    nq = p0.n_q
    j_lo = p0.q_max - plot_to_q
    j_hi = p0.q_max + plot_to_q + 1

    # Compute skew diffs: let dh[iq] = h[iq, l_pres] - h[iq, l_abs].
    # ask diff at iq: (dh[iq] - dh[iq-z]) / z   (valid for iq >= z)
    # bid diff at iq: (dh[iq] - dh[iq+z]) / z   (valid for iq < nq-z)
    skew_a_list, skew_b_list = [], []
    for sol in solutions:
        p = sol.params
        h = sol.h_at(time)
        k_pres = int(p.l_index(np.array([liquidity]))[0])
        k_abs  = int(p.l_index(np.array([0]))[0])
        dh = h[:, k_pres] - h[:, k_abs]
        sa = np.full(nq, np.nan)
        sa[size:] = (dh[size:] - dh[:nq - size]) / size
        sb = np.full(nq, np.nan)
        sb[:nq - size] = (dh[:nq - size] - dh[size:]) / size
        skew_a_list.append(sa)
        skew_b_list.append(sb)

    if param_values is None:
        param_values = list(range(1, len(solutions) + 1))

    norm = Normalize(vmin=min(param_values), vmax=max(param_values))
    cmap_a = LinearSegmentedColormap.from_list('cmap_a', ['xkcd:dandelion', 'red'])
    cmap_b = LinearSegmentedColormap.from_list('cmap_b', ['paleturquoise', 'darkblue'])

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6))

    for i, (sa, sb) in enumerate(zip(skew_a_list, skew_b_list)):
        col_a = cmap_a(norm(param_values[i]))
        col_b = cmap_b(norm(param_values[i]))
        ax.plot(q_vals, sa, color=col_a)
        ax.plot(q_vals, sb, color=col_b)

    # Colour bars: ask (right) + bid (ticks suppressed, left of ask)
    sm_a = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm)
    sm_a.set_array([])
    sm_b = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm)
    sm_b.set_array([])
    cbar_ax_a = fig.add_axes([0.920, 0.15, 0.01, 0.6])
    cbar_ax_b = fig.add_axes([0.905, 0.15, 0.01, 0.6])
    if param_name in ('rho', 'rho_tilde'):
        cbar_label = r'$\tilde{\rho}$, red: ask, blue: bid'
    elif param_name == 'psi':
        cbar_label = r'$\psi$, red: ask, blue: bid'
    elif param_name:
        cbar_label = f'{param_name}, red: ask, blue: bid'
    else:
        cbar_label = 'Parameter set index, red: ask, blue: bid'
    fig.colorbar(sm_a, cax=cbar_ax_a, label=cbar_label)
    fig.colorbar(sm_b, cax=cbar_ax_b, ticks=[])

    ax.set_xlim(-plot_to_q, plot_to_q)
    if size in p0.kappa and param_name != 'kappa':
        lim = 0.5 / p0.kappa[size]
        ax.set_ylim(-lim, lim)
    ax.set_ylabel('Price adjustment (bps)')
    ax.set_xlabel(r'$q$')
    ax.set_title(f'Size={size} pricing adjustment due to client liquidity')

    fig.subplots_adjust(right=0.88)
    plt.savefig('price_adjustment.pdf' if save_pdf else 'price_adjustment.png', bbox_inches='tight')
    plt.show()


def plot_sample_paths_2col(
    sim: SimResult,
    plot_start: int = 0,
    plot_num: int = 1,
    pnl_scale: float = 1.0,
    vline_opacity: float = 0.3,
    label: str = '',
    save_pdf: bool = False,
) -> None:
    """Single-row, two-panel sample-path plot: P&L and inventory ``Q``.

    Styled to match :func:`plot_sample_paths` (colours, tick sizes, vlines)
    but for a single strategy with no ``L`` panel.

    Parameters
    ----------
    sim:
        Simulation result to plot.
    label:
        Row label shown as the y-axis label of the left panel.
    pnl_scale:
        Multiply PnL before plotting (e.g. ``100`` for dollars).
    """
    times = sim.times
    pnl   = (sim.X + sim.Q * sim.S) * pnl_scale
    sel   = slice(plot_start, plot_start + plot_num)

    ymin_pnl = float(pnl[sel].min())
    ymax_pnl = float(pnl[sel].max())
    ymin_q   = int(sim.Q[sel].min())
    ymax_q   = int(sim.Q[sel].max())

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))

    single = (plot_num == 1)

    for n in range(plot_start, plot_start + plot_num):
        exec_t = sim.internal_trade_times(n)
        if single:
            ax[0].plot(times, pnl[n],   color='xkcd:blue')
            ax[1].plot(times, sim.Q[n], color='xkcd:red')
            vcol = 'xkcd:apple green'
        else:
            (ln,) = ax[0].plot(times, pnl[n])
            vcol  = ln.get_color()
            ax[1].plot(times, sim.Q[n], color=vcol)

        ax[0].vlines(exec_t, ymin=ymin_pnl, ymax=ymax_pnl,
                     colors=vcol, linestyles='--', alpha=vline_opacity, linewidth=1.0)
        ax[1].vlines(exec_t, ymin=ymin_q,   ymax=ymax_q,
                     colors=vcol, linestyles='--', alpha=vline_opacity, linewidth=1.0)

    for a in ax:
        a.tick_params(axis='both', labelsize=13)

    if label:
        ax[0].set_ylabel(label)
    ax[0].set_title('P&L', fontsize=16)
    ax[1].set_title(r'Position $Q$', fontsize=16)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.supxlabel('Time (seconds)')

    plt.savefig('example_as.pdf' if save_pdf else 'example_as.png', bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Convenience driver
# ---------------------------------------------------------------------------
def solve_and_simulate(
    p: ModelParams,
    n_paths: int = 5000,
    sim_dt: float = 0.3,
    pde_dt: float = 0.02,
    stationary: bool = True,
    include_benchmark: bool = True,
    iota: float = 0.1,
    seed: int = 0,
) -> dict:
    """Solve the HJBQVI and AS benchmark, then simulate both strategies.

    Returns a dict with ``solution``, ``optimal`` (SimResult), and -- if
    ``include_benchmark`` -- ``h_as`` and ``naive`` (SimResult).
    """
    sol = solve_hjbqvi(p, dt=pde_dt, snap_dt=sim_dt)
    opt_policy = OptimalPolicy(sol, stationary=stationary)
    out = {
        "solution": sol,
        "optimal": simulate(p, opt_policy, n_paths=n_paths, dt=sim_dt, seed=seed),
    }
    if include_benchmark:
        h_as = solve_avellaneda_stoikov(p, dt=pde_dt)
        naive_policy = NaivePolicy(p, h_as, iota=iota)
        out["h_as"] = h_as
        out["naive"] = simulate(p, naive_policy, n_paths=n_paths, dt=sim_dt, seed=seed)
    return out
