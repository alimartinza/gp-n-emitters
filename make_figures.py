#!/usr/bin/env python
"""
make_figures.py -- figures of main_n_emmiters.tex (self-contained).

Conventions (as in the paper): S_a = 1/2 sum sigma_a, H_eff = H - i Gamma,
Gamma = (gamma/2) a^dag a + (p/2) sum sigma_+ sigma_-.  Units g = 1.
Default parameters: gamma = 0.1, p = 0.05, k = 0.25, J = -0.15,
chi = 2(k-J), Delta = chi(2n-N) (chiral locus).

Geometric phase: phi = arg<psi(0)|psi(t)> - Im int <psi|dpsi>   (Mukunda-Simon),
computed gauge-invariantly from step overlaps.

Gauges: loops z(t) = <psi0|psi(t)> are drawn in the no-jump (Schrodinger) gauge,
psi(t) = exp(-i h t) psi0 and exp(-i(h - i Gamma)t) psi0 / norm, in the frame
rotating with the mean energy of the block.  Reality diagnostics |Im z|/|z| are
evaluated in the parallel-transport gauge, where arg z = phi.

Output (in ./figs):
  fig1_chain.pdf          symmetric-sector chains (paper Fig. 1)
  fig2_residual.pdf       ray infidelity scalings (Fig. 2)
  fig3_quantization.pdf   quantization in the full Lindblad dynamics (Fig. 3)
  fig4_ray_deviation.pdf  excursion of the ray toward the zero mode, N=2 (Fig. 4)
  fig5_bargmann.pdf       overlap z(t) in the complex plane, N=2 (Fig. 5)
  extra_site_energies.pdf on-site energies, flatness conditions (not used in the paper)
  extra_initial_states.pdf  sensitivity to the initial state, N=2 (not used in the paper;
                          run explicitly: python make_figures.py extra_initial_states)

Run:  python make_figures.py [name ...]     (no argument: all figures)
"""
import os
import sys
import itertools
import numpy as np
from numpy.linalg import eigh, norm
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
})

CA, CB, CG, CP, CT = "#2166ac", "#b2182b", "#4d4d4d", "#6a51a3", "#2a9d8f"


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  ->", name)


def panel(ax, s, x=0.03, y=0.95, **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, **kw)


# ---------------------------------------------------------------- model
def block(N, n, D, chi, k, J, g=1.0):
    """Symmetric-sector block (S = N/2) in the |phi_m> basis."""
    d = min(N, n) + 1
    m = np.arange(d)
    eps = D * (m - N / 2) + chi * (n - m) ** 2 + 2 * k * m * (N - m) \
        + (J / 2) * ((2 * m - N) ** 2 - N)
    H = np.diag(eps).astype(float)
    for mm in range(d - 1):
        H[mm, mm + 1] = H[mm + 1, mm] = g * np.sqrt((mm + 1) * (N - mm) * (n - mm))
    return H


def gamma_op(N, n, gam, p):
    """Gamma of the paper: (1/2)[gamma (n-m) + p m]."""
    m = np.arange(min(N, n) + 1)
    return 0.5 * np.diag(gam * (n - m) + p * m).astype(float)


def parity(d):
    return np.diag([(-1.0) ** m for m in range(d)])


def setup(N, n, gam=0.1, p=0.05, k=0.25, J=-0.15, g=1.0, dD=0.0):
    chi = 2 * (k - J)
    D = chi * (2 * n - N) + dD
    d = min(N, n) + 1
    H = block(N, n, D, chi, k, J, g)
    h = H - np.mean(np.diag(H)) * np.eye(d)
    return h, h - 1j * gamma_op(N, n, gam, p), parity(d), d


def balanced(h, P):
    """Balanced state of the highest doublet, u_1 = (E+ + P E+)/norm."""
    w, V = eigh(h)
    i = int(np.argmax(w))
    vp = V[:, i]
    return (vp + P @ vp) / norm(vp + P @ vp), w[i]


def traj(M, psi0, ts):
    U = expm(-1j * M * (ts[1] - ts[0]))
    v = psi0.astype(complex).copy()
    out = [v.copy()]
    for _ in ts[1:]:
        v = U @ v
        out.append(v.copy())
    return np.array(out)


def gp_wrapped(psi):
    """phi(t) = arg z(t) - Im int <psi|dpsi>, principal value of arg z."""
    z = psi @ np.conj(psi[0])
    ov = np.sum(np.conj(psi[:-1]) * psi[1:], axis=1)       # <psi_k|psi_k+1>
    conn = np.concatenate([[0.0], np.cumsum(np.angle(ov))])
    return np.angle(z) - conn


def gp_lifted(psi):
    """Same, with arg z lifted continuously in time (valid while z != 0)."""
    z = psi @ np.conj(psi[0])
    ov = np.sum(np.conj(psi[:-1]) * psi[1:], axis=1)
    conn = np.concatenate([[0.0], np.cumsum(np.angle(ov))])
    return np.unwrap(np.angle(z)) - conn


def parallel_transport(psi):
    """Parallel-transport gauge along a sampled path: <psi_k|psi_k+1> real and
    positive, anchored at psi(0).  In this gauge arg<psi(0)|psi(t)> = phi(t)."""
    out = np.array(psi, dtype=complex, copy=True)
    for k in range(1, len(out)):
        ov = np.vdot(out[k - 1], out[k])
        out[k] *= np.exp(-1j * np.angle(ov))
    return out


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ================================================================ Fig. 1
def fig1_chain():
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.3))
    for ax, (N, n) in zip(axes, [(2, 2), (3, 3)]):
        d = min(N, n) + 1
        xs = np.arange(d) * 1.0
        for m in range(d):
            col = CA if m % 2 == 0 else CB
            ax.add_patch(Circle((xs[m], 0), 0.19, color=col, zorder=3))
            ax.text(xs[m], 0, f"{m}", ha="center", va="center", color="w",
                    fontsize=9, zorder=4)
            ax.text(xs[m], -0.30, r"$|D^{%d}_{%d}\rangle|%d\rangle$" % (N, m, n - m),
                    ha="center", va="top", fontsize=8)
        for m in range(d - 1):
            ax.annotate("", xy=(xs[m + 1] - 0.21, 0), xytext=(xs[m] + 0.21, 0),
                        arrowprops=dict(arrowstyle="-", color=CG, lw=1.3))
            ax.text((xs[m] + xs[m + 1]) / 2, 0.07, r"$t_{%d}$" % m,
                    ha="center", va="bottom", fontsize=8.5, color=CG)
        ax.text(-0.75, 0, rf"$N\!=\!{N}$" "\n" rf"$n\!=\!{n}$", ha="right",
                va="center", fontsize=8.5)
        ax.set_xlim(-1.35, 3.45)
        ax.set_ylim(-0.62, 0.36)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.subplots_adjust(hspace=0.05)
    save(fig, "fig1_chain")


# ================================================================ Fig. 2
def extra_site_energies():
    N, n = 5, 5
    k, J = 0.25, -0.15
    chi0 = 2 * (k - J)
    m = np.arange(min(N, n) + 1)
    cases = [
        ("flat", chi0, chi0 * (2 * n - N), "k", "-"),
        (r"$\chi\neq2(k-J)$", 1.30 * chi0, 2 * (1.30 * chi0) * n - 2 * N * (k - J), CB, "--"),
        (r"$\Delta\neq\chi(2n-N)$", chi0, chi0 * (2 * n - N) + 0.6, CA, "-."),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for lbl, chi, D, c, ls in cases:
        eps = D * (m - N / 2) + chi * (n - m) ** 2 + 2 * k * m * (N - m) \
            + (J / 2) * ((2 * m - N) ** 2 - N)
        ax.plot(m, eps - eps.mean(), ls, color=c, lw=1.5, marker="o", ms=4, label=lbl)
    ax.axhline(0, color=CG, lw=0.6)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$(\varepsilon_m-\bar\varepsilon)/g$")
    ax.set_xticks(m)
    ax.legend(loc="upper left")
    save(fig, "extra_site_energies")


# ================================================================ Fig. 3
def fig_residual():
    def dev(N, n, gam=0.1, p=0.05, ncyc=6, nt=1200):
        h, Heff, P, d = setup(N, n, gam, p)
        psi0, Om = balanced(h, P)
        ts = np.linspace(0, ncyc * np.pi / Om, nt)
        PD = traj(Heff, psi0, ts)
        PD = PD / norm(PD, axis=1)[:, None]
        PU = traj(h, psi0, ts)
        return (1 - np.abs(np.sum(PU.conj() * PD, axis=1)) ** 2).max()

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
    dif = np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.12])
    r = [dev(3, 3, gam=0.1, p=0.1 - x) for x in dif]
    axs[0].loglog(dif, r, "o", color=CA, ms=4)
    axs[0].loglog(dif, r[2] * (dif / dif[2]) ** 2, "-", color=CG, lw=1,
                  label=r"$\propto(\gamma-p)^2$")
    axs[0].set_xlabel(r"$(\gamma-p)/g$")
    axs[0].set_ylabel("max. infidelity")
    axs[0].legend(loc="lower right")

    ns = np.array([2, 3, 4, 6, 8, 12, 18, 26])
    r = [dev(2, int(n)) for n in ns]
    axs[1].loglog(ns, r, "o", color=CA, ms=4)
    axs[1].loglog(ns, r[0] * (ns / ns[0]) ** -1.0, "-", color=CG, lw=1,
                  label=r"$\propto 1/n$")
    axs[1].set_xlabel(r"$n$")
    axs[1].set_xticks([2, 5, 10, 20])
    axs[1].set_xticklabels(["2", "5", "10", "20"])
    axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[1].set_ylim(4e-6, 2.5e-4)
    axs[1].legend(loc="lower left")

    Ns = [2, 3, 4, 5, 6, 8, 10, 12, 16]
    axs[2].semilogy(Ns, [dev(N, N) for N in Ns], "o-", color=CA, ms=4, lw=1.2,
                    label=r"$n=N$")
    axs[2].semilogy(Ns, [dev(N, 1) for N in Ns], "s--", color=CB, ms=4, lw=1.2,
                    label=r"$n=1$")
    axs[2].set_xlabel(r"$N$")
    axs[2].set_ylim(5e-6, 5e-4)
    axs[2].legend(loc="lower left")
    for ax, s in zip(axs, ["(a)", "(b)", "(c)"]):
        panel(ax, s, x=0.04, y=0.95)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig2_residual")


# ================================================================ Fig. 4
def fig3_quantization():
    from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, basis,
                       mesolve, sesolve, Qobj)

    def full(N, n, D, chi, k, J, g=1.0, gam=0.1, p=0.05):
        Nc = n + 1

        def at(op, i):
            ops = [qeye(2)] * N + [qeye(Nc)]
            ops[i] = op
            return tensor(ops)
        a = tensor([qeye(2)] * N + [destroy(Nc)])
        sm = [at(sigmam(), i) for i in range(N)]
        sp = [at(sigmap(), i) for i in range(N)]
        sz = [at(sigmaz(), i) for i in range(N)]
        H = (D / 2) * sum(sz) + chi * (a.dag() * a) ** 2 + g * (a * sum(sp) + a.dag() * sum(sm))
        H += 2 * k * sum(sm[i] * sp[j] + sp[i] * sm[j]
                         for i, j in itertools.combinations(range(N), 2))
        H += J * sum(sz[i] * sz[j] for i, j in itertools.combinations(range(N), 2))
        c = [np.sqrt(gam) * a] + [np.sqrt(p) * sm[i] for i in range(N)]
        e, gr = basis(2, 0), basis(2, 1)
        vecs = []
        for mm in range(min(N, n) + 1):
            acc = 0
            for pos in itertools.combinations(range(N), mm):
                acc = acc + tensor([e if i in pos else gr for i in range(N)]
                                   + [basis(Nc, n - mm)])
            vecs.append((acc / acc.norm()).full().ravel())
        return H, c, np.array(vecs).T

    def transported(states, kind, ref0):
        """Pointer state followed by continuity, parallel transport, anchored."""
        out, prev = [], None
        for j, st in enumerate(states):
            if kind == "rho":
                ev, evec = np.linalg.eigh(st.full())
                i = int(np.argmax(ev)) if prev is None else \
                    int(np.argmax(np.abs(prev.conj() @ evec)))
                v = evec[:, i]
            else:
                v = st.full().ravel()
                v = v / norm(v)
            if j == 0:
                ov0 = np.vdot(ref0, v)
                if abs(ov0) > 1e-12:
                    v = v * np.exp(-1j * np.angle(ov0))
            if prev is not None:
                ov = np.vdot(prev, v)
                if abs(ov) > 1e-12:
                    v = v * np.exp(-1j * np.angle(ov))
            prev = v
            out.append(v)
        return out

    N, n = 3, 3
    k, J = 0.25, -0.15
    chi0 = 2 * (k - J)
    h, _, P, d = setup(N, n)
    psi0b, Om = balanced(h, P)
    opts = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 200000}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for lbl, dD, c, ls in [("locus", 0.0, "k", "-"),
                           (r"$\delta\Delta=0.05g$", 0.05, CA, "--"),
                           (r"$\delta\Delta=0.3g$", 0.3, CB, "-.")]:
        H, cops, V = full(N, n, chi0 * (2 * n - N) + dD, chi0, k, J)
        z0 = V @ psi0b
        psi0 = Qobj(z0, dims=[H.dims[0], [1] * len(H.dims[0])])
        ts = np.linspace(0, 4 * np.pi / Om, 401)
        st = mesolve(H, psi0, ts, c_ops=cops, options=opts).states
        y = []
        for v in transported(st, "rho", z0):
            zz = np.vdot(z0, v)
            y.append(abs(zz.imag) / abs(zz) if abs(zz) > 1e-6 else np.nan)
        ax1.semilogy(ts * Om / np.pi, np.maximum(y, 1e-17), ls, color=c, lw=1.2, label=lbl)
    ax1.set_xlabel(r"$\Omega t/\pi$")
    ax1.set_ylabel(r"$|\mathrm{Im}\,z|/|z|$")
    ax1.set_ylim(1e-17, 50)
    ax1.legend(loc="center right")

    dDs = np.linspace(-0.3, 0.3, 21)
    dphis = []
    for dD in dDs:
        H, cops, V = full(N, n, chi0 * (2 * n - N) + dD, chi0, k, J)
        z0 = V @ psi0b
        psi0 = Qobj(z0, dims=[H.dims[0], [1] * len(H.dims[0])])
        ts = np.linspace(0, 3.6 * np.pi / Om, 301)
        pg = transported(mesolve(H, psi0, ts, c_ops=cops, options=opts).states, "rho", z0)
        pu = transported(sesolve(H, psi0, ts, options=opts).states, "ket", z0)
        dd = np.angle(np.vdot(z0, pg[-1])) - np.angle(np.vdot(z0, pu[-1]))
        dphis.append(wrap(dd) / np.pi)
    ax2.plot(dDs, dphis, "o-", color=CA, ms=3.5, lw=1.2)
    ax2.axvline(0, color=CB, lw=1.0, ls=":")
    ax2.axhline(0, color=CG, lw=0.6)
    ax2.set_xlabel(r"$\delta\Delta/g$")
    ax2.set_ylabel(r"$\delta\phi/\pi$")
    panel(ax1, "(a)")
    panel(ax2, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_quantization")


# ======================================================= N = 2 helpers
class TwoEmitters:
    """N = 2, n = 2 block on the chiral line (h is the same on the whole line)."""

    def __init__(self, n=2, gam=0.10, p=0.05, spc=8000):
        g = 1.0
        t0, t1 = g * np.sqrt(2 * n), g * np.sqrt(2 * (n - 1))
        # basis (phi_0, phi_1, phi_2) = (|gg,n>, |T,n-1>, |ee,n-2>)
        self.h = np.array([[0.0, t0, 0.0], [t0, 0.0, t1], [0.0, t1, 0.0]])
        self.G = 0.5 * np.diag([gam * n, gam * (n - 1) + p, gam * (n - 2) + 2 * p])
        self.P = np.diag([1.0, -1.0, 1.0])
        ev, V = eigh(self.h)
        self.ev, self.V = ev, V
        self.Em, self.Z, self.Ep = V[:, 0], V[:, 1], V[:, 2]
        # fix signs as in the paper: E+- = (u +- phi_1)/sqrt2 with u > 0 on phi_0,
        # and Z = (t0 phi_2 - t1 phi_0)/Omega  (positive on phi_2)
        for name, idx in (("Em", 0), ("Ep", 0), ("Z", 2)):
            v = getattr(self, name)
            if v[idx] < 0:
                setattr(self, name, -v)
        self.Om = ev[2]
        self.T = np.pi / self.Om                     # Rabi period of the doublet
        self.W, self.R = np.linalg.eig(self.h - 1j * self.G)
        self.RI = np.linalg.inv(self.R)
        self.spc = spc

    def times(self, ncyc, spc=None):
        spc = spc or self.spc
        return np.linspace(0, ncyc * self.T, int(ncyc * spc) + 1)

    def evolve(self, psi0, ts):
        psi0 = np.asarray(psi0, complex) / norm(psi0)
        pd = (np.exp(-1j * np.outer(ts, self.W)) * (self.RI @ psi0)) @ self.R.T
        pd /= norm(pd, axis=1)[:, None]
        pu = (np.exp(-1j * np.outer(ts, self.ev)) * (self.V.T @ psi0)) @ self.V.T
        return pu, pd

    def no_jump(self, psi0, ts):
        """Unnormalized no-jump ket exp(-i(h - i Gamma)t) psi0 (block barycenter removed)."""
        psi0 = np.asarray(psi0, complex) / norm(psi0)
        return (np.exp(-1j * np.outer(ts, self.W)) * (self.RI @ psi0)) @ self.R.T

    def doublet(self, th):
        return np.cos(th / 2) * self.Ep + np.sin(th / 2) * self.Em

    def zero_mode_family(self, th):
        """Real states on the N_eff=2 circle of the population simplex
        (E-, Z, E+); th = pi/2 is the balanced superposition of E- and Z."""
        e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        e2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
        q = np.clip(np.ones(3) / 3 + np.sqrt(1 / 6) * (np.cos(th) * e1 + np.sin(th) * e2), 0, None)
        v = np.sqrt(q[0]) * self.Em + np.sqrt(q[1]) * self.Z + np.sqrt(q[2]) * self.Ep
        return v / norm(v)


# ================================================================ Fig. 5
def fig5_bargmann():
    s = TwoEmitters()
    ncyc = 6
    ts = s.times(ncyc)
    psi_in = (s.Ep + s.Em) / np.sqrt(2)
    psi_zm = (s.Em + s.Z) / np.sqrt(2)
    psi_unb = s.doublet(0.40 * np.pi)
    cmap = plt.colormaps["viridis"]

    fig = plt.figure(figsize=(3.4, 3.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.12)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    for ax, psi0, lab in [(axA, psi_in, "(a)"), (axB, psi_zm, "(b)")]:
        pu, pd = s.evolve(psi0, ts)
        zu, zd = pu @ np.conj(pu[0]), pd @ np.conj(pd[0])
        ax.plot(zu.real, zu.imag, color="0.6", lw=2.4, zorder=1)
        spc = s.spc
        for c in range(ncyc):
            sl = slice(c * spc, (c + 1) * spc + 1)
            ax.plot(zd[sl].real, zd[sl].imag, lw=1.0,
                    color=cmap(0.06 + 0.86 * c / (ncyc - 1)), zorder=2)
        ax.plot(0, 0, "o", ms=4.5, mfc="w", mec="k", mew=1.1, zorder=6)
        ax.plot(1, 0, "*", ms=8, color="k", zorder=6)
        ax.axhline(0, color="k", lw=0.6, zorder=0)
        ax.axvline(0, color="k", lw=0.6, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.1, 1.15)
        ax.set_ylim(-0.65, 0.65)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-0.5, 0, 0.5])
        ax.set_xlabel(r"$\mathrm{Re}\,z$", labelpad=1)
        panel(ax, lab, x=0.04, y=0.96)
    axA.set_ylabel(r"$\mathrm{Im}\,z$")
    axB.tick_params(labelleft=False)

    for psi0, lab, col, ls in [(psi_in, r"(a)", CA, "-"),
                               (psi_unb, r"$\theta=0.4\pi$", CP, "--"),
                               (psi_zm, r"(b)", CB, "-.")]:
        _pu, pd = s.evolve(psi0, ts)
        pd = parallel_transport(pd)          # (c) is a GP diagnostic: |Im z|/|z| = |sin phi_g|
        zd = pd @ np.conj(pd[0])
        r = np.abs(zd.imag) / np.maximum(np.abs(zd), 1e-300)
        axC.semilogy(ts / s.T, np.maximum(r, 1e-17), ls, color=col, lw=1.1, label=lab)
    axC.set_xlabel(r"$t/T_R$")
    axC.set_ylabel(r"$|\mathrm{Im}\,z_d|/|z_d|$")
    axC.set_xlim(0, ncyc)
    axC.set_ylim(1e-17, 1e10)
    axC.set_yticks([1e-15, 1e-10, 1e-5, 1])
    axC.legend(loc="upper right", ncol=3, fontsize=7.5, handlelength=1.6,
               columnspacing=0.8)
    panel(axC, "(c)", x=0.02, y=0.94)
    save(fig, "fig5_bargmann")


# ================================================================ Fig. 6
def extra_initial_states():
    s = TwoEmitters()

    # ---- left column: doublet family, cos(th/2) E+ + sin(th/2) E-
    ncyc_scan = 6
    ts6 = s.times(ncyc_scan, spc=2000)
    ths_d = np.linspace(0.30, 0.70, 81)
    dphi_d = []
    for thp in ths_d:
        pu, pd = s.evolve(s.doublet(thp * np.pi), ts6)
        dphi_d.append(wrap(gp_wrapped(pd)[-1] - gp_wrapped(pu)[-1]) / np.pi)
    dphi_d = np.array(dphi_d)

    ncyc_long = 20
    tsL = s.times(ncyc_long, spc=2000)
    strob = np.arange(0, len(tsL), 2000)              # t = k T_R
    drift = {}
    for thp in [0.30, 0.40, 0.45, 0.50]:
        pu, pd = s.evolve(s.doublet(thp * np.pi), tsL)
        dd = wrap(gp_wrapped(pd) - gp_wrapped(pu)) / np.pi
        drift[thp] = dd[strob]
        slope = np.polyfit(np.arange(len(strob)), dd[strob], 1)[0]
        print(f"     doublet theta={thp:.2f}pi: slope = {slope:.2e} pi per period")

    # ---- right column: zero-mode family (N_eff = 2 circle)
    ts6f = s.times(ncyc_scan)
    ths_z = np.round(np.arange(0.4600, 0.49951, 0.00025), 6)
    dphi_z = []
    for thp in ths_z:
        pu, pd = s.evolve(s.zero_mode_family(thp * np.pi), ts6f)
        dphi_z.append((gp_lifted(pd)[-1] - gp_lifted(pu)[-1]) / np.pi)
    dphi_z = np.array(dphi_z)
    jumps = np.where(np.abs(np.diff(dphi_z)) > 1.0)[0]
    thc = 0.5 * (ths_z[jumps] + ths_z[jumps + 1])
    print("     theta_c/pi =", np.round(thc, 4), " baseline =",
          round(float(dphi_z[0]), 4), "pi")

    tz = {}
    for thp in [0.470, 0.483, 0.491, 0.498]:
        pu, pd = s.evolve(s.zero_mode_family(thp * np.pi), ts6f)
        tz[thp] = (gp_lifted(pd) - gp_lifted(pu)) / np.pi

    # ---- plot
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.0))
    (a, b), (c, d) = axs

    a.plot(ths_d, dphi_d * 1e5, color=CA, lw=1.5)
    a.axhline(0, color=CG, lw=0.6)
    a.axvline(0.5, color=CG, lw=0.8, ls=":")
    a.plot([0.5], [0.0], "o", color="k", ms=4, zorder=5)
    a.set_xlabel(r"$\theta/\pi$")
    a.set_ylabel(r"$\delta\phi(6T_R)\ [10^{-5}\pi]$")

    b.plot(ths_z, dphi_z, color="k", lw=1.5)
    for tc in thc:
        b.axvline(tc, color="0.6", ls="--", lw=0.8, zorder=0)
    b.axhline(0, color=CG, lw=0.6)
    b.set_xlabel(r"$\theta/\pi$")
    b.set_ylabel(r"$\delta\phi(6T_R)/\pi$")
    b.set_xlim(ths_z[0], ths_z[-1])
    b.set_ylim(-10.0, 2.0)
    b.set_yticks([0, -2, -4, -6])

    cols = [CB, CP, CT, "k"]
    for (thp, y), col in zip(drift.items(), cols):
        c.plot(np.arange(len(y)), y * 1e5, "o-", color=col, ms=2.8, lw=1.0,
               label=rf"${thp:.2f}\pi$")
    c.axhline(0, color=CG, lw=0.6)
    c.set_xlabel(r"$t/T_R$")
    c.set_ylabel(r"$\delta\phi\ [10^{-5}\pi]$")
    c.set_xlim(0, ncyc_long)
    c.set_ylim(-2, 42)
    c.legend(loc="upper left", bbox_to_anchor=(0.07, 0.99), ncol=2,
             title=r"$\theta$", title_fontsize=8)

    for (thp, y), col in zip(tz.items(), cols):
        d.plot(ts6f / s.T, y, color=col, lw=1.2, label=rf"${thp:.3f}\pi$")
    d.axhline(0, color=CG, lw=0.6)
    d.set_xlabel(r"$t/T_R$")
    d.set_ylabel(r"$\delta\phi/\pi$")
    d.set_xlim(0, ncyc_scan)
    d.set_ylim(-10.0, 2.0)
    d.set_yticks([0, -2, -4, -6])
    d.legend(loc="lower center", ncol=4, title=r"$\theta$", title_fontsize=8,
             handlelength=1.4, columnspacing=0.9)

    for ax, sl in zip([a, b, c, d], ["(a)", "(b)", "(c)", "(d)"]):
        panel(ax, sl, x=0.03, y=0.95)
    fig.tight_layout(h_pad=1.2, w_pad=2.0)
    save(fig, "extra_initial_states")


# ================================================================ Fig. 6
def fig4_ray_deviation():
    """The open-system ray leaves the doublet toward the zero mode and returns,
    inside P: doublet populations stay equal, the excursion is bounded."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ---- (a) weights, default parameters
    s = TwoEmitters(spc=4000)
    ncyc = 4
    ts = s.times(ncyc)
    psi0 = (s.Ep + s.Em) / np.sqrt(2)
    raw = s.no_jump(psi0, ts)
    nrm2 = np.sum(np.abs(raw) ** 2, axis=1)
    pd = raw / np.sqrt(nrm2)[:, None]
    cp, cm, cz = pd @ s.Ep, pd @ s.Em, pd @ s.Z
    pu, _ = s.evolve(psi0, ts)
    infid = 1 - np.abs(np.sum(np.conj(pu) * pd, axis=1)) ** 2
    imb = np.abs(np.abs(cp) ** 2 - np.abs(cm) ** 2)
    print(f"     max |c_Z|^2 = {np.max(np.abs(cz)**2):.3e},  max infidelity = {infid.max():.3e},"
          f"  max ||c+|^2-|c-|^2| = {imb.max():.1e},  max |Im c_Z| = {np.max(np.abs(cz.imag)):.1e}")

    fig = plt.figure(figsize=(7.0, 2.7))
    ax = fig.add_subplot(1, 2, 1)
    ax.semilogy(ts / s.T, nrm2, "--", color=CT, lw=1.4, label=r"$\|\tilde\psi\|^2$")
    ax.semilogy(ts / s.T, np.maximum(np.abs(cz) ** 2, 1e-18), color=CB, lw=1.4,
                label=r"$|c_Z|^2$")
    ax.semilogy(ts / s.T, np.maximum(infid, 1e-18), ":", color="k", lw=1.2,
                label=r"$1-F$")
    ax.set_xlabel(r"$t/T_R$")
    ax.set_xlim(0, ncyc)
    ax.set_ylim(1e-9, 1e3)
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.legend(loc="upper right", ncol=3, fontsize=7.5, handlelength=1.6,
              columnspacing=0.8)
    panel(ax, "(a)", x=0.03, y=0.95)

    # ---- (b) geometry, exaggerated delta = 0.5 g
    sx = TwoEmitters(gam=0.55, p=0.05, spc=4000)
    tsx = sx.times(1)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    phi = np.linspace(0, 2 * np.pi, 400)
    ax3.plot(np.cos(phi), np.sin(phi), 0 * phi, color="0.55", lw=2.0, label="unitary")
    for ph, col, ls, lab in [(0.0, CB, "-", r"$\varphi=0$"),
                             (np.pi / 2, CT, "--", r"$\varphi=\pi/2$")]:
        p0 = (sx.Ep + np.exp(1j * ph) * sx.Em) / np.sqrt(2)
        _pu, pdx = sx.evolve(p0, tsx)
        a, b, z = pdx @ sx.Ep, pdx @ sx.Em, pdx @ sx.Z
        w = np.abs(a) ** 2 + np.abs(b) ** 2
        X = 2 * np.real(np.conj(a) * b) / w
        Y = 2 * np.imag(np.conj(a) * b) / w
        ax3.plot(X, Y, np.abs(z), ls, color=col, lw=1.5, label=lab)
    ax3.set_xlabel(r"$x$", labelpad=-8)
    ax3.set_ylabel(r"$y$", labelpad=-8)
    ax3.set_zlabel(r"$|c_Z|$", labelpad=-2)
    ax3.set_xticks([-1, 0, 1]); ax3.set_yticks([-1, 0, 1])
    ax3.tick_params(axis="both", labelsize=7, pad=-2)
    ax3.view_init(elev=22, azim=-60)
    ax3.legend(loc="upper left", bbox_to_anchor=(-0.12, 0.86), fontsize=7.5)
    ax3.text2D(-0.08, 1.0, "(b)", transform=ax3.transAxes, fontsize=10, va="top")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig4_ray_deviation")


FIGS = {
    "fig1_chain": fig1_chain,
    "fig2_residual": fig_residual,
    "fig3_quantization": fig3_quantization,
    "fig4_ray_deviation": fig4_ray_deviation,
    "fig5_bargmann": fig5_bargmann,
}
EXTRA = {"extra_initial_states": extra_initial_states,
         "extra_site_energies": extra_site_energies}   # not in the paper

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for name in names:
        print(name, "...")
        {**FIGS, **EXTRA}[name]()
