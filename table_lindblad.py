#!/usr/bin/env python
"""
table_lindblad.py -- Table II of the paper (full Lindblad simulations).

Conventions as in make_figures.py: H_eff = H - i Gamma,
Gamma = (gamma/2) a^dag a + (p/2) sum sigma_+ sigma_-.  Units g = 1.
Default parameters: gamma = 0.1, p = 0.05, k = 0.25, J = -0.15,
chi = 2(k-J), Delta = chi(2n-N) (chiral locus).

The full master equation is integrated with qutip.mesolve (no block reduction).
The pointer state is the eigenvector of rho(t) followed by continuity (maximum
overlap with the previous time step), in the parallel-transport gauge anchored
at psi_+(0) = psi_0.  In that gauge arg z = phi_g, with z = <psi_0|psi_+(t)>.

Top block:    max_t |Im z| / |z|          (locus, dDelta = 0.05, 0.3, chi x 1.3)
Bottom block: delta_phi / pi = (phi_g - phi_u)/pi at Omega t / pi = 3.6

Run:  python table_lindblad.py        (about a minute)
"""
import itertools
import numpy as np
from numpy.linalg import eigh, norm
from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, basis,
                   mesolve, sesolve, Qobj)

K, J = 0.25, -0.15
GAM, P = 0.1, 0.05
OPTS = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 200000}


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


def balanced_state(N, n, k=K, J=J):
    """Balanced state of the highest doublet on the locus, and its Omega."""
    chi = 2 * (k - J)
    d = min(N, n) + 1
    H = block(N, n, chi * (2 * n - N), chi, k, J)
    h = H - np.mean(np.diag(H)) * np.eye(d)
    w, V = eigh(h)
    i = int(np.argmax(w))
    vp = V[:, i]
    vm = np.diag([(-1.0) ** m for m in range(d)]) @ vp
    return (vp + vm) / norm(vp + vm), w[i]


def full_model(N, n, D, chi, k, J, g=1.0, gam=GAM, p=P):
    """Full Hamiltonian and jump operators on (C^2)^N x C^(n+1), and the
    embedding V of the symmetric-sector basis |D^N_m>|n-m>."""
    Nc = n + 1

    def at(op, i):
        ops = [qeye(2)] * N + [qeye(Nc)]
        ops[i] = op
        return tensor(ops)

    a = tensor([qeye(2)] * N + [destroy(Nc)])
    sm = [at(sigmam(), i) for i in range(N)]
    sp = [at(sigmap(), i) for i in range(N)]
    sz = [at(sigmaz(), i) for i in range(N)]
    H = (D / 2) * sum(sz) + chi * (a.dag() * a) ** 2 \
        + g * (a * sum(sp) + a.dag() * sum(sm))
    H += 2 * k * sum(sm[i] * sp[j] + sp[i] * sm[j]
                     for i, j in itertools.combinations(range(N), 2))
    H += J * sum(sz[i] * sz[j] for i, j in itertools.combinations(range(N), 2))
    c_ops = [np.sqrt(gam) * a] + [np.sqrt(p) * sm[i] for i in range(N)]

    e, gr = basis(2, 0), basis(2, 1)
    vecs = []
    for m in range(min(N, n) + 1):
        acc = 0
        for pos in itertools.combinations(range(N), m):
            acc = acc + tensor([e if i in pos else gr for i in range(N)]
                               + [basis(Nc, n - m)])
        vecs.append((acc / acc.norm()).full().ravel())
    return H, c_ops, np.array(vecs).T


def transported(states, kind, ref0):
    """Pointer state in the parallel-transport gauge, anchored at ref0.
    kind='rho': eigenvector of rho followed by continuity (NOT the largest
    eigenvalue: eigenvalues of different N_exc blocks cross exactly).
    kind='ket': normalized state vector."""
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


def initial(N, n, D, chi):
    H, c_ops, V = full_model(N, n, D, chi, K, J)
    z0 = V @ balanced_state(N, n)[0]
    psi0 = Qobj(z0, dims=[H.dims[0], [1] * len(H.dims[0])])
    return H, c_ops, psi0, z0


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- table
def quantization_diagnostic(N, n, dD=0.0, chifac=1.0, nt=801):
    chi0 = 2 * (K - J)
    _, Om = balanced_state(N, n)
    H, c_ops, psi0, z0 = initial(N, n, chi0 * (2 * n - N) + dD, chi0 * chifac)
    ts = np.linspace(0, 4 * np.pi / Om, nt)
    st = mesolve(H, psi0, ts, c_ops=c_ops, options=OPTS).states
    return max(abs(np.vdot(z0, v).imag) / abs(np.vdot(z0, v))
               for v in transported(st, "rho", z0) if abs(np.vdot(z0, v)) > 1e-5)


def delta_phi(N, n, dD=0.0, nt=801):
    chi0 = 2 * (K - J)
    _, Om = balanced_state(N, n)
    H, c_ops, psi0, z0 = initial(N, n, chi0 * (2 * n - N) + dD, chi0)
    ts = np.linspace(0, 3.6 * np.pi / Om, nt)
    pg = transported(mesolve(H, psi0, ts, c_ops=c_ops, options=OPTS).states, "rho", z0)
    pu = transported(sesolve(H, psi0, ts, options=OPTS).states, "ket", z0)
    return wrap(np.angle(np.vdot(z0, pg[-1])) - np.angle(np.vdot(z0, pu[-1]))) / np.pi


if __name__ == "__main__":
    print("max_t |Im z|/|z|       locus     dD=0.05g   dD=0.3g   chi x1.3")
    for N, n in [(2, 2), (3, 3), (3, 1), (4, 2)]:
        row = [quantization_diagnostic(N, n),
               quantization_diagnostic(N, n, dD=0.05),
               quantization_diagnostic(N, n, dD=0.3),
               quantization_diagnostic(N, n, chifac=1.3)]
        print(f"  N={N} n={n}   " + "  ".join(f"{x:9.2e}" for x in row), flush=True)

    print("\ndelta_phi/pi           locus     dD=0.05g   dD=0.3g")
    for N, n in [(2, 2), (3, 3), (4, 2)]:
        row = [delta_phi(N, n), delta_phi(N, n, dD=0.05), delta_phi(N, n, dD=0.3)]
        print(f"  N={N} n={n}   " + "  ".join(f"{x:+9.2e}" for x in row), flush=True)
