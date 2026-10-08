#!/usr/bin/env python
"""
staircase.py -- delta_phi(t) along the whole evolution (Sec. "Numerical
verification", remarks on the branch of the phase).

On the chiral locus, delta_phi(t) = phi_g(t) - phi_u(t) takes only the values
0 and +-pi at every instant; +-pi appears only briefly, when the open and
closed overlaps cross zero at slightly different times (0% of the time for
N=2, 0.07% for N=3).  Off the locus delta_phi(t) is continuous.

Evolution in the symmetric-sector block, with exp(-i h t) (closed) and
exp(-i (h - i Gamma) t) (no-jump), over 6 pi / Omega.  Two branch conventions:
  acc : arg z lifted continuously; on the locus z is real and crosses zero,
        and each sign change of Re z counts as a jump of pi (same for both
        branches).
  wrap: principal value, delta_phi wrapped to (-pi, pi].

Run:  python staircase.py        (a few seconds)
"""
import numpy as np
from numpy.linalg import eigh, norm
from scipy.linalg import expm


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


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def dphi_vs_t(N, n, dD, M=6000, gam=0.1, p=0.05, k=0.25, J=-0.15):
    """delta_phi(t) in the 'acc' and 'wrap' conventions."""
    chi = 2 * (k - J)
    d = min(N, n) + 1
    Hl = block(N, n, chi * (2 * n - N), chi, k, J)
    hl = Hl - np.trace(Hl) / d * np.eye(d)
    H = block(N, n, chi * (2 * n - N) + dD, chi, k, J)
    h = H - np.trace(H) / d * np.eye(d)
    w, V = eigh(hl)
    i = int(np.argmax(w))
    vp, Om = V[:, i], w[i]
    P = parity(d)
    psi0 = (vp + P @ vp) / norm(vp + P @ vp)     # balanced state of the locus
    T = 6 * np.pi / Om
    out = {}
    for lbl, Heff in [("closed", h), ("open", h - 1j * gamma_op(N, n, gam, p))]:
        U = expm(-1j * Heff * (T / M))
        v = psi0.astype(complex).copy()
        zs, conn = [1.0 + 0j], [0.0]
        for _ in range(M):
            vn = U @ v
            zs.append(np.vdot(psi0, vn))
            conn.append(conn[-1] + np.angle(np.vdot(v, vn)))   # Im int <psi|dpsi>
            v = vn
        zs, conn = np.array(zs), np.array(conn)
        mag = np.abs(zs)
        ok = mag > 1e-6
        real = np.max(np.abs(zs.imag[ok]) / mag[ok]) < 1e-9
        pan = np.zeros(len(zs))
        for s in range(1, len(zs)):
            if real:
                pan[s] = pan[s - 1] + (np.pi if zs[s].real * zs[s - 1].real < 0 else 0.0)
            else:
                pan[s] = pan[s - 1] + np.angle(zs[s] * np.conj(zs[s - 1]))
        out[lbl] = dict(acc=pan - conn, wrap=wrap(np.angle(zs) - conn))
    return (out["open"]["acc"] - out["closed"]["acc"],
            wrap(out["open"]["wrap"] - out["closed"]["wrap"]))


if __name__ == "__main__":
    print(f"  {'case':19s} {'conv':5s} {'max|dphi|/pi':>13s} {'time |dphi|>pi/2':>17s}"
          f" {'median|dphi|/pi':>16s}  only 0, +-pi?")
    for N, n in [(2, 2), (3, 3), (4, 2)]:
        for dD in [0.0, 0.05, 0.3]:
            da, dw = dphi_vs_t(N, n, dD)
            for cname, dd in [("acc", da), ("wrap", dw)]:
                frac = np.mean(np.abs(dd) > 0.5 * np.pi) * 100
                near = np.max(np.abs(dd / np.pi - np.round(dd / np.pi)))
                print(f"  N={N} n={n} dD={dD:4.2f}   {cname:5s} {np.max(np.abs(dd)) / np.pi:13.4f}"
                      f" {frac:16.2f}% {np.median(np.abs(dd)) / np.pi:16.2e}  "
                      f"{'yes' if near < 1e-6 else 'no (continuous)'}")
