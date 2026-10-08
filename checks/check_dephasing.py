#!/usr/bin/env python
"""
check_dephasing.py -- pure dephasing in the N-emitter nonlinear Tavis-Cummings
model.  Accompanies Sec. "Pure dephasing" of NOTES_N_emitters.tex.

Run:  python3 check_dephasing.py      (~12 min)

Dephasing jump operators added to the Lindbladian of the notes:
    individual   sqrt(gph/2)  sz_i        (single-emitter coherence decays at gph)
    collective   sqrt(gphc/2) S_z
    cavity       sqrt(kph)    a^dag a
None of them changes N_exc, so the sector-n block rho_n obeys a CLOSED but no
longer rank-one equation.  Everything is done in the full n-excitation sector
(configuration basis |conf> x |n-m>, all total-spin sectors S), not only the
Dicke chain, except where stated.

Predictions tested
  (T1) Theta(rho) = P rho^* P is a symmetry of the sector-n Liouvillian on the
       chiral locus, dephasing included, EXACTLY if k=0; for k!=0 only on
       permutation-symmetric states (energy offsets of the S sectors differ)
  (T2) GP of the pointer branch quantized in {0,pi} for any gph; rho_n stays
       Theta-invariant; off the locus it is not
  (T3) what dephasing does degrade: branch weight, gap, ray, staircase
  (T4) full Lindblad (qutip, all sectors): rho block diagonal in N_exc; the
       continuously tracked pointer branch stays in sector n; the "largest
       eigenvalue" pointer jumps to n-1 at an exact (decoupled-block) crossing
  (T5) inhomogeneous rates (p_i or gph_i): harmless if k=0, break if k!=0
  (T6) classical frequency noise: static Gaussian (mean 0 / biased / skewed),
       collective vs individual, k=0 vs k!=0
  (T7a) colored noise, EXACT ensemble: random telegraph noise per emitter
       (stochastic Liouville equation), crossover in the correlation time
  (T7b) Monte Carlo cross-check with Ornstein-Uhlenbeck noise
  (T8) single quantum trajectories (sz kicks) stay in Pcal when k=0
  (T9) a dissipator that is NOT of the protected type (complex, unequal
       cross-correlated dephasing) breaks the quantization -- the criterion is
       sharp; equal weights are rescued by Theta o (emitter swap)
  (T10) corollary: thermal jumps at Delta=chi=0, k=J (all sectors chiral)
"""
import itertools
import time
import numpy as np
from numpy.linalg import norm, eigh
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True, linewidth=160)
T0 = time.time()

K0, J0, GAM, P0 = 0.25, -0.15, 0.1, 0.05     # parameters of the notes (g=1)


def hdr(t):
    print("\n" + "=" * 94 + "\n" + t + "\n" + "=" * 94, flush=True)


# ------------------------------------------------------------- sector-n model
def sector_ops(N, n):
    """Operators of the full n-excitation sector in the configuration basis."""
    confs = [c for m in range(min(N, n) + 1) for c in itertools.combinations(range(N), m)]
    idx = {c: i for i, c in enumerate(confs)}
    D = len(confs)
    m = np.array([len(c) for c in confs])
    sz = [np.diag([1.0 if i in c else -1.0 for c in confs]) for i in range(N)]
    Tp = [np.zeros((D, D)) for _ in range(N)]          # a sigma_+^(i)
    for c in confs:
        for i in range(N):
            if i not in c and len(c) < min(N, n):
                Tp[i][idx[tuple(sorted(c + (i,)))], idx[c]] = np.sqrt(n - len(c))
    X = np.zeros((D, D))                                 # sum_{i<j} s-_i s+_j + h.c.
    for c in confs:
        for i, j in itertools.permutations(range(N), 2):
            if i in c and j not in c:
                X[idx[tuple(sorted([x for x in c if x != i] + [j]))], idx[c]] += 1
    V = np.zeros((D, min(N, n) + 1))                     # Dicke vectors (S=N/2)
    for i, c in enumerate(confs):
        V[i, len(c)] = 1.0
    V /= np.sqrt((V ** 2).sum(0))
    return dict(D=D, m=m, sz=sz, Tp=Tp, X=X, P=np.diag((-1.0) ** m), V=V, n=n, N=N)


def hamiltonian(o, D, chi, k, J, gs=None, deltas=None):
    N, n, m = o['N'], o['n'], o['m']
    gs = gs if gs is not None else [1.0] * N
    H = sum(D / 2 * s for s in o['sz']) + np.diag(chi * (n - m) ** 2.0)
    H = H + sum(gs[i] * (o['Tp'][i] + o['Tp'][i].T) for i in range(N))
    H = H + 2 * k * o['X']
    H = H + J * sum(o['sz'][i] @ o['sz'][j] for i, j in itertools.combinations(range(N), 2))
    if deltas is not None:
        H = H + sum(deltas[i] / 2 * o['sz'][i] for i in range(N))
    return H


def liouvillian(Heff, jumps):
    """vec convention: column stacking, vec(A X B) = (B^T kron A) vec(X)."""
    I = np.eye(Heff.shape[0])
    M = -1j * (np.kron(I, Heff) - np.kron(Heff.conj(), I))
    for L in jumps:
        M = M + np.kron(L.conj(), L)
    return M


def chain(N, n, k, J):
    chi = 2 * (k - J)
    d = min(N, n) + 1
    m = np.arange(d)
    H = np.diag(chi * (2 * n - N) * (m - N / 2) + chi * (n - m) ** 2.0
                + 2 * k * m * (N - m) + J / 2 * ((2 * m - N) ** 2 - N))
    for j in range(d - 1):
        H[j, j + 1] = H[j + 1, j] = np.sqrt((j + 1) * (N - j) * (n - j))
    return H


def balanced(N, n, k=K0, J=J0):
    """Balanced state of the doublet with largest Omega (lives in sublattice A)."""
    H = chain(N, n, k, J)
    d = H.shape[0]
    h = H - np.trace(H) / d * np.eye(d)
    P = np.diag((-1.0) ** np.arange(d))
    w, V = eigh(h)
    v = V[:, -1] + P @ V[:, -1]
    return v / norm(v), w[-1]


def generator(o, k=K0, J=J0, gam=GAM, p=P0, gph=0.0, gphc=0.0, kph=0.0, dD=0.0,
              ps=None, gphs=None, deltas=None, extra_jumps=()):
    N, n, m = o['N'], o['n'], o['m']
    chi = 2 * (k - J)
    H = hamiltonian(o, chi * (2 * n - N) + dD, chi, k, J, deltas=deltas)
    ps = ps if ps is not None else [p] * N
    gphs = gphs if gphs is not None else [gph] * N
    nat = [(s.diagonal() + 1) / 2 for s in o['sz']]
    G = np.diag(gam * (n - m) + sum(ps[i] * nat[i] for i in range(N)))
    jumps = [np.sqrt(gphs[i] / 2) * o['sz'][i] for i in range(N) if gphs[i] > 0]
    if gphc > 0:
        jumps.append(np.sqrt(gphc / 2) * sum(o['sz']))
    if kph > 0:
        jumps.append(np.sqrt(kph) * np.diag((n - m).astype(float)))
    jumps += list(extra_jumps)
    G = G + sum((L.conj().T @ L for L in jumps), np.zeros_like(G))
    return H, liouvillian(H - 0.5j * G, jumps)


# ------------------------------------------------------------- pointer branch
def pointer_branch(R, z0):
    """Tong pointer state = the eigenvalue branch that starts at 1, continued
    CONTINUOUSLY (max overlap with the previous step), in the parallel-transport
    gauge anchored to <z0|psi_+(0)> > 0.  Returns z(t)=<z0|psi_+(t)>, the branch
    eigenvalue, the gap to the next eigenvalue, the vectors, and the number of
    times the branch is NOT the largest eigenvalue of rho."""
    prev, zs, lam, gap, vecs, notmax = None, [], [], [], [], 0
    for j, rho in enumerate(R):
        rho = (rho + rho.conj().T) / 2
        w, W = eigh(rho)
        i = int(np.argmax(w)) if j == 0 else int(np.argmax(np.abs(W.conj().T @ prev)))
        v = W[:, i]
        v = v * np.exp(-1j * np.angle(np.vdot(z0 if j == 0 else prev, v)))
        prev = v
        others = np.delete(w, i)
        zs.append(np.vdot(z0, v))
        lam.append(w[i])
        gap.append(w[i] - (others.max() if len(others) else 0.0))
        vecs.append(v)
        notmax += int(i != int(np.argmax(w)))
    return np.array(zs), np.array(lam), np.array(gap), np.array(vecs), notmax


def qdef(z, thr=1e-3):
    """max_t |Im z|/|z|  (the branch-free quantization defect of the notes)."""
    a = np.abs(z)
    s = a > thr * a.max()
    return float(np.max(np.abs(z.imag[s]) / a[s]))


def stairs(z):
    """sign changes of Re z counted as +pi each (the exact convention of Sec. 7.7)."""
    s = np.sign(z.real)
    s[s == 0] = 1
    return np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])


def evolve(o, M, z0, T, nt):
    D = o['D']
    E = expm(M * (T / nt))
    r = np.outer(z0, z0.conj()).ravel(order='F').astype(complex)
    out = np.empty((nt + 1, D, D), complex)
    for j in range(nt + 1):
        out[j] = r.reshape(D, D, order='F')
        r = E @ r
    return out


def run(N, n, ncyc=6, nt=2400, psi0=None, **kw):
    """Sector-n evolution. Returns a dict of diagnostics."""
    o = sector_ops(N, n)
    H, M = generator(o, **kw)
    b, Om = balanced(N, n, kw.get('k', K0), kw.get('J', J0))
    z0 = o['V'] @ b if psi0 is None else (o['V'] @ psi0 if len(psi0) != o['D'] else psi0)
    z0 = z0 / norm(z0)
    T = ncyc * np.pi / Om
    R = evolve(o, M, z0, T, nt)
    z, lam, gap, vecs, notmax = pointer_branch(R, z0)
    tr = np.real(np.trace(R, axis1=1, axis2=2))
    P = o['P']
    sym = max(norm(P @ r.conj() @ P - r) / norm(r) for r in R)
    # unitary reference (same H, no dissipation), parallel transported
    U1 = expm(-1j * H * (T / nt))
    u, zu, prev, inf = z0.astype(complex), [], None, []
    for j in range(nt + 1):
        uu = u / norm(u)
        if prev is not None:
            uu = uu * np.exp(-1j * np.angle(np.vdot(prev, uu)))
        prev = uu
        zu.append(np.vdot(z0, uu))
        inf.append(1 - abs(np.vdot(uu, vecs[j])) ** 2)
        u = U1 @ u
    zu = np.array(zu)
    su, sg = stairs(zu), stairs(z)
    return dict(z=z, zu=zu, lam=lam, gap=gap, tr=tr, sym=sym, qdef=qdef(z),
                wmin=float(np.min(lam / tr)), gapmin=float(np.min(gap / tr)),
                inf=float(np.max(inf)), steps=(int(su[-1]), int(sg[-1])),
                mism=float(np.mean(su != sg)), notmax=notmax, Om=Om, o=o, R=R, z0=z0)


def rtn_ensemble(N, n, k, tau, gphi, ncyc=6, nt=1200, common=False):
    """EXACT noise average for random telegraph noise delta_i(t) = +-sigma (one
    fluctuator per emitter, or one common fluctuator), switching rate 1/(2 tau),
    sigma^2 tau = gphi: stochastic Liouville equation on (system x fluctuators).
    Returns (qdef of the pointer branch, max_t ||P rho^* P - rho||/||rho||)."""
    o = sector_ops(N, n)
    b, Om = balanced(N, n, k)
    z0 = o['V'] @ b
    sig = np.sqrt(gphi / tau)
    lam = 1 / (2 * tau)
    D2 = o['D'] ** 2
    confs = [(1,) * N, (-1,) * N] if common else list(itertools.product((1, -1), repeat=N))
    idx = {c: j for j, c in enumerate(confs)}
    nc = len(confs)
    big = np.zeros((nc * D2, nc * D2), complex)
    for c, j in idx.items():
        _, M = generator(o, k=k, deltas=[sig * x for x in c])
        big[j * D2:(j + 1) * D2, j * D2:(j + 1) * D2] += M
        flips = [tuple(-x for x in c)] if common else \
            [tuple(-c[i] if q == i else c[q] for q in range(N)) for i in range(N)]
        for c2 in flips:
            j2 = idx[c2]
            big[j * D2:(j + 1) * D2, j2 * D2:(j2 + 1) * D2] += lam * np.eye(D2)
            big[j * D2:(j + 1) * D2, j * D2:(j + 1) * D2] -= lam * np.eye(D2)
    T = ncyc * np.pi / Om
    E = expm(big * (T / nt))
    r0 = np.outer(z0, z0).ravel(order='F').astype(complex)
    X = np.concatenate([r0 / nc] * nc)
    R = []
    for j in range(nt + 1):
        R.append(X.reshape(nc, D2).sum(0).reshape(o['D'], o['D'], order='F'))
        X = E @ X
    R = np.array(R)
    P = o['P']
    return qdef(pointer_branch(R, z0)[0]), max(norm(P @ r.conj() @ P - r) / norm(r) for r in R)



if __name__ == "__main__":
    # ============================================================== (T1)
    hdr("(T1) ||Theta L Theta^-1 - L|| / ||L||  on the full n-sector, chiral locus")
    for N, n in [(2, 2), (3, 3), (4, 2)]:
        o = sector_ops(N, n)
        PP = np.kron(o['P'], o['P'])
        rows = []
        for lbl, kw in [("gph=0", {}), ("gph=0.3", dict(gph=0.3)),
                        ("gphc=0.3", dict(gphc=0.3)), ("kph=0.3", dict(kph=0.3)),
                        ("gph_i inhom", dict(gphs=list(np.linspace(0.05, 0.3, N))))]:
            for k in (0.0, K0):
                _, M = generator(o, k=k, **kw)
                rows.append(f"{lbl},k={k}: {norm(PP @ M.conj() @ PP - M) / norm(M):.1e}")
        print(f"  N={N} n={n}:  " + "  ".join(rows))
    print("  (k=0: exact symmetry of the whole sector.  k!=0: broken on the full\n"
          "   sector by the S-dependent offsets; see (T2) for permutation-symmetric states)")

    # ============================================================== (T2)
    hdr("(T2) pointer-branch GP quantization vs dephasing strength (on the locus)")
    print("  max_t|Im z|/|z|  [max_t ||P rho^* P - rho||/||rho||],  6 half-cycles")
    gl = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0]
    print("  case            " + "".join(f"gph={x:<14}" for x in gl))
    for N, n in [(1, 1), (1, 3), (2, 2), (3, 1), (3, 3), (4, 2), (4, 4), (5, 3)]:
        print(f"  N={N} n={n}  indiv  " + "".join(
            f"{(r := run(N, n, gph=x))['qdef']:.1e} [{r['sym']:.0e}]  " for x in gl), flush=True)
    for N, n in [(2, 2), (3, 3)]:
        print(f"  N={N} n={n}  collec " + "".join(
            f"{(r := run(N, n, gphc=x))['qdef']:.1e} [{r['sym']:.0e}]  " for x in gl))
        print(f"  N={N} n={n}  cavity " + "".join(
            f"{(r := run(N, n, kph=x))['qdef']:.1e} [{r['sym']:.0e}]  " for x in gl))
    print("  off the locus, gph=0.1:")
    for N, n in [(2, 2), (3, 3), (4, 2)]:
        print(f"    N={N} n={n}: dD=0.05g -> {run(N, n, gph=0.1, dD=0.05)['qdef']:.2e}"
              f"   dD=0.3g -> {run(N, n, gph=0.1, dD=0.3)['qdef']:.2e}")
    rng = np.random.default_rng(1)
    print("  initial states, N=3 n=3, gph=0.3:")
    o = sector_ops(3, 3)
    mx = 0
    for _ in range(8):
        a = rng.normal(size=4)
        mx = max(mx, run(3, 3, gph=0.3, psi0=np.where(np.arange(4) % 2 == 0, a, 1j * a))['qdef'])
    print(f"    8 random Dicke states of Pcal: max {mx:.1e}")
    a = rng.normal(size=4) + 1j * rng.normal(size=4)
    print(f"    generic complex Dicke state:  {run(3, 3, gph=0.3, psi0=a)['qdef']:.2e}")
    a = rng.normal(size=o['D'])
    psf = np.where(o['m'] % 2 == 0, a, 1j * a)
    print(f"    random Pcal state of the FULL sector (not symmetric), k=0:     "
          f"{run(3, 3, gph=0.3, k=0.0, psi0=psf)['qdef']:.1e}")
    print(f"    same state, k=0.25 (offsets between S sectors now matter):   "
          f"{run(3, 3, gph=0.3, k=K0, psi0=psf)['qdef']:.2e}")
    print("  collective dephasing, Dicke chain only, n=N, N up to 10 (gphc=0.3):")
    for N in [4, 6, 8, 10]:
        # Dicke chain: S_z = 2m-N is diagonal; run the chain directly
        d = N + 1
        m = np.arange(d)
        H = chain(N, N, K0, J0)
        Lz = np.sqrt(0.3 / 2) * np.diag(2.0 * m - N)
        G = np.diag(GAM * (N - m) + P0 * m) + Lz @ Lz
        M = liouvillian(H - 0.5j * G, [Lz])
        b, Om = balanced(N, N)
        oo = dict(D=d)
        R = evolve(oo, M, b, 6 * np.pi / Om, 2400)
        z = pointer_branch(R, b)[0]
        Pc = np.diag((-1.0) ** m)
        print(f"    N={N:2d} (d={d:2d}): {qdef(z):.1e}  sym={max(norm(Pc@r.conj()@Pc-r)/norm(r) for r in R):.0e}")

    # ============================================================== (T3)
    hdr("(T3) what dephasing DOES degrade (individual gph, on the locus, 6 half-cycles)")
    print("  w_min = min_t lambda_branch/Tr rho_n;  gap_min = min_t (lambda_branch-lambda_next)/Tr rho_n")
    print("  inf = max_t ray infidelity vs unitary;  steps unit/diss;  mism = fraction of time dphi=pi")
    for N, n in [(1, 1), (2, 2), (3, 1), (3, 3), (4, 2)]:
        for x in [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            r = run(N, n, gph=x)
            print(f"  N={N} n={n} gph={x:<5}: w_min={r['wmin']:.4f} gap_min={r['gapmin']:+.4f} "
                  f"inf={r['inf']:.2e} steps={r['steps'][0]}/{r['steps'][1]} mism={r['mism']:.4f} "
                  f"(branch not largest at {r['notmax']} steps)  Omega={r['Om']:.3f}", flush=True)
    print("  longer window (20 half-cycles), N=2 n=2: staircase mismatch")
    for x in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]:
        r = run(2, 2, gph=x, ncyc=20, nt=6000)
        print(f"    gph={x:<5}: steps={r['steps'][0]}/{r['steps'][1]} mism={r['mism']:.4f} "
              f"qdef={r['qdef']:.1e} w_min={r['wmin']:.3f}")

    # ============================================================== (T4)
    hdr("(T4) full Lindblad (qutip, all sectors): block structure and the two pointer conventions")
    from qutip import tensor, qeye, destroy, sigmam, sigmap, sigmaz, basis, Qobj, mesolve

    def full(N, n, k=K0, J=J0, gph=0.0, dD=0.0):
        Nc = n + 1

        def at(op, i):
            ops = [qeye(2)] * N + [qeye(Nc)]
            ops[i] = op
            return tensor(ops)
        a = tensor([qeye(2)] * N + [destroy(Nc)])
        sm = [at(sigmam(), i) for i in range(N)]
        sp = [at(sigmap(), i) for i in range(N)]
        sz = [at(sigmaz(), i) for i in range(N)]
        chi = 2 * (k - J)
        D = chi * (2 * n - N) + dD
        H = sum(D / 2 * s for s in sz) + chi * (a.dag() * a) ** 2
        H += sum(a * sp[i] + a.dag() * sm[i] for i in range(N))
        H += 2 * k * sum(sm[i] * sp[j] + sp[i] * sm[j] for i, j in itertools.combinations(range(N), 2))
        H += J * sum(sz[i] * sz[j] for i, j in itertools.combinations(range(N), 2))
        c = [np.sqrt(GAM) * a] + [np.sqrt(P0) * s for s in sm]
        c += [np.sqrt(gph / 2) * s for s in sz] if gph > 0 else []
        e, gr = basis(2, 0), basis(2, 1)
        vecs = []
        for m in range(min(N, n) + 1):
            acc = 0
            for pos in itertools.combinations(range(N), m):
                acc = acc + tensor([e if i in pos else gr for i in range(N)] + [basis(Nc, n - m)])
            vecs.append((acc / acc.norm()).full().ravel())
        Nexc = a.dag() * a + sum(sp[i] * sm[i] for i in range(N))
        return H, c, np.array(vecs).T, np.round(np.real(np.diag(Nexc.full())))

    opts = {'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 200000}
    for N, n in [(2, 2), (3, 3), (4, 2)]:
        b, Om = balanced(N, n)
        for gph in [0.0, 0.1, 0.3]:
            H, c, V, ne = full(N, n, gph=gph)
            psi0 = Qobj(V @ b, dims=[H.dims[0], [1] * len(H.dims[0])])
            ts = np.linspace(0, 6 * np.pi / Om, 1201)
            R = np.array([s.full() for s in mesolve(H, psi0, ts, c_ops=c, options=opts).states])
            z0 = psi0.full().ravel()
            blk = ne[:, None] != ne[None, :]
            offb = max(np.abs(r[blk]).max() for r in R)
            z, lam, gap, vecs, notmax = pointer_branch(R, z0)
            insec = np.sum(np.abs(vecs[:, np.abs(ne - n) < 1e-9]) ** 2, axis=1)
            # the "largest eigenvalue" convention
            jump = next((ts[j] * Om / np.pi for j, r in enumerate(R)
                         if np.sum(np.abs(eigh(r)[1][np.abs(ne - n) < 1e-9, -1]) ** 2) < 0.5), None)
            print(f"  N={N} n={n} gph={gph}: max off-block |rho|={offb:.0e}; branch: weight in "
                  f"sector n in [{insec.min():.6f},{insec.max():.6f}], qdef={qdef(z):.1e}; "
                  f"argmax pointer leaves sector n at Omega t/pi = "
                  f"{'never (<6)' if jump is None else f'{jump:.2f}'}", flush=True)

    # ============================================================== (T5)
    hdr("(T5) inhomogeneous rates: p_i and gph_i, relative spread eps, k=0 vs k=0.25")
    for N, n in [(2, 2), (3, 3), (3, 1)]:
        for k in (0.0, K0):
            rows = []
            for eps in [0.01, 0.1, 0.5]:
                spread = np.linspace(-1, 1, N)
                r1 = run(N, n, k=k, ps=list(P0 * (1 + eps * spread)))
                r2 = run(N, n, k=k, gphs=list(0.1 * (1 + eps * spread)))
                rows.append(f"eps={eps}: p_i {r1['qdef']:.1e} [{r1['sym']:.0e}]  "
                            f"gph_i {r2['qdef']:.1e} [{r2['sym']:.0e}]")
            print(f"  N={N} n={n} k={k}:  " + "   ".join(rows), flush=True)

    # ============================================================== (T6)
    hdr("(T6) quasi-static Gaussian frequency noise  H += sum_i delta_i sz_i/2, ensemble average")

    def static_ensemble(N, n, nodes, weights, shared, k=K0, ncyc=6, nt=1200):
        o = sector_ops(N, n)
        b, Om = balanced(N, n, k)
        z0 = o['V'] @ b
        T = ncyc * np.pi / Om
        if shared:
            combos = [((x,) * N, w) for x, w in zip(nodes, weights)]
        else:
            combos = [(xs, np.prod(ws)) for xs, ws in
                      zip(itertools.product(nodes, repeat=N), itertools.product(weights, repeat=N))]
        R = 0
        shot = None
        for xs, w in combos:
            _, M = generator(o, k=k, deltas=xs)
            Rs = evolve(o, M, z0, T, nt)
            R = R + w * Rs
            shot = Rs
        P = o['P']
        sym = max(norm(P @ r.conj() @ P - r) / norm(r) for r in R)
        return qdef(pointer_branch(R, z0)[0]), sym, qdef(pointer_branch(shot, z0)[0])

    x9, w9 = np.polynomial.hermite_e.hermegauss(9)
    w9 = w9 / w9.sum()
    x5, w5 = np.polynomial.hermite_e.hermegauss(5)
    w5 = w5 / w5.sum()
    sig = 0.1
    for N, n, shared, k in [(1, 1, True, K0), (2, 2, True, K0), (3, 3, True, K0),
                            (2, 2, False, 0.0), (3, 3, False, 0.0), (3, 1, False, 0.0),
                            (2, 2, False, K0), (3, 3, False, K0), (3, 1, False, K0)]:
        xs, ws = (x9, w9) if (shared or N == 2) else (x5, w5)
        rows = []
        for j, (lbl, nodes, wts) in enumerate(
                [("mean 0", sig * xs, ws), ("mean 0.05g", 0.05 + sig * xs, ws),
                 ("skewed mean 0", np.array([-0.2, 0.1]), np.array([1 / 3, 2 / 3]))]):
            q, s, q1 = static_ensemble(N, n, nodes, wts, shared, k=k)
            rows.append(f"{lbl}: {q:.1e} [{s:.0e}]")
            if j == 0:
                shot = f"   | one shot, all delta_i=+{sig * xs[-1]:.2f}: {q1:.2f}"
        print(f"  N={N} n={n} {'collective' if shared else 'individual'} k={k}: "
              + "   ".join(rows) + shot, flush=True)

    # ============================================================== (T7)
    def ou_ensemble(N, n, k, tau, gphi, npath, ncyc=6, seed=0):
        o = sector_ops(N, n)
        m = o['m']
        chi = 2 * (k - J0)
        H0 = hamiltonian(o, chi * (2 * n - N), chi, k, J0)
        G = np.diag(GAM * (n - m) + P0 * m)
        b, Om = balanced(N, n, k)
        z0 = o['V'] @ b
        sig = np.sqrt(gphi / tau)
        T = ncyc * np.pi / Om
        nt = int(np.ceil(T / min(tau / 20, 0.02)))
        dt = T / nt
        U0 = expm(-1j * (H0 - 0.5j * G) * dt)
        szd = np.array([s.diagonal() for s in o['sz']])
        rng = np.random.default_rng(seed)
        d = rng.normal(size=(npath, N)) * sig
        a, bb = np.exp(-dt / tau), sig * np.sqrt(1 - np.exp(-2 * dt / tau))
        copies = [(s, list(pm)) for s in (1, -1) for pm in itertools.permutations(range(N))]
        psi = np.tile(z0.astype(complex), (npath * len(copies), 1))
        stride = max(1, nt // 1200)
        R = []
        for j in range(nt + 1):
            if j % stride == 0:
                R.append(psi.T @ psi.conj() / len(psi))
            if j == nt:
                break
            dall = np.concatenate([s * d[:, pm] for s, pm in copies])
            ph = np.exp(-0.25j * dt * (dall @ szd))        # half step of sum delta_i sz_i/2
            psi = ph * ((ph * psi) @ U0.T)
            d = a * d + bb * rng.normal(size=d.shape)
        R = np.array(R)
        P = o['P']
        return qdef(pointer_branch(R, z0)[0]), max(norm(P @ r.conj() @ P - r) / norm(r) for r in R)

    hdr("(T7a) EXACT ensemble: random telegraph noise delta_i = +-sigma on each emitter,\n"
        "      switching rate 1/(2 tau_c), stochastic Liouville equation, gph_eff = sigma^2 tau_c")

    for N, n in [(2, 2), (3, 1)]:
        for k in (0.0, K0):
            rows = []
            for tau in [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]:
                q, sd = rtn_ensemble(N, n, k, tau, 0.1)
                rows.append(f"tau={tau:g}: {q:.0e} [{sd:.0e}]")
            print(f"  N={N} n={n} k={k}:  " + "  ".join(rows), flush=True)
    print("  common-mode telegraph noise (delta_i = delta), k=0.25:  " + "  ".join(
        f"tau={tau:g}: {rtn_ensemble(2, 2, K0, tau, 0.1, common=True)[0]:.0e}" for tau in [0.01, 1.0, 100.0]))

    hdr("(T7b) Monte Carlo cross-check: Ornstein-Uhlenbeck noise, N=2 n=2, paths symmetrized\n"
        "      under delta->-delta and emitter permutations (k!=0: limited by sampling at small tau_c)")
    for k in (0.0, K0):
        for tau in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]:
            rows = []
            for npath in ([500, 2000] if k else [500]):
                q, s = ou_ensemble(2, 2, k, tau, 0.1, npath, seed=npath)
                rows.append(f"npath={npath}: qdef={q:.1e} symdef={s:.1e}")
            print(f"  k={k} tau_c={tau:<5}: " + "   ".join(rows), flush=True)

    # ============================================================== (T8)
    hdr("(T8) single quantum trajectories: random sz_i kicks + no-jump evolution, N=3 n=3")
    for k in (0.0, K0):
        o = sector_ops(3, 3)
        H, _ = generator(o, k=k)
        G = np.diag(GAM * (3 - o['m']) + P0 * o['m'])
        b, Om = balanced(3, 3, k)
        psi = o['V'] @ b + 0j
        P = o['P']
        dt = 0.01
        U = expm(-1j * (H - 0.5j * G) * dt)
        rng = np.random.default_rng(3)
        worst = 0
        for j in range(3000):
            psi = U @ psi
            if rng.random() < 0.01:
                psi = o['sz'][rng.integers(3)] @ psi
            v = psi / norm(psi)
            ph = np.vdot(P @ v.conj(), v)              # P v^* = e^{i a} v  => fix phase
            v = v * np.exp(-0.5j * np.angle(ph))
            worst = max(worst, norm(P @ v.conj() - v))
        print(f"  k={k}: max_t ||P psi^* - psi|| along a 30/g trajectory with ~30 kicks = {worst:.1e}")

    # ============================================================== (T9)
    hdr("(T9) control: a dephasing channel NOT of the protected type, N=2 n=2")
    o = sector_ops(2, 2)
    # k=0 isolates the cross-correlation (unequal weights are harmless at k=0, (T5)).
    # With EQUAL weights, sz1 + e^{i th} sz2 is rescued by Theta o (1<->2 swap).
    for lbl, L in [("L = sz1 + 0.5 i sz2 (complex)", o['sz'][0] + 0.5j * o['sz'][1]),
                   ("L = sz1 + 0.5 sz2   (real)   ", o['sz'][0] + 0.5 * o['sz'][1]),
                   ("L = sz1 + i sz2 (complex, equal weights)", o['sz'][0] + 1j * o['sz'][1])]:
        for c in (0.05, 0.3):
            r = run(2, 2, k=0.0, extra_jumps=(np.sqrt(c) * L,))
            print(f"  {lbl}, rate {c}, k=0: qdef={r['qdef']:.1e}  sym={r['sym']:.0e}")

    # ============================================================== (T10)
    hdr("(T10) corollary: thermal jumps at Delta=chi=0, k=J (every sector chiral), N=2, start n=2")
    for kJ, nth, gph in [(0.0, 0.1, 0.0), (0.2, 0.1, 0.0), (0.2, 0.1, 0.1), (0.2, 0.3, 0.1)]:
        H, c, V, ne = full(2, 5, k=kJ, J=kJ, gph=gph)      # cutoff 6 photons
        from qutip import tensor as _t
        Nc = 6
        a = _t([qeye(2)] * 2 + [destroy(Nc)])
        # rebuild with the n=2 initial state (cutoff from n=5)
        e, gr = basis(2, 0), basis(2, 1)
        dicke = [_t([gr, gr, basis(Nc, 2)]),
                 (_t([e, gr, basis(Nc, 1)]) + _t([gr, e, basis(Nc, 1)])).unit(),
                 _t([e, e, basis(Nc, 0)])]
        Hc = chain(2, 2, kJ, kJ)
        hc = Hc - np.trace(Hc) / 3 * np.eye(3)
        w, W = eigh(hc)
        bb = W[:, -1] + np.diag([1, -1, 1]) @ W[:, -1]
        bb /= norm(bb)
        psi0 = sum(bb[i] * dicke[i] for i in range(3))
        sm = [_t([sigmam(), qeye(2), qeye(Nc)]), _t([qeye(2), sigmam(), qeye(Nc)])]
        c_th = c + [np.sqrt(GAM * nth) * a.dag()] + [np.sqrt(P0 * nth) * s.dag() for s in sm]
        Om = w[-1]
        ts = np.linspace(0, 6 * np.pi / Om, 1201)
        R = np.array([s.full() for s in mesolve(H, psi0, ts, c_ops=c_th, options=opts).states])
        Pf = np.diag(np.real(_t([sigmaz(), sigmaz(), qeye(Nc)]).full().diagonal()))
        sym = max(norm(Pf @ r.conj() @ Pf - r) / norm(r) for r in R)
        z = pointer_branch(R, psi0.full().ravel())[0]
        print(f"  k=J={kJ} nth={nth} gph={gph}: qdef={qdef(z):.1e}  max ||P rho^* P - rho||/||rho|| = {sym:.0e}")
    H, c, V, ne = full(2, 5, k=K0, J=J0)
    print("  (control on the n=2 locus of the notes, where the n=1,3 sectors are NOT chiral:)")
    Nc = 6
    a = tensor([qeye(2)] * 2 + [destroy(Nc)])
    e, gr = basis(2, 0), basis(2, 1)
    dicke = [tensor([gr, gr, basis(Nc, 2)]),
             (tensor([e, gr, basis(Nc, 1)]) + tensor([gr, e, basis(Nc, 1)])).unit(),
             tensor([e, e, basis(Nc, 0)])]
    chi = 2 * (K0 - J0)
    bq, Om = balanced(2, 2)
    psi0 = sum(bq[i] * dicke[i] for i in range(3))
    # full() put Delta on the n=5 locus; rebuild H for the n=2 locus
    sz = [tensor([sigmaz(), qeye(2), qeye(Nc)]), tensor([qeye(2), sigmaz(), qeye(Nc)])]
    sp = [tensor([sigmap(), qeye(2), qeye(Nc)]), tensor([qeye(2), sigmap(), qeye(Nc)])]
    smm = [s.dag() for s in sp]
    H = chi * 2 / 2 * (sz[0] + sz[1]) + chi * (a.dag() * a) ** 2 + sum(a * s + a.dag() * s.dag() for s in smm)
    H += 2 * K0 * (smm[0] * sp[1] + sp[0] * smm[1]) + J0 * sz[0] * sz[1]
    c_th = [np.sqrt(GAM) * a, np.sqrt(GAM * 0.1) * a.dag()] + [np.sqrt(P0) * s for s in smm] \
        + [np.sqrt(P0 * 0.1) * s.dag() for s in smm]
    ts = np.linspace(0, 6 * np.pi / Om, 1201)
    R = np.array([s.full() for s in mesolve(H, psi0, ts, c_ops=c_th, options=opts).states])
    Pf = np.diag(np.real(tensor([sigmaz(), sigmaz(), qeye(Nc)]).full().diagonal()))
    print(f"  chi=2(k-J), Delta=2chi, nth=0.1: qdef={qdef(pointer_branch(R, psi0.full().ravel())[0]):.1e}"
          f"  sym={max(norm(Pf @ r.conj() @ Pf - r) / norm(r) for r in R):.0e}")

    print(f"\n[total {time.time() - T0:.0f} s]")
