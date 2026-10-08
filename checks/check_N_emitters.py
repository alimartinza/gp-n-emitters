#!/usr/bin/env python
"""
check_N_emitters.py -- verificacion numerica de la generalizacion a N emisores
del mecanismo de simetria quiral hallado para N=2 (main_tc_calude.tex,
NOTES_bargmann.tex Secs. 10, 19, 22, 23).

Acompana a NOTES_N_emitters.tex.  Correr:  python check_N_emitters.py

Predicciones testeadas
  (A) bloque analitico de n excitaciones en el sector S=N/2 == construccion
      qutip completa
  (B) linea quiral:  chi = 2(k-J)   Y   Delta = chi (2n - N)
  (C) caso d=2 (min(N,n)=1): condicion UNICA  Delta = chi - 2(N-1)(k-J)
  (D) [P,Gamma]=0 siempre;  P U(t) P = U(t)^*  sobre la linea
  (E) psi0 real en una sola subred  =>  |c_j^+|=|c_j^-| EXACTO para todo
      doblete j, y <psi0|psi(t)> real  =>  GP cuantizada en {0,pi}
  (F) d=2  =>  el rayo disipativo recorre EXACTAMENTE el camino unitario
      (solo reparametrizado);  d>=3  =>  se sale del camino
  (G) residuo ~ (gamma-p)^2, ~1/n, y ACOTADO en N
  (H) espectro Omega_j -> 2 g sqrt(n) j;  #modos cero = d mod 2
  (I) quiralidad exacta dentro de CADA sector de espin total S
  (J) desorden: g_i (off-diagonal) tolerado si k=0;  Delta_i (diagonal) no;
      p_i inhomogeneo siempre tolerado
  (K) Lindblad COMPLETO con gauge de transporte paralelo: GP cuantizada
      sobre la linea, no cuantizada fuera
"""
import numpy as np
import itertools
from numpy.linalg import eigh, norm
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True, linewidth=160)
OK = lambda c: "OK" if c else "  <-- FALLA"


# ---------------------------------------------------------------- bloque teorico
def block(N, n, D, chi, k, J, g=1.0):
    """H^(n) en el sector simetrico S=N/2: cadena abierta de d=min(N,n)+1 sitios.
    Sitio m = numero de atomos excitados."""
    d = min(N, n) + 1
    m = np.arange(d)
    eps = D * (m - N / 2) + chi * (n - m) ** 2 + 2 * k * m * (N - m) \
        + (J / 2) * ((2 * m - N) ** 2 - N)
    H = np.diag(eps).astype(float)
    for mm in range(d - 1):
        t = g * np.sqrt((mm + 1) * (N - mm) * (n - mm))
        H[mm, mm + 1] = H[mm + 1, mm] = t
    return H


def gamma_op(N, n, gam, p):
    m = np.arange(min(N, n) + 1)
    return np.diag(gam * (n - m) + p * m).astype(float)


def parity(d):
    return np.diag([(-1.0) ** m for m in range(d)])


def chiral_setup(N, n, gam=0.1, p=0.05, k=0.25, J=-0.15, g=1.0):
    """Devuelve (h traceless, Heff, P, d) sobre la linea quiral."""
    chi = 2 * (k - J)
    D = chi * (2 * n - N)
    d = min(N, n) + 1
    H = block(N, n, D, chi, k, J, g)
    h = H - np.mean(np.diag(H)) * np.eye(d)
    return h, h - 0.5j * gamma_op(N, n, gam, p), parity(d), d


def balanced(h, P):
    """Estado balanceado del doblete quiral de mayor Omega (vive en UNA subred)."""
    w, V = eigh(h)
    i = int(np.argmax(w))
    vp = V[:, i]
    vm = P @ vp
    return (vp + vm) / norm(vp + vm), w[i], vp, vm


# ------------------------------------------------------------- modelo completo
def full_model(N, n, D, chi, k, J, g=1.0, gam=0.1, p=0.05, gs=None, Ds=None, ps=None):
    from qutip import tensor, qeye, destroy, sigmam, sigmap, sigmaz, basis
    Nc = n + 1

    def at(op, i):
        ops = [qeye(2)] * N + [qeye(Nc)]
        ops[i] = op
        return tensor(ops)

    a = tensor([qeye(2)] * N + [destroy(Nc)])
    sm = [at(sigmam(), i) for i in range(N)]
    sp = [at(sigmap(), i) for i in range(N)]
    sz = [at(sigmaz(), i) for i in range(N)]
    gs = gs if gs is not None else [g] * N
    Ds = Ds if Ds is not None else [D] * N
    ps = ps if ps is not None else [p] * N

    H = sum(Ds[i] / 2 * sz[i] for i in range(N)) + chi * (a.dag() * a) ** 2
    H += sum(gs[i] * (a * sp[i] + a.dag() * sm[i]) for i in range(N))
    H += 2 * k * sum(sm[i] * sp[j] + sp[i] * sm[j]
                     for i, j in itertools.combinations(range(N), 2))
    H += J * sum(sz[i] * sz[j] for i, j in itertools.combinations(range(N), 2))

    c = [np.sqrt(gam) * a] + [np.sqrt(ps[i]) * sm[i] for i in range(N)]
    Gam = gam * (a.dag() * a) + sum(ps[i] * (sm[i].dag() * sm[i]) for i in range(N))
    Nexc = a.dag() * a + sum(sp[i] * sm[i] for i in range(N))
    Nat = sum(sp[i] * sm[i] for i in range(N))
    S2 = 0.25 * ((sum(sp) + sum(sm)) ** 2
                 + (-1j * (sum(sp) - sum(sm))) ** 2 + (sum(sz)) ** 2)

    e, gr = basis(2, 0), basis(2, 1)
    vecs = []
    for m in range(min(N, n) + 1):
        acc = 0
        for pos in itertools.combinations(range(N), m):
            acc = acc + tensor([e if i in pos else gr for i in range(N)]
                               + [basis(Nc, n - m)])
        vecs.append((acc / acc.norm()).full().ravel())
    return dict(H=H, c=c, Gam=Gam, Nexc=Nexc, Nat=Nat, S2=S2,
                V=np.array(vecs).T, a=a, sm=sm)


def sector_idx(md, n):
    ne = np.real(np.diag(md['Nexc'].full()))
    idx = np.where(np.abs(ne - n) < 1e-9)[0]
    na = np.real(np.diag(md['Nat'].full()))[idx]
    return idx, np.diag((-1.0) ** np.round(na))


def chi_break(Hb, P):
    """||parte P-par de h|| / ||h||   (h = Hb sin traza)"""
    d = Hb.shape[0]
    h = Hb - np.trace(Hb) / d * np.eye(d)
    return norm(P @ h @ P + h) / max(norm(h), 1e-300)


def transported(states, kind, ref0=None):
    """Gauge de transporte paralelo (formulacion cinematica estandar).
    ref0 fija la fase del primer vector pidiendo <ref0|psi(0)> > 0; sin eso
    la fase arbitraria de LAPACK contamina toda la cadena.
    kind='rho': el pointer state es la RAMA que arranca en el autovalor 1 de
    rho(0), seguida por continuidad (maximo solapamiento con el paso previo),
    NO el autovector de autovalor maximo: rho es diagonal por bloques en N_exc
    y los autovalores de bloques distintos se cruzan exactamente."""
    out = []
    prev = None
    for j, st in enumerate(states):
        if kind == 'rho':
            ev, evec = np.linalg.eigh(st.full())
            i = int(np.argmax(ev)) if prev is None else int(np.argmax(np.abs(prev.conj() @ evec)))
            v = evec[:, i]
        else:
            v = st.full().ravel()
            v = v / norm(v)
        if j == 0 and ref0 is not None:
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


def hdr(t):
    print("\n" + "=" * 94)
    print(t)
    print("=" * 94)


# ================================================================== (A)
hdr("(A) bloque analitico  vs  construccion qutip completa")
rng = np.random.default_rng(0)
for (N, n) in [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 2), (5, 3), (1, 2), (4, 1)]:
    D, chi, k, J = rng.uniform(-1, 1, 4)
    md = full_model(N, n, D, chi, k, J)
    Bq = md['V'].conj().T @ md['H'].full() @ md['V']
    err = norm(block(N, n, D, chi, k, J) - Bq.real)
    print(f"  N={N} n={n} d={min(N,n)+1}: ||dif|| = {err:.2e}   {OK(err < 1e-10)}")

# ================================================================== (B)
hdr("(B) linea quiral:  chi = 2(k-J),  Delta = chi(2n-N)   [diagonal plana <=> {h,P}=0]")
for (N, n) in [(2, 2), (2, 3), (3, 3), (3, 5), (4, 4), (5, 5), (6, 6), (1, 1), (1, 3)]:
    k, J = rng.uniform(-1, 1, 2)
    chi = 2 * (k - J)
    D = chi * (2 * n - N)
    d = min(N, n) + 1
    H = block(N, n, D, chi, k, J)
    spread = np.diag(H).max() - np.diag(H).min()
    h = H - np.mean(np.diag(H)) * np.eye(d)
    P = parity(d)
    a_ = norm(P @ h @ P + h)
    print(f"  N={N} n={n}: spread(eps_m)={spread:.1e}  ||{{h,P}}||={a_:.1e}"
          f"   {OK(spread < 1e-12 and a_ < 1e-12)}")
print("  control negativo (rompiendo UNA sola condicion):")
for (N, n) in [(3, 3), (4, 4), (5, 5)]:
    k, J = 0.3, -0.1
    chi = 2 * (k - J)
    s1 = np.ptp(np.diag(block(N, n, chi * (2 * n - N), 1.1 * chi, k, J)))
    s2 = np.ptp(np.diag(block(N, n, chi * (2 * n - N) + 0.1, chi, k, J)))
    print(f"    N={N} n={n}: spread(chi mal)={s1:.2e}   spread(Delta mal)={s2:.2e}")

# ================================================================== (C)
hdr("(C) d=2 (min(N,n)=1): condicion UNICA  Delta = chi - 2(N-1)(k-J);  Omega = 2 g sqrt(N)")
for N in [1, 2, 3, 4, 5, 8]:
    k, J, chi = rng.uniform(-1, 1, 3)
    H = block(N, 1, chi - 2 * (N - 1) * (k - J), chi, k, J)
    s = np.ptp(np.diag(H))
    print(f"  N={N} n=1: spread={s:.1e}  Omega=2t_0={2*H[0,1]:.6f}"
          f"  2g sqrt(N)={2*np.sqrt(N):.6f}   {OK(s < 1e-12)}")

# ================================================================== (D)
hdr("(D) [P,Gamma]=0 identicamente;   P U(t) P = U(t)^*  sobre la linea")
for (N, n) in [(2, 2), (3, 3), (4, 4), (5, 5), (6, 4), (3, 7), (7, 3)]:
    h, Heff, P, d = chiral_setup(N, n)
    G = gamma_op(N, n, 0.1, 0.05)
    comm = norm(P @ G - G @ P)
    e_ = max(norm(P @ expm(-1j * Heff * t) @ P - expm(-1j * Heff * t).conj())
             for t in [0.7, 3.3, 11.0])
    print(f"  N={N} n={n} d={d}: ||[P,Gamma]||={comm:.1e}   max||PUP-U*||={e_:.1e}"
          f"   {OK(comm < 1e-14 and e_ < 1e-11)}")

# ================================================================== (E)
hdr("(E) psi0 real en UNA subred => |c_j^+|=|c_j^-| exacto para TODO j, y GP en {0,pi}")
for (N, n) in [(2, 2), (2, 4), (3, 3), (3, 5), (4, 4), (5, 5), (7, 7), (3, 9)]:
    h, Heff, P, d = chiral_setup(N, n)
    w, V = eigh(h)
    pairs = [(w[i], V[:, i], P @ V[:, i]) for i in range(d) if w[i] > 1e-9]
    zeros = [V[:, i] for i in range(d) if abs(w[i]) <= 1e-9]
    psi0, Om, _, _ = balanced(h, P)
    mb = mz = 0.0
    for t in np.linspace(0, 60, 2001):
        psi = expm(-1j * Heff * t) @ psi0
        for (_, vpk, vmk) in pairs:
            cp = vpk.conj() @ psi
            cm = vmk.conj() @ psi
            mb = max(mb, abs(abs(cp) - abs(cm)) / (abs(cp) + abs(cm) + 1e-300))
        z = psi0.conj() @ psi
        mz = max(mz, abs(z.imag) / abs(z))
    print(f"  N={N} n={n} d={d} [{len(pairs)} dobl., {len(zeros)} cero]: "
          f"maxDesbal={mb:.1e}  max|Im z|/|z|={mz:.1e}   {OK(mb < 1e-9 and mz < 1e-9)}")

print("  (E2) CUALQUIER estado real de una sola subred (6 al azar) vs control real generico:")
for (N, n) in [(3, 3), (4, 4), (5, 5)]:
    h, Heff, P, d = chiral_setup(N, n)
    res = []
    for lbl, sub in [("A", 0), ("B", 1), ("generico", None)]:
        worst = 0
        for _ in range(6):
            psi0 = np.zeros(d)
            if sub is None:
                psi0 = rng.normal(size=d)
            else:
                psi0[sub::2] = rng.normal(size=len(psi0[sub::2]))
            psi0 /= norm(psi0)
            for t in np.linspace(0, 50, 601):
                z = psi0.conj() @ (expm(-1j * Heff * t) @ psi0)
                worst = max(worst, abs(z.imag) / (abs(z) + 1e-300))
        res.append(f"{lbl}={worst:.1e}")
    print(f"    N={N} n={n}: " + "   ".join(res))


# ================================================================== (F)
def traj(M, psi0, ts):
    U = expm(-1j * M * (ts[1] - ts[0]))
    v = psi0.copy()
    out = [v.copy()]
    for _ in ts[1:]:
        v = U @ v
        out.append(v.copy())
    A = np.array(out)
    return A / np.linalg.norm(A, axis=1)[:, None]


def deviation(N, n, gam=0.1, p=0.05, ncyc=6, nt=1500, nref=30000):
    h, Heff, P, d = chiral_setup(N, n, gam, p)
    psi0, Om, _, _ = balanced(h, P)
    T = ncyc * np.pi / Om
    ts = np.linspace(0, T, nt)
    tsf = np.linspace(0, 1.4 * T, nref)
    PD = traj(Heff, psi0, ts)
    PUs = traj(h, psi0, ts)
    PU = traj(h, psi0, tsf)
    infid = (1 - np.abs(np.sum(PUs.conj() * PD, axis=1)) ** 2).max()
    dist = (1 - (np.abs(PD.conj() @ PU.T) ** 2).max(axis=1)).max()
    return infid, dist, d, Om


hdr("(F) d=2 => MISMO camino que el unitario (solo reparametrizado);  d>=3 => sale del camino")
for (N, n) in [(1, 1), (1, 6), (3, 1), (10, 1), (2, 2), (3, 3), (4, 4), (5, 5), (2, 6)]:
    a_, b_, d, Om = deviation(N, n)
    tag = "   <== camino IDENTICO" if b_ < 1e-7 else ""
    print(f"  N={N:2d} n={n:2d} d={d:2d}: infid(t)={a_:.3e}  dist_al_camino={b_:.3e}{tag}")

# ================================================================== (G)
hdr("(G) residuo: ~ (gamma-p)^2,  ~1/n,  y ACOTADO en N (no se degrada al agregar emisores)")
print("  --- vs (gamma-p) [N=3,n=3] ---")
for gam, p in [(0.1, 0.05), (0.1, 0.0), (0.2, 0.1), (0.05, 0.025), (0.1, 0.1)]:
    print(f"    gamma={gam} p={p} (dif={gam-p:.3f}): infid = {deviation(3,3,gam,p)[0]:.3e}")
print("  --- vs n [N=2] ---")
for n in [2, 3, 4, 6, 10]:
    print(f"    N=2 n={n:2d}: infid = {deviation(2,n)[0]:.3e}")
print("  --- vs N ---")
for lbl, rule in [("n=N", lambda N: N), ("n=1", lambda N: 1)]:
    print(f"    [{lbl}]  " + "  ".join(f"N={N}:{deviation(N,rule(N))[0]:.2e}"
                                       for N in [2, 3, 4, 6, 8, 12, 16]))

# ================================================================== (H)
hdr("(H) espectro  Omega_j -> 2 g sqrt(n) j  (j=N/2,N/2-1,...);  #modos cero = d mod 2")
for N in [2, 3, 4, 5]:
    for n in [N, 10 * N, 500 * N]:
        h = chiral_setup(N, n)[0]
        w = np.linalg.eigvalsh(h)
        sc = max(abs(w))
        pos = np.sort(w[w > 1e-9 * sc])[::-1]
        pred = np.array([2 * np.sqrt(n) * (N / 2 - j) for j in range(int(np.ceil(N / 2)))])
        print(f"  N={N} n={n:5d}: Omega_j={np.round(pos,3)}  pred={np.round(pred,3)}  "
              f"err_rel={np.max(np.abs(pos-pred)/pred):.1e}")
    w = np.linalg.eigvalsh(chiral_setup(N, N)[0])
    print(f"    modos cero: teorico d mod 2 = {(N+1)%2}, "
          f"medido = {int(np.sum(np.abs(w) < 1e-9*max(abs(w))))}")

# ================================================================== (I)
hdr("(I) quiralidad EXACTA dentro de cada sector de espin total S (mismas condiciones)")
for N, n in [(3, 3), (3, 2), (4, 4), (5, 3)]:
    k, J = 0.25, -0.15
    chi = 2 * (k - J)
    D = chi * (2 * n - N)
    md = full_model(N, n, D, chi, k, J)
    idx, P = sector_idx(md, n)
    Hb = md['H'].full()[np.ix_(idx, idx)].real
    S2b = md['S2'].full()[np.ix_(idx, idx)].real
    ws, Vs = np.linalg.eigh(S2b)
    groups = {}
    for i, wv in enumerate(ws):
        groups.setdefault(round(wv, 6), []).append(Vs[:, i])
    out = []
    for key, vs in sorted(groups.items()):
        Q = np.array(vs).T
        S = (-1 + np.sqrt(1 + 4 * key)) / 2
        Hs = Q.T @ Hb @ Q
        note = " (dark)" if norm(Hs - np.diag(np.diag(Hs))) < 1e-12 else ""
        out.append(f"S={S:.1f}(dim{Q.shape[1]}):{chi_break(Hs, Q.T @ P @ Q):.1e}{note}")
    print(f"  N={N} n={n}: " + "  ".join(out))

# ================================================================== (J)
hdr("(J) DESORDEN.  chi_break = ||parte P-par de h||/||h|| en el sector n COMPLETO")
rng2 = np.random.default_rng(11)
for N, n in [(3, 3), (4, 3), (5, 2)]:
    for klab, kv in [("k=0.25", 0.25), ("k=0   ", 0.0)]:
        J = -0.15
        chi = 2 * (kv - J)
        D = chi * (2 * n - N)
        rows = []
        for lbl, gs, Ds in [("limpio", None, None),
                            ("g_i +-40%", list(1 + 0.4 * rng2.normal(size=N)), None),
                            ("Delta_i +-0.3", None, list(D + 0.3 * rng2.normal(size=N)))]:
            md = full_model(N, n, D, chi, kv, J, gs=gs, Ds=Ds)
            idx, P = sector_idx(md, n)
            rows.append(f"{lbl}={chi_break(md['H'].full()[np.ix_(idx,idx)].real, P):.1e}")
        print(f"  N={N} n={n} {klab}: " + "   ".join(rows))
print("  p_i inhomogeneo: Gamma sigue diagonal =>")
for N, n in [(3, 3), (4, 3)]:
    ps = list(0.05 + 0.04 * rng2.normal(size=N))
    md = full_model(N, n, 0, 0, 0, 0, ps=ps)
    idx, P = sector_idx(md, n)
    Gb = md['Gam'].full()[np.ix_(idx, idx)].real
    print(f"    N={N} n={n}: offdiag(Gamma)={norm(Gb-np.diag(np.diag(Gb))):.1e}"
          f"  ||[P,Gamma]||={norm(P@Gb-Gb@P):.1e}")

# ================================================================== (K)
hdr("(K) LINDBLAD COMPLETO + transporte paralelo: Im<psi0|psi_+>/|.| ->0 solo SOBRE la linea")
from qutip import mesolve, sesolve, Qobj

for N, n in [(2, 2), (3, 3), (3, 1), (4, 2)]:
    k, J = 0.25, -0.15
    chi0 = 2 * (k - J)
    h, _, P, d = chiral_setup(N, n, k=k, J=J)
    psi0b, Om, _, _ = balanced(h, P)
    rows = []
    for lbl, Doff, cf in [("linea", 0, 1.0), ("dD=+0.05g", 0.05, 1.0),
                          ("dD=+0.3g", 0.3, 1.0), ("chi x1.3", 0, 1.3)]:
        md = full_model(N, n, chi0 * (2 * n - N) + Doff, chi0 * cf, k, J)
        H, c = md['H'], md['c']
        psi0 = Qobj(md['V'] @ psi0b, dims=H.dims[0:1] + [[1] * len(H.dims[0])])
        ts = np.linspace(0, 4 * np.pi / Om, 801)
        st = mesolve(H, psi0, ts, c_ops=c,
                     options={'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 200000}).states
        z0 = psi0.full().ravel()
        mi = max(abs(np.vdot(z0, v).imag) / abs(np.vdot(z0, v))
                 for v in transported(st, 'rho', ref0=z0) if abs(np.vdot(z0, v)) > 1e-5)
        rows.append(f"{lbl}={mi:.2e}")
    print(f"  N={N} n={n} (dim {2**N*(n+1)}): " + "   ".join(rows))

print("  y el observable del paper, delta_phi = phi_g - phi_u:")
for N, n in [(2, 2), (3, 3), (4, 2)]:
    k, J = 0.25, -0.15
    chi0 = 2 * (k - J)
    h, _, P, d = chiral_setup(N, n, k=k, J=J)
    psi0b, Om, _, _ = balanced(h, P)
    rows = []
    for lbl, Doff in [("linea", 0.0), ("dD=+0.05g", 0.05), ("dD=+0.3g", 0.3)]:
        md = full_model(N, n, chi0 * (2 * n - N) + Doff, chi0, k, J)
        H, c = md['H'], md['c']
        psi0 = Qobj(md['V'] @ psi0b, dims=H.dims[0:1] + [[1] * len(H.dims[0])])
        ts = np.linspace(0, 3.6 * np.pi / Om, 801)
        o = {'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 200000}
        z0 = psi0.full().ravel()
        pg = transported(mesolve(H, psi0, ts, c_ops=c, options=o).states, 'rho', ref0=z0)
        pu = transported(sesolve(H, psi0, ts, options=o).states, 'ket', ref0=z0)
        dphi = np.angle(np.vdot(z0, pg[-1])) - np.angle(np.vdot(z0, pu[-1]))
        rows.append(f"{lbl}: dphi/pi={((dphi+np.pi)%(2*np.pi)-np.pi)/np.pi:+.3e}")
    print(f"    N={N} n={n}: " + "   ".join(rows))

print("\n(nota: sobre la linea dphi da ~5e-9 en los tres casos. La cuantizacion Z_2")
print(" admite ademas el valor exacto +-pi, si las dos ramas acumulan distinto numero")
print(" de saltos; no se observo aqui. La firma de RUPTURA es un dphi CONTINUO, 1e-3..1e-2.")
print(" IMPORTANTE: sin ref0 en transported(), la fase arbitraria de LAPACK en el primer")
print(" autovector produce un dphi espurio de +-pi que se confunde con un salto real.)")
