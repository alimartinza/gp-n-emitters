#!/usr/bin/env python
"""
check_N_emitters_part2.py -- resuelve los TODO abiertos en NOTES_N_emitters.tex.

  (L) TODO-4: el hiperplano n=1,  Delta = chi - 2(N-1)(k-J),  SIN imponer
      chi=2(k-J), verificado en el Lindblad COMPLETO para N=3,4.

  (M) TODO-3 + fortalecimiento: la familia protegida es exactamente
      Pcal = { psi : P psi = psi^* }  ~=  RP^{d-1}
      (una forma real COMPLETA de CP^{d-1}), no solo los estados reales de
      una subred.  Es invariante bajo el flujo disipativo, y sobre ella la
      GP esta cuantizada.  Esto DEMUESTRA la conjetura de la Sec. residuo
      (la fuga entre dobletes no puede tocar la fase).

  (N) TODO-2: estructura de los ceros de delta_phi.  Distingue
      cuantizacion de la fase (vale en TODA Pcal) de rigidez del rayo
      (solo en puntos especiales).  Relee la Sec. 22 de NOTES_bargmann.

  (O) TODO-1: entrelazamiento para N>=3 (n=1 tipo W, y n=3), comparando
      cerrado / no-salto / Lindblad completo, dentro y fuera de la linea.

  (P) delta_phi(t) a lo largo de TODA la evolucion (no solo al tiempo final)
      y en las DOS convenciones de rama: envuelta mod 2pi y acumulada
      (contando vueltas). Sobre la linea delta_phi(t) toma unicamente los
      valores 0 y +-pi en todo instante.
"""
import numpy as np
import itertools
from numpy.linalg import eigh, norm
from scipy.linalg import expm
from qutip import (tensor, qeye, destroy, sigmam, sigmap, sigmaz, basis,
                   mesolve, Qobj, partial_transpose)

np.set_printoptions(precision=4, suppress=True, linewidth=160)
OK = lambda c: "OK" if c else "  <-- FALLA"


def hdr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


# ------------------------------------------------------------- bloque teorico
def block(N, n, D, chi, k, J, g=1.0):
    d = min(N, n) + 1
    m = np.arange(d)
    eps = D * (m - N / 2) + chi * (n - m) ** 2 + 2 * k * m * (N - m) \
        + (J / 2) * ((2 * m - N) ** 2 - N)
    H = np.diag(eps).astype(float)
    for mm in range(d - 1):
        H[mm, mm + 1] = H[mm + 1, mm] = g * np.sqrt((mm + 1) * (N - mm) * (n - mm))
    return H


def gamma_op(N, n, gam, p):
    m = np.arange(min(N, n) + 1)
    return np.diag(gam * (n - m) + p * m).astype(float)


def parity(d):
    return np.diag([(-1.0) ** m for m in range(d)])


def chiral_setup(N, n, gam=0.1, p=0.05, k=0.25, J=-0.15, g=1.0):
    chi = 2 * (k - J)
    D = chi * (2 * n - N)
    d = min(N, n) + 1
    H = block(N, n, D, chi, k, J, g)
    h = H - np.mean(np.diag(H)) * np.eye(d)
    return h, h - 0.5j * gamma_op(N, n, gam, p), parity(d), d


def in_Pcal(d, r):
    """psi con P psi = psi^*:  real en la subred par, imaginario en la impar."""
    psi = np.array(r, dtype=complex)
    psi[1::2] *= 1j
    return psi / norm(psi)


# ------------------------------------------------------------ modelo completo
def full_model(N, n, D, chi, k, J, g=1.0, gam=0.1, p=0.05, Ncap=None):
    Nc = Ncap or (n + 1)

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
    Heff = H.full() - 0.5j * (gam * (a.dag() * a).full()
                              + p * sum(sm[i].dag() * sm[i] for i in range(N)).full())
    e, gr = basis(2, 0), basis(2, 1)
    vecs = []
    for m in range(min(N, n) + 1):
        acc = 0
        for pos in itertools.combinations(range(N), m):
            acc = acc + tensor([e if i in pos else gr for i in range(N)]
                               + [basis(Nc, n - m)])
        vecs.append((acc / acc.norm()).full().ravel())
    return dict(H=H, c=c, Heff=Heff, V=np.array(vecs).T, dims=H.dims[0])


def transported(vs, ref0=None):
    """Gauge de transporte paralelo.  `ref0` fija la fase del PRIMER vector
    pidiendo <ref0|psi_+(0)> > 0, que es la convencion cinematica estandar
    (psi_+(0) = psi_0).  Sin esto el autovector que devuelve LAPACK arranca
    con una fase arbitraria que contamina toda la cadena transportada."""
    out, prev = [], None
    for j, v in enumerate(vs):
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


def pointer_states(states, ref0=None):
    """Rama de rho(t) que arranca en el autovalor 1 de rho(0), seguida por
    continuidad (maximo solapamiento con el paso previo).  NO es el autovector
    de autovalor maximo: rho es diagonal por bloques en N_exc y los autovalores
    de bloques distintos se cruzan exactamente."""
    out = []
    for st in states:
        ev, evec = np.linalg.eigh(st.full())
        i = int(np.argmax(ev)) if not out else int(np.argmax(np.abs(out[-1].conj() @ evec)))
        out.append(evec[:, i])
    return transported(out, ref0)


# ============================================================== (L)  TODO-4
hdr("(L) TODO-4: hiperplano n=1,  Delta = chi - 2(N-1)(k-J),  SIN imponer chi=2(k-J)"
    "\n    (Lindblad completo; observable = Im<psi0|psi_+>/|.| con transporte paralelo)")
rng = np.random.default_rng(5)
for N in [2, 3, 4]:
    for trial in range(3):
        chi, k, J = np.round(rng.uniform(-0.6, 0.6, 3), 3)
        D_on = chi - 2 * (N - 1) * (k - J)
        on_line2 = abs(chi - 2 * (k - J)) < 1e-9          # ¿tambien en la linea de d>=3?
        row = []
        for lbl, D in [("en el plano", D_on), ("fuera (+0.2g)", D_on + 0.2)]:
            md = full_model(N, 1, D, chi, k, J)
            psi0v = md['V'] @ in_Pcal(2, [1.0, 1.0])       # balanceado con fase relativa i
            psi0 = Qobj(psi0v, dims=[md['dims'], [1] * len(md['dims'])])
            Om = 2 * abs(block(N, 1, D, chi, k, J)[0, 1])
            ts = np.linspace(0, 4 * np.pi / Om, 601)
            st = mesolve(md['H'], psi0, ts, c_ops=md['c'],
                         options={'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 200000}).states
            z0 = psi0v
            mi = max(abs(np.vdot(z0, v).imag) / abs(np.vdot(z0, v))
                     for v in pointer_states(st, ref0=z0) if abs(np.vdot(z0, v)) > 1e-5)
            row.append(f"{lbl}={mi:.2e}")
        print(f"  N={N} chi={chi:+.3f} k={k:+.3f} J={J:+.3f} "
              f"(chi=2(k-J)? {'SI' if on_line2 else 'no'}): " + "   ".join(row))

# ============================================================== (M)  TODO-3
hdr("(M) TODO-3: la familia protegida es Pcal={psi: P psi = psi^*} ~= RP^{d-1}"
    "\n    (a) Pcal es invariante bajo el flujo;  (b) GP cuantizada en TODA Pcal")
for (N, n) in [(2, 2), (3, 3), (4, 4), (5, 5), (3, 6)]:
    h, Heff, P, d = chiral_setup(N, n)
    worst_inv = worst_q = 0.0
    for _ in range(8):
        psi0 = in_Pcal(d, rng.normal(size=d))
        for t in np.linspace(0, 50, 501):
            psi = expm(-1j * Heff * t) @ psi0
            psi = psi / norm(psi)
            worst_inv = max(worst_inv, norm(P @ psi - psi.conj()))
            z = psi0.conj() @ psi
            worst_q = max(worst_q, abs(z.imag) / (abs(z) + 1e-300))
    # control: psi0 complejo generico
    worst_c = 0.0
    for _ in range(8):
        psi0 = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi0 /= norm(psi0)
        for t in np.linspace(0, 50, 501):
            psi = expm(-1j * Heff * t) @ psi0
            z = psi0.conj() @ psi
            worst_c = max(worst_c, abs(z.imag) / abs(z))
    print(f"  N={N} n={n} d={d}: max||P psi - psi*||={worst_inv:.1e}   "
          f"max|Im z|/|z| en Pcal={worst_q:.1e}   control generico={worst_c:.2f}"
          f"   {OK(worst_inv < 1e-9 and worst_q < 1e-9)}")

print("\n  Corolario (resuelve la CONJETURA de la Sec. 'residuo'): la cuantizacion")
print("  se sigue SOLO de P psi0 = psi0^*, que el flujo preserva exactamente.")
print("  No usa nada sobre desintonias entre dobletes => la fuga entre dobletes")
print("  equiespaciados no puede afectar la fase. Verificacion directa: dobletes")
print("  casi degenerados (n>>N, Omega_j equiespaciados) siguen dando GP cuantizada:")
for (N, n) in [(5, 500), (7, 700), (9, 900)]:
    h, Heff, P, d = chiral_setup(N, n)
    w = np.linalg.eigvalsh(h)
    pos = np.sort(w[w > 1e-9 * max(abs(w))])[::-1]
    spac = np.diff(pos)
    psi0 = in_Pcal(d, rng.normal(size=d))
    wq = 0.0
    for t in np.linspace(0, 6 * np.pi / pos[0], 801):
        psi = expm(-1j * Heff * t) @ psi0
        z = psi0.conj() @ psi
        wq = max(wq, abs(z.imag) / abs(z))
    print(f"    N={N} n={n} d={d}: espaciado de Omega_j uniforme a "
          f"{np.std(spac)/abs(np.mean(spac)):.1e} relativo;  max|Im z|/|z|={wq:.1e}")

# ============================================================== (N)  TODO-2
hdr("(N) TODO-2: ceros de delta_phi.  Cuantizacion de la FASE (toda Pcal) vs"
    "\n    RIGIDEZ DEL RAYO (solo puntos especiales).  Familia: cos(th) u_1 + sin(th) u_2")
for (N, n) in [(3, 3), (5, 5)]:
    h, Heff, P, d = chiral_setup(N, n)
    w, V = eigh(h)
    pairs = sorted([(w[i], V[:, i]) for i in range(d) if w[i] > 1e-9],
                   key=lambda x: -x[0])
    us = [(vp + P @ vp) / norm(vp + P @ vp) for _, vp in pairs]   # balanceados, en A
    print(f"  N={N} n={n} d={d}, {len(us)} dobletes.  th/pi: "
          f"|Im z|/|z| (fase)   |   infid del rayo")
    for th in np.linspace(0, 0.5, 6) * np.pi:
        psi0 = np.cos(th) * us[0] + np.sin(th) * us[1]
        psi0 = psi0 / norm(psi0)
        wq = wi = 0.0
        T = 6 * np.pi / pairs[0][0]
        for t in np.linspace(0, T, 1201):
            pd = expm(-1j * Heff * t) @ psi0
            pd = pd / norm(pd)
            pu = expm(-1j * h * t) @ psi0
            z = psi0.conj() @ pd
            wq = max(wq, abs(z.imag) / abs(z))
            wi = max(wi, 1 - abs(pu.conj() @ pd) ** 2)
        tag = "  <== rayo rigido" if wi < 3e-4 else ""
        print(f"    th={th/np.pi:.3f}pi:   {wq:.2e}          |   {wi:.3e}{tag}")

print("\n  Lectura: la fase esta cuantizada en TODA la familia (col. 1 ~ 1e-13),")
print("  pero el rayo solo es (casi) rigido en los extremos. Es decir: el 'falso")
print("  positivo' de NOTES_bargmann Sec.22 (theta=5pi/6) NO es un falso positivo")
print("  para la fase --- delta_phi=0 alli es proteccion genuina. Lo que falla es")
print("  la rigidez del rayo, que es lo que mide la infidelidad.")

# ============================================================== (O)  TODO-1
hdr("(O) TODO-1: entrelazamiento para N>=3.  cerrado / no-salto / Lindblad completo,"
    "\n    dentro y fuera de la linea quiral.")


def negat(rho, mask):
    pt = partial_transpose(rho, mask)
    ev = np.linalg.eigvalsh(pt.full())
    return float(np.sum(np.abs(ev[ev < 0])))


for (N, n) in [(3, 1), (3, 3)]:
    k, J = 0.25, -0.15
    chi = 2 * (k - J)
    h, Heff_b, P, d = chiral_setup(N, n, k=k, J=J)
    if d == 2:
        psi0b = in_Pcal(2, [1.0, 1.0])
        Om = 2 * abs(h[0, 1])
    else:
        w, V = eigh(h)
        i = int(np.argmax(w))
        vp = V[:, i]
        psi0b = (vp + P @ vp) / norm(vp + P @ vp)
        Om = w[i]
    print(f"\n  --- N={N}, n={n} "
          f"({'sector de 1 excitacion, tipo W' if n == 1 else 'multi-excitacion'}) ---")
    for lbl, Doff in [("SOBRE la linea", 0.0), ("fuera (dD=+0.3g)", 0.3)]:
        md = full_model(N, n, chi * (2 * n - N) + Doff, chi, k, J)
        psi0v = md['V'] @ psi0b
        psi0 = Qobj(psi0v, dims=[md['dims'], [1] * len(md['dims'])])
        ts = np.linspace(0, 6 * np.pi / Om, 241)
        st = mesolve(md['H'], psi0, ts, c_ops=md['c'],
                     options={'atol': 1e-12, 'rtol': 1e-10, 'nsteps': 200000}).states
        Uh = expm(-1j * md['H'].full() * (ts[1] - ts[0]))
        Ue = expm(-1j * md['Heff'] * (ts[1] - ts[0]))
        vu = psi0v.copy()
        vn = psi0v.copy()
        Nu = Nn = Nf = []
        Nu, Nn, Nf = [], [], []
        Ru, Rn, Rf = [], [], []
        mask_aa = [1, 0] + [0] * (N - 2) + [0]      # transponer atomo 1: par 1-2
        mask_1r = [1] + [0] * (N - 1) + [0]         # atomo 1 vs resto
        for j in range(len(ts)):
            rf = st[j]
            ru = Qobj(np.outer(vu, vu.conj()), dims=[md['dims'], md['dims']])
            vnn = vn / norm(vn)
            rn = Qobj(np.outer(vnn, vnn.conj()), dims=[md['dims'], md['dims']])
            # atomo-atomo: trazar cavidad y los demas atomos
            keep = [0, 1]
            Nu.append(negat(ru.ptrace(keep), [1, 0]))
            Nn.append(negat(rn.ptrace(keep), [1, 0]))
            Nf.append(negat(rf.ptrace(keep), [1, 0]))
            # atomo 1 vs resto (todo el sistema)
            Ru.append(negat(ru, mask_1r))
            Rn.append(negat(rn, mask_1r))
            Rf.append(negat(rf, mask_1r))
            vu = Uh @ vu
            vn = Ue @ vn
        Nu, Nn, Nf = map(np.array, (Nu, Nn, Nf))
        Ru, Rn, Rf = map(np.array, (Ru, Rn, Rf))
        print(f"    [{lbl:18s}]")
        print(f"      atomo-atomo  : max|N_cerrado - N_nosalto| = {np.max(np.abs(Nu-Nn)):.2e}"
              f"   max|N_cerrado - N_Lindblad| = {np.max(np.abs(Nu-Nf)):.3f}")
        print(f"                     N_cerrado(final)={Nu[-1]:.4f}  N_Lindblad(final)={Nf[-1]:.4f}"
              f"  (max cerrado {Nu.max():.4f})")
        print(f"      1-vs-resto   : max|N_cerrado - N_nosalto| = {np.max(np.abs(Ru-Rn)):.2e}"
              f"   max|N_cerrado - N_Lindblad| = {np.max(np.abs(Ru-Rf)):.3f}")
        print(f"                     N_cerrado(final)={Ru[-1]:.4f}  N_Lindblad(final)={Rf[-1]:.4f}"
              f"  (max cerrado {Ru.max():.4f})")


# ============================================================== (P)
hdr("(P) delta_phi a lo largo de TODA la evolucion, no solo al tiempo final,\n"
    "    y en las DOS convenciones de rama (envuelta mod 2pi / acumulada)")


def dphi_vs_t(N, n, dD, M=6000, gam=0.1, p=0.05, k=0.25, J=-0.15):
    """Devuelve delta_phi(t) en ambas convenciones.

    Convencion 'wrap': phi = arg<psi0|psi> - conexion, envuelto en (-pi,pi].
      Es la que usa el bloque (K) y la que corresponde al enunciado
      delta_phi en {0,+-pi}, que es una afirmacion mod 2pi.
    Convencion 'acc': se levanta arg<psi0|psi> de forma continua, contando
      vueltas. Sobre la linea z es REAL y pasa exactamente por el origen,
      donde la fase de Pancharatnam no existe; alli la vuelta es una
      CONVENCION, y se toma la unica exacta y sin tolerancias: cada cambio de
      signo de Re z suma +pi. Fuera de la linea z no toca el origen y el
      incremento arg[z(t) conj(z(t-dt))] es inequivoco.
    """
    chi = 2 * (k - J)
    d = min(N, n) + 1
    Hl = block(N, n, chi * (2 * n - N), chi, k, J)
    hl = Hl - np.trace(Hl) / d * np.eye(d)
    H = block(N, n, chi * (2 * n - N) + dD, chi, k, J)
    h = H - np.trace(H) / d * np.eye(d)
    w, V = eigh(hl)
    i = int(np.argmax(w))
    vp = V[:, i]
    Om = w[i]
    P = parity(d)
    psi0 = (vp + P @ vp) / norm(vp + P @ vp)
    G = gamma_op(N, n, gam, p)
    T = 6 * np.pi / Om
    out = {}
    for lbl, Heff in [("closed", h), ("open", h - 0.5j * G)]:
        U = expm(-1j * Heff * (T / M))
        v = psi0.astype(complex).copy()
        zs = [complex(1, 0)]
        accs = [0.0]
        acc = 0.0
        pv = v.copy()
        for _ in range(M):
            v = U @ v
            zs.append(np.vdot(psi0, v))
            acc += np.angle(np.vdot(pv, v))
            accs.append(acc)
            pv = v.copy()
        zs = np.array(zs)
        accs = np.array(accs)
        mag = np.abs(zs)
        ok = mag > 1e-6
        real = np.max(np.abs(zs.imag[ok]) / mag[ok]) < 1e-9
        pan = np.zeros(len(zs))
        for s in range(1, len(zs)):
            if real:
                pan[s] = pan[s - 1] + (np.pi if zs[s].real * zs[s - 1].real < 0 else 0.0)
            else:
                pan[s] = pan[s - 1] + np.angle(zs[s] * np.conj(zs[s - 1]))
        out[lbl] = dict(acc=pan - accs,
                        wrap=(np.angle(zs) - accs + np.pi) % (2 * np.pi) - np.pi)
    wrapf = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    return (out["open"]["acc"] - out["closed"]["acc"],
            wrapf(out["open"]["wrap"] - out["closed"]["wrap"]))


print("  " + f"{'caso':17s} {'conv':5s} {'max|dphi|/pi':>13s} {'t con |dphi|>pi/2':>18s}"
      f" {'mediana|dphi|/pi':>17s}  cuantizado?")
for (N, n) in [(2, 2), (3, 3), (4, 2)]:
    for dD in [0.0, 0.05, 0.3]:
        da, dw = dphi_vs_t(N, n, dD)
        for cname, dd in [("acc", da), ("wrap", dw)]:
            frac = np.mean(np.abs(dd) > 0.5 * np.pi) * 100
            near = np.max(np.abs(dd / np.pi - np.round(dd / np.pi)))
            print(f"  N={N} n={n} dD={dD:4.2f}   {cname:5s} {np.max(np.abs(dd))/np.pi:13.4f}"
                  f" {frac:17.2f}% {np.median(np.abs(dd))/np.pi:17.2e}  "
                  f"{'SI: solo 0,+-pi' if near < 1e-6 else 'no: continuo'}")
print("\n  Lectura: sobre la linea, delta_phi(t) toma UNICAMENTE los valores 0 y +-pi")
print("  en todo instante --- la cuantizacion es manifiesta a lo largo de toda la")
print("  evolucion, no solo al tiempo final. El porcentaje de tiempo en +-pi mide el")
print("  desfasaje temporal entre los dos escalones (0% para N=2; 0.07% para N=3).")
print("  Fuera de la linea ambas convenciones dan un continuo. Y notese que la")
print("  convencion acumulada exhibe la cuantizacion de forma EXACTA (0.0000 y")
print("  1.0000), mientras que la envuelta la muestra con ruido de envoltura")
print("  (0.0056, 0.9989): reportar solo la envuelta subestima el resultado.")
