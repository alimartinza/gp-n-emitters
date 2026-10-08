# Chiral symmetry for geometric phase protection in the nonlinear Tavis-Cummings model — code

Numerical code accompanying

> A. Martin Zynda, P. I. Villar, F. C. Lombardo,
> *Chiral Symmetry for Geometric Phase Protection against Dissipation in the
> Tavis-Cummings Model with Nonlinearities*.

Every figure, the table, and the numbers quoted in the text can be regenerated
with the scripts below. Units are g = 1. Unless stated otherwise the parameters
are γ = 0.1, p = 0.05, k = 0.25, J = −0.15, with χ = 2(k − J) and Δ = χ(2n − N)
on the chiral locus.

## Requirements

Python ≥ 3.10, NumPy, SciPy, Matplotlib, and QuTiP 5. Install them with

```bash
pip install -r requirements.txt
```

The results in the paper were produced with Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
Matplotlib 3.11.1, and QuTiP 5.3.1 (OpenBLAS 0.3.34).

## Contents

| Script | Reproduces | Runtime |
|---|---|---|
| `make_figures.py` | Figs. 1–5 (written to `figs/`) | ~25 s |
| `table_lindblad.py` | Table II (full Lindblad simulations) | ~20 s |
| `staircase.py` | δφ(t) along the whole evolution: on the locus it takes only the values 0 and ±π, with ±π during 0.07% of the time for N = 3 (Sec. *Numerical verification*) | ~1 s |
| `checks/` | Extended verification of the analytical results (see below) | ~15 min |

```bash
python make_figures.py                     # all paper figures
python make_figures.py fig3_quantization   # a single figure
python table_lindblad.py
python staircase.py
```

`make_figures.py` also prints the numbers quoted in the text for N = 2:
the maximum ray infidelity (9.6 × 10⁻⁵), the zero-mode weight |c_Z|² ~ 10⁻⁴,
and the doublet imbalance ||c₊|² − |c₋|²| ≲ 10⁻¹⁵ (Sec. *Example: two-emitter
Tavis-Cummings*). Two supplementary figures are not part of the default run
and are not in the paper:

```bash
python make_figures.py extra_initial_states   # also prints the drift slope of δφ
                                              # (1.3e-5 π per Rabi period, θ = 0.4π)
python make_figures.py extra_site_energies
```

### Map from figures to functions

| Paper | Function in `make_figures.py` | Method |
|---|---|---|
| Fig. 1 | `fig1_chain` | schematic |
| Fig. 2 | `fig_residual` | symmetric-sector block, exact propagation |
| Fig. 3 | `fig3_quantization` | full Lindblad equation (`qutip.mesolve`) |
| Fig. 4 | `fig4_ray_deviation` | N = 2 no-jump evolution |
| Fig. 5 | `fig5_bargmann` | N = 2 no-jump evolution |

### `checks/`

These are the original verification scripts used while preparing the paper.
Their comments and output are in Spanish. Each one is self-contained and is
run from inside `checks/`.

- `check_N_emitters.py`: the analytical block against the full QuTiP model,
  the chiral-locus conditions, the scalings of the residual, the spectrum, the
  spin sectors, disorder, and (block K) the full Lindblad data of Table II.
- `check_N_emitters_part2.py`: the protected family 𝒫 = {ψ : Pψ = ψ*} and its
  invariance under the dissipative flow, the agreement of the pointer state of
  ρ(t) with the no-jump ket, the entanglement on and off the locus, and (block P)
  the staircase of δφ(t).
- `check_dephasing.py`: pure dephasing, classical frequency noise and
  non-protected dissipators, which back the "preliminary numerical checks"
  mentioned in the paper (~12 min).

## Numerical notes

- The Lindblad simulations use absolute and relative tolerances of 10⁻¹² and
  10⁻¹⁰. Truncating the cavity at n photons is exact, because the dynamics never
  leaves the blocks with at most n excitations.
- The pointer state is the eigenvector of ρ(t) followed by continuity
  (maximum overlap with the previous step), not the one with the largest
  eigenvalue, because eigenvalues of different excitation blocks cross exactly.
- Phases are fixed in the parallel-transport gauge, anchored at ψ₊(0) = ψ₀.
  In that gauge arg z = φ_g, so max_t |Im z|/|z| is a branch-free quantization
  diagnostic.
- On the locus this diagnostic is at the level of round-off (10⁻¹⁶ to 10⁻¹⁰).
  Its exact digits depend on the time grid, the library versions and the BLAS
  build, so only its order of magnitude is reproducible. All other entries of
  Table II are stable.
