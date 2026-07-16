# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KPM.jl implements the Kernel Polynomial Method for tight-binding Hamiltonians:
density of states, DC conductivity (Kubo–Bastin), optical conductivity, and
second-order (CPGE) response, with optional CUDA GPU acceleration via a
package extension. It is an unregistered Julia package (≥ 1.10). This repo is
the `jhwilson` fork of `Pixley-Research-Group-in-CMT/KPM.jl` (remote
`upstream`); **pull requests target the fork (`origin`), not upstream**.

## Commands

```bash
# Full test suite (what CI runs, on Julia 1.10 and latest)
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# Single test file: most need only KPM + Test
julia --project=. -e 'using Test, KPM; include("test/dos_test.jl")'
# …but integration_test.jl (JLD2) and optional_cuda_test.jl (CUDA) need the
# test extras — run those through Pkg.test() or a TestEnv-activated session.

# Docs (Documenter.jl; deployed from main by CI). The committed
# docs/Manifest.toml is pinned to Julia 1.10 — the build fails on newer Julia
# unless you refresh it, so match CI:
julia +1.10 --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'

# GPU validation — NOT in CI; run manually on a CUDA machine
julia --project=<env with KPM + CUDA + Arpack> test/gpu_smoke.jl
```

Test files are plain scripts wired into `test/runtests.jl` testsets; when
adding one, register it there.

## Architecture

`src/KPM.jl` includes everything in dependency order; each layer only calls
downward:

1. **`src/device.jl`** — CPU/GPU indirection. A single
   `ACTIVE_DEVICE::Ref{AbstractDevice}` is set once by the CUDA extension's
   `__init__` (when `CUDA.functional()`); the main workspaces and transfers go
   through `maybe_to_device` / `maybe_to_host` / `maybe_on_device_zeros` /
   `maybe_on_device_rand`, which dispatch on it. Not everything is
   device-generic: several paths allocate host arrays on purpose, and
   `maybe_to_device` never moves `SubArray`s.
2. **`src/utils/`** — the Chebyshev three-term recurrence
   (`chebyshev_iteration.jl`: `ψᵢ = 2Hψᵢ₋₁ − ψᵢ₋₂` over ring-buffer
   workspaces; on CPU threaded over probe columns with fused 5-arg `mul!`),
   rescaling (`normalizeH`), and probe vectors.
3. **`src/kernels/`** — damping kernels (Jackson default, Lorentz) and the
   `hₙ`/kernel application to raw moments.
4. **`src/moment.jl`** — the compute core: `kpm_1d[!]`, `kpm_2d[!]`,
   `kpm_3d[!]` build moments via stochastic trace over `NR` random-phase
   probe columns. Index convention for the 2D moments (see the
   `conductivity.jl` docstrings): `mu2D[n, m] ≈ Tr[Jα T_m(H_norm) Jβ
   T_n(H_norm)] / D` — note the transposed-looking order. `kpm_1d` uses
   moment doubling (`NC` moments from `NC/2` matvec steps), so **`NC` must be
   even** there; the current-operator paths (`kpm_1d_current`, `kpm_2d`,
   `kpm_3d`) cannot use the doubling trick.
5. **`src/applications/`** — physics reconstruction from moments: `dos.jl`,
   `conductivity.jl` (+ `dc_cond_util.jl`, `dc_cond_long.jl`), `optical_cond.jl`,
   `cpge.jl`, `ldos_mu.jl`.
6. **`src/frontend.jl`** — typed front end (preferred user API):
   `rescale(H)` → `RescaledHamiltonian(H_norm, a, b)`;
   `dos_moments`/`cond_moments` → `DosMoments`/`ConductivityMoments` carrying
   `(mu, a, b, NH, NR)` so reconstruction (`dos(m)`, `kubo_bastin_cond(m2, Ef)`)
   never threads bookkeeping by hand. Reproducibility contract:
   `rng=Xoshiro(seed)` → `random_phase_vectors`. Typed wrappers for
   `optical_cond*`/`cpge`/`dc_long`/`ldos_mu` are known follow-up work
   (`optical_cond1/2` and `cpge` take energies/frequencies in *rescaled*
   units; `dc_long` takes physical `Ef` and converts internally; `ldos_mu`
   takes no energy). Design decision: value types + dispatch, **not** lazy
   stateful result objects.

**GPU extension pattern** (`ext/KPMCUDAExt.jl`): the extension adds methods on
CUDA-specific types only — `KPM.to_device`/`KPM.device_zeros` on `CUDADevice`,
`KPM.chebyshev_iter_single` on `CuArray` (fused 5-arg `mul!`) — never
overwriting base-package methods, so it precompiles cleanly. Write new compute
paths against the `maybe_*` device helpers, but GPU support is not automatic —
audit every allocation, view, `mul!`, and reduction, and expect to add
CUDA-specialized methods in the extension. Known GPU limits:
`Hermitian`-wrapped CuSparse `mul!` multiplies only the stored triangle (why
`dc_long` materializes on host); the Γ contraction is not
ForwardDiff-differentiable on GPU, so `d_dc_cond(...; dE_order ≥ 1)` fails
with the CUDA device active unless the caller forces CPU execution (no
automatic fallback); and `kpm_2d` accumulates its moment matrix on the host —
`kubo_bastin_cond` reconstruction is CPU-side by design. `KPM.whichcore()`
reports whether the GPU path is active.

## Physics conventions (load-bearing — do not change casually)

- **Rescaling**: everything is expanded in `T_n(H_norm)` with
  `H_norm = (H − b·I)/a`, spectrum inside (−1, 1). Plain `normalizeH(H)`
  scales by the spectral radius (`b = 0`) — valid for asymmetric spectra but
  wasteful of Chebyshev resolution; `center=true` finds both spectral edges
  and returns `(a, b, H_norm)`, and is the right choice whenever the spectrum
  is not symmetric about zero.
- **DOS normalization**: with the default unit-norm random-phase probes,
  moments estimate `Tr[Tₙ]/D` (so `μ₀ = 1`) and the reconstructed DOS
  integrates to one. `kpm_1d` defaults to `force_norm=false`, so unnormalized
  user-supplied `psi_in` scales the moments by its squared norm.
- **Current operators** are bond currents built from the **unrescaled** H:
  `(J_α)_ij = H_ij (r_i − r_j)_α` (J = iħv, with ħ = e = 1).
- **Conductivity units**: for a 2D sample (`area=A`), `kubo_bastin_cond`
  returns σ in e²/h (other spatial dimensions change the physical dimensions —
  see its docstring),
  σ = −(2e²/h)·(D/(A a²))·∫dx f(ax+b)·Re[Γ μ̃]/(1−x²)², with sign convention
  σ_xy = +C·e²/h (C the Chern number). This normalization is pinned by
  `test/kubo_bastin_test.jl` against exact diagonalization on the Haldane
  model (`test/ed_reference.jl`), with an independent Fukui–Hatsugai–Suzuki
  Chern-number sign anchor — any change to conductivity normalization must
  keep that test passing. By contrast `dc_cond0`/`dc_cond_single` are *bare*
  Chebyshev Fermi-surface sums, deliberately not in physical units (documented
  as such).
- **Probe vectors are NOT mean-centered** (`normalize_by_col(…; centering=false)`
  is the default). Centering biases the stochastic trace by a rank-one
  projection onto the uniform state — do not "fix" this. When users supply
  `psi_in` to `kpm_1d`, left and right vectors must be identical kets.

## Repo workflow

- Before opening a PR, run an adversarial Codex review (`-m gpt-5.6-sol`) on
  the branch diff and address or explicitly reject each finding.
- `scripts/infra/` is local-only cluster plumbing — never commit anything
  under it or reference its cluster details in committed files. It is ignored
  only via clone-local `.git/info/exclude` (not the tracked `.gitignore`), so
  `git status` showing it as untracked in a fresh clone is not permission to
  add it.
- `paper/` holds the JOSS paper sources; `examples/` are standalone model
  scripts, not part of the test suite.
