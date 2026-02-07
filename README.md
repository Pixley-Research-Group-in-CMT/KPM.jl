# KPM

[![Build Status](https://github.com/angkunwu/KPMsub.jl/workflows/CI/badge.svg)](https://github.com/angkunwu/KPMsub.jl/actions)
[![CI](https://github.com/angkunwu/KPMsub.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/angkunwu/KPMsub.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://angkunwu.github.io/KPMsub.jl/dev/)
[![Julia](https://img.shields.io/badge/julia-1.10-blue.svg)](https://julialang.org)



## Capability

Kernel Polynomial Method (KPM) for various quantities:

KPM for density of states (DOS) (RevModPhys.78.275):
```
mu = KPM.kpm_1d(H_norm, NC, NR)
E, rho = KPM.dos(mu, D)
```
where `H_norm` is the scaled Hamiltonian with original half bandwidth `D`, `NC` is the expansion order and `NR` is the number of random vectors for stochastic estimation of trace in KPM. The function `KPM.dos` evaluate the DOS `rho` with energy points `E`.

KPM for DC conductivity (linear response)  (RevModPhys.78.275):
```
μ2Dxy = KPM.kpm_2d(H_norm, Jx, Jy, NC, NR, NH)
dσxyE = KPM.d_dc_cond(μ2Dxy, D, Evals)
```
where  `Jx, Jy` are current operators and `NH` is the dimensionality of the Hamiltonian `H_norm`. The function `KPM.d_dc_cond` evaluate the differential first-order conductivity `dσxyE` with given energy points `Evals`, integration of which (with Fermi distribution) gives DC conductivity at certain temperature.

KPM for frequency-dependent nonlinear response (arXiv:1810.03732):
```
mu_3d_xyz = KPM.kpm_3d(H_norm, Jx, Jy, Jz, NC, NR, NH)
dchi_xyz = KPM.d_cpge(mu_3d_xyz, NC, w1, w2, E)
```
where `dchi_xyz` is the differential second-order conductivity (arXiv:2312.14244), `w1, w2` are two frequencies and `E` is the energy to evaluate, integration of which (with Fermi distribution) gives non at certain temperature.


## Installation
<!--  First, install the old version of CUDA with
  ```
  ] add CUDA@3.12.0
  ```
  Then, add the KPM package
  ```
  ] add https://github.com/Pixley-Research-Group-in-CMT/KPM.jl
  ```
  If the code automatically upgrade CUDA, go to `~/.julia/packages/CUDA/` (or where you put your package) and delete the higher version of CUDA. Then, `] rm CUDA` to uninstall CUDA from registries and `] add CUDA@3.12.0`, which will also precompile KPM. Also recent Julia updates change GPUcompiler, which also affect the old version CUDA. To downgrade Julia, we recommend users to install `juliaup` (https://github.com/JuliaLang/juliaup), i.e., 
  ```
  brew install juliaup
  ```
Then, to add old julia to your system
```
juliaup add 1.9.1
```
You can start the specific version by
```
julia +1.9.1
```
We also recommend to create a separate directory for KPM related projects say `KPMenv` and start julia with the separate environment
```
julia +1.9.1 --project=./KPMenv
```
-->
  This is an [unregistered package](https://docs.julialang.org/en/v1.0/stdlib/Pkg/#Adding-unregistered-packages-1) for now, so we need to use github URL to add package. Github username and password needed. We recently updated the CUDA dependence and there is no need to constrain the compatible version now. 

Install the package with the latest CUDA.jl:
  ```
  ] add https://github.com/Pixley-Research-Group-in-CMT/KPM.jl
  ```
  The package now supports CUDA.jl versions 4 and 5, making it compatible with modern Julia packages.

  After installation, you should be able to import the package by
  ```
      julia> using KPM
  ```
  without further action with github. To update, use
  ```
  ] update KPM
  ```
  and type github username / password when prompted. 
  

## Getting started with DOS

  You will need Hamiltonian (`H_orig`).  

  The most simple way to calculate DOS is
  ```
  julia> E, rhoE = KPM.dos(H_orig)
  ```

  For many situation, you may need more control over the calculation. In such case, you may calculate moment first and then convert to DOS. 
When calculating density of state, left and right input vectors for KPM need to be random and identical when both written as a ket. This is
done by default in `kpm_1d`, or you may supply your own input vectors.

  When using `kpm_1d` (and `kpm_1d!`, the in place version), Hamiltonian must be normalized. `H_norm` is the Hamiltonian normalized by `a` such that `H_orig * a == H_norm`, and `H_norm` has all eigenvalues in (-1, 1). Calculating `mu` and then DOS allows calculating DOS and its derivative at zero energy simpler.

  ```
  julia> NC = 1024; NR = 13;
  julia> mu = KPM.KPM_1d(H_norm, NC, NR)
  julia> rho_0 = KPM.dos0(mu, a)
  julia> d2rho_0 = KPM.dos0(mu, a; dE_order=2)
  julia> E_grid, rho_E = dos(mu, a, 50; E_range=[-1.5, 1.5])
  ```

  [DC conductivity](https://arxiv.org/abs/1410.8140) **WIP**:

## Some practical tips:
  * Store your github credentials (for convenience). See [this](https://docs.github.com/en/free-pro-team@latest/github/using-git/caching-your-github-credentials-in-git) instruction.
  * When using Amarel, the interactive computing nodes do not have git available. Add/update package on login node. 
  * When using RUPC, ssh to one of n34, n35, n40. Add package there. Load it once to compile. 
  * If there is any permission issue when running, add permission: `chmod -r +x ~/.julia/`. For permission error when building, simply ignore.


## Notes about GPU:

This package uses GPU *automatically* when GPU is available. No action needed. To check whether GPU is enabled, run
```
  KPM.whichcore()
```
It is also recommended to add this line right after importing `KPM` in julia code to guarantee using GPU correctly.
