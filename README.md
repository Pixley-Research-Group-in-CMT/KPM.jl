# KPM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yixingfu.github.io/KPM.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yixingfu.github.io/KPM.jl/dev)
[![Build Status](https://github.com/yixingfu/KPM.jl/workflows/CI/badge.svg)](https://github.com/yixingfu/KPM.jl/actions)
[![Coverage](https://codecov.io/gh/yixingfu/KPM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/yixingfu/KPM.jl)


## Installation
  First, install the old version of CUDA with
  ```
  ] add CUDA@3.12.0
  ```
  Then, add the KPM package
  ```
  ] add https://github.com/angkunwu/KPM.jl
  ```
  If the code automatically upgrade CUDA, go to `~/.julia/packages/CUDA/` (or where you put your package) and delete the higher version of CUDA. Then, `] rm CUDA` to uninstall CUDA from registries and `] add CUDA@3.12.0`, which will also precompile KPM.

  This is an [unregistered package](https://docs.julialang.org/en/v1.0/stdlib/Pkg/#Adding-unregistered-packages-1) for now, so we need to use github URL to add package. Github username and password needed.

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
