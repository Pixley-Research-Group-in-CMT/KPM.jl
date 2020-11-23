# KPM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yixingfu.github.io/KPM.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yixingfu.github.io/KPM.jl/dev)
[![Build Status](https://github.com/yixingfu/KPM.jl/workflows/CI/badge.svg)](https://github.com/yixingfu/KPM.jl/actions)
[![Coverage](https://codecov.io/gh/yixingfu/KPM.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/yixingfu/KPM.jl)


## Installation
  In the interactive Julia REPL, type key `]` to enter package mode as you see for example 
  ```
      (v1.x) pkg>
  ```
  For [unregistered packages](https://docs.julialang.org/en/v1.0/stdlib/Pkg/#Adding-unregistered-packages-1) like this one, we can use github URL to add package.  
  ```
      (v1.x) pkg> add https://github.com/yixingfu/KPM.jl
  ```
  After installation, you should be able to import the package
  ```
      julia> using KPM
  ```
  
  

## Basic usage
  steps: Create Hamiltonian and/or current operator (Trye SuperLattice.jl or create your own matrices or sparse matrices), calculate moments, calculate physical quantities. 

### Minimal demo
  The code in demo folder gives minimal example of DOS calculation. 

###  Compute moments
  For DOS:
  ```
  julia> using KPMjulia.KPM
  julia> mu = DOS_KPM(H, NC, NR, NH) ## NC = 10, NR = 15, NH = 21
  ```
  
  For conductivity:
  ```
  julia> mu2D = KPM_2D_FAST(H, Jx, Jy, NC, NR, NH)
  ```

  When calculating conductivity, time complexity can be reduced at the cost
of extra memory usage. This is done by setting `arr_size` to larger than 3.
Roughly the extra memory cost is `arr_size / 3` times default, and reduce
computation time to `3 / arr_size` of default.

### Compute DOS / conductivity

  For calculating DOS / conductivity, it is usually good to first take average
of moments and then convert to DOS/cond with the averaged moments(when
averaging is needed). However, this may not be always possible if the
normalization of Hamiltonian take different values.

  DOS:
  ```
  julia> E_grid, dos = computeDOSslow(mu, a, 50; Erange=[0.0,0.0])
  ```

  [DC conductivity](https://arxiv.org/abs/1410.8140):
  ```
  julia> Evals, dσxxE = computeOCIntegrand(μ2Dxx, a, 100; Erange=[0.0,0.0],NC=0)
  ```

  Optical conductivity: first compute current density j(E1, E2)
  ```
  j = jab(μ2Dxx; N=Ntilde+1)
  ```
  then integrate
  ```
  σ = j2OC(j)
  ```


## Some practical tips:
  * Store your github credentials (for convenience).  Better not store username/password by setting them in the config. (If insist to do so, in your  shell run  `git config --global credential.helper store` to store credentials after next time you are prompted to enter it. Using `cache` option as opposed to `store` to only keep credentials for 900 secs. )
  * When using Amarel, the interactive computing nodes do not have git available. Do the add package step on login node. 
  * When using RUPC, ssh to one of n34, n35, n40. Add package there. Load it once to compile. 
  * If there is any permission issue when running, add permission: `chmod -r +x ~/.julia/`. For permission error when building, simply ignore.


## Notes about GPU:

This package uses GPU *automatically* when GPU is available. To check whether GPU is enabled, run
```
  KPM.whichcore()
```

* in the case of amarel:
Go to interactive GPU node
```
srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --constraint=pascal --mem=2000 --time=00:30:00 --export=ALL --pty bash -i
```
The `--constraint=pascal` option ensures using GPU cards at a high CUDA level. Without the constraint it may or may not work, especially when running in parallel. 

Then recompile. To recompile, in Julia update the package
```
julia> import Pkg; Pkg.update("KPM"); Pkg.build("KPM")
```
Then restart Julia. 
