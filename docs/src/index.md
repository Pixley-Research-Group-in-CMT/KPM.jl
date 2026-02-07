```@meta
CurrentModule = KPM
```

# KPM

```@index
```
# Moment calculation

The first step in KPM is calculating moments using Hamiltonians (and current operators for conductivity etc.)
Functions with `!` are more efficient in-place version; those without `!` are convenient methods based on the
in-place methods.

```@docs; canonical=false
kpm_1d
kpm_1d!
kpm_2d
kpm_2d!
```

# applications

## DOS

To calculate density of state (DOS), first calculating moment first using `kpm_1d` or `kpm_1d!` with
default (random) input vectors. Then use the output (moment `mu`) to calculate density of state. 
There is also an option to pass Hamiltonian directly to `dos`, which does the
moment calculation automatically.

```@docs; canonical=false
dos
```


# Kernels

Kernels are functions defined as
```
kernel(n::Int64, N::Int64) -> Float64
```
such that when `n==0`, returns `1`; when `n==N-1`, returns small number close to `0`.

We implement JacksonKernel and LorentzKernels in the package.
Jackson Kernel is the default kernel for most application.
```@docs; canonical=false
JacksonKernel
```

Lorentz Kernel is good for Green functions as it respects symmetry. The function `LorentzKernels`
takes parameter λ and returns a kernel function.
```@docs; canonical=false
LorentzKernels
```


# full API reference
```@autodocs
Modules = [KPM, KPM.Utils]
```
