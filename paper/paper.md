---
title: 'KPM.jl: A Julia Package for Kernel Polynomial Method in Condensed Matter Physics'
tags:
  - Julia
  - Kernel Polynomial Method
  - Condensed Matter Physics
  - spectrum
  - transport and optical response
authors:
  - name: Ang-Kun Wu
    orcid: 0000-0002-7018-1674
    ## equal-contrib: true
    affiliation: "1, 3" # (Multiple affiliations must be quoted)
  - name: Yixing Fu
    orcid: 0000-0002-9470-8848
    ## equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Justin H. Wilson
    orcid: 0000-0001-6903-0417
    ## equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "4, 5"
  - name: J. H. Pixley
    orcid: 0000-0002-3109-640X
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physics and Astronomy, Center for Materials Theory, Rutgers University, Piscataway, New Jersey 08854, USA
   index: 1
 - name: Center for Computational Quantum Physics, Flatiron Institute, 162 5th Avenue, New York, New York 10010, USA
   index: 2
 - name: Department of Physics and Astronomy, University of Tennessee, Knoxville, Knoxville, Tennessee, 37996, USA
   index: 3
 - name: Department of Physics and Astronomy, Louisiana State University, Baton Rouge, LA 70803, USA
   index: 4
 - name: Center for Computation and Technology, Louisiana State University, Baton Rouge, LA 70803, USA
   index: 5
date: 1 February 2026
bibliography: paper.bib

---

# Summary

The Kernel Polynomial Method (KPM) is a numerical technique for approximating spectral and response functions of large Hermitian operators without full diagonalization [@RevModPhys.78.275]. KPM expands spectral functions in Chebyshev polynomials and evaluates them via recursive matrix-vector multiplications, using stochastic trace estimation with random vectors to compute traces efficiently. Carefully chosen kernels (e.g., Jackson) suppress Gibbs oscillations and improve convergence of the truncated Chebyshev series. Because the dominant cost is sparse matrix–vector products, KPM scales (nearly) linearly with system size and can be applied to extremely large sparse Hamiltonians to compute quantities such as the density of states (DOS), local density of states (LDOS), and frequency-dependent response functions [@joao2019].

`KPM.jl` is a Julia implementation of KPM for tight-binding models in condensed matter physics, providing a high-performance, user-friendly toolbox. `KPM.jl` targets large sparse Hamiltonians, integrates with `CUDA.jl` for automatic GPU acceleration when available, and is designed for scalability and ease of integration into Julia-based workflows for large-scale spectral and transport calculations.

# Statement of need

`KPM.jl` provides a high-performance implementation of the Kernel Polynomial Method tailored to tight‑binding models. By avoiding full diagonalization and using Chebyshev expansions with stochastic trace estimation, the package enables spectral and transport calculations on very large sparse Hamiltonians that are infeasible with exact diagonalization (ED).
It fills the gap between low-level C/Fortran libraries and interactive, reproducible Julia workflows by offering:

1.  **Scalability:** Capable of handling large sparse matrices representing realistic tight-binding models.
2.  **Versatility:** Support for DOS and LDOS, as well as linear (DC conductivity) and nonlinear optical response functions.
3.  **Performance:** Automatic GPU acceleration via `CUDA.jl` when available.
4.  **Integration:** As a Julia package, it easily interfaces with other tools in the Julia ecosystem for Hamiltonian generation and data analysis.

In condensed matter physics, understanding the effects of disorder, interactions, and complex lattice geometries often requires numerical simulations of very large Hamiltonians. Traditional ED methods for computing the full spectrum are limited to relatively small system sizes (typically $N \sim 10^4$ states), which can be insufficient to resolve the spectral features of disordered systems or incommensurate structures like twisted bilayer graphene.
`KPM.jl` is especially useful for studies of disorder, Moiré systems, and topological materials where large system sizes ($N \sim 10^7$ states) are essential to capture realistic spectral and transport behavior.



# Software design

`KPM.jl` is designed with modularity and performance in mind. The package operates on sparse Hamiltonians and computes Chebyshev moments using stochastic trace estimation with random vectors. The core functionality is divided into three main tiers corresponding to the complexity of the response function:

* **Density of States (1D):** The `kpm_1d` function computes the Chebyshev moments for the DOS. It supports stochastic estimation using multiple random vectors (`NR`) and allows users to supply custom input vectors to compute the LDOS. The moments are converted to the spectral density $\rho(E)$ using `KPM.dos`.

* **Linear Response (2D):** For transport properties like DC conductivity, `kpm_2d` calculates the moments required for the Kubo-Greenwood formula involving two current operators ($J_x, J_y$). The function `d_dc_cond` processes these moments to obtain the energy-dependent conductivity $\sigma_{xy}(E)$.

* **Nonlinear Response (3D):** The package includes specialized routines (`kpm_3d`) for frequency-dependent nonlinear responses, such as the Circular Photogalvanic Effect (CPGE). This involves computing moments for three operators ($J_x, J_y, J_z$) and post-processing them (`d_cpge`) to extract the second-order conductivity $\chi_{xyz}(\omega_1, \omega_2)$.

The software automatically detects available hardware and offloads matrix-vector multiplications to a GPU if a compatible CUDA device is present (`KPM.whichcore()`), ensuring efficient performance on modern clusters.

# Research impact statement

`KPM.jl` has been utilized in numerous studies to investigate electronic structure and response in complex materials:

1.  **Moiré systems and disorder:** Enabled large-scale simulations of twisted bilayer graphene and demonstrated how twist-angle disorder broadens flat bands and modifies spectral features [@Wilson2020; @fu2020magic; @Chang2024; @Yi2022].

2.  **Experimentally relevant transport calculations:** Applied to transport modeling based on tight-binding models derived from DFT+U to directly compare theory with experiment [@Liu2021].

3.  **Quasiperodic potentials for topological material:** Used to compute conductivity and disorder-averaged spectral properties and Green's function in disordered materials [@Yi2022; @fu2021flat].

4.  **Nonlinear optical responses and disordered topological material:** Employed to calculate nonlinear responses such as the CPGE in disordered topological semimetals and to identify spectral signatures of surface and bulk phase transitions in disordered axion insulators [@Wu2024; @Grindall2025].

5.  **Method benchmarking and numerical validation:** Served as a reference implementation for comparing and validating new Chebyshev-regularization and spectral-density algorithms [@Yi2025].

These applications demonstrate the package's capability to tackle cutting-edge problems in condensed matter theory.


# Acknowledgements

We acknowledge the support of NSF Career Grant No. DMR-1941569 and the Alfred P. Sloan
Foundation through a Sloan Research Fellowship. 

# References