using JLD2
using Test
using KPM

@load "test_data/ops_small.jld2"
#@load "test_data/exact_dos.jld2"
H_orig = H_norm * a
E, rho_E = KPM.dos(H_orig)

mu = KPM.kpm_1d(H_norm, 1024, 13)
rho_0 = KPM.dos0(mu, a)
d2rho_0 = KPM.dos0(mu, a; dE_order=2)
println("rho_0 = $(rho_0)")
println("d2rho_0 = $(d2rho_0)")
