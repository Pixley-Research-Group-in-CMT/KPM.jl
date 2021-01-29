using JLD2
using Test
using KPM
using LinearAlgebra

@load "test_data/ops_small.jld2"

mu = KPM.kpm_1d(H_norm, 2048, 21)
rho_0 = KPM.dos0(mu, a)
d2rho_0 = KPM.dos0(mu, a; dE_order=2)
println("rho_0 = $(rho_0)")
println("d2rho_0 = $(d2rho_0)")
@test abs(rho_0) < 1e-4


NC=16
NR=2
NH=H_norm.n

Gamma = zeros(ComplexF64, NC, NC, NC)

psi_in_l = exp.(2pi * 1im * rand(H_norm.n, NR));
psi_in_l = psi_in_l ./ sqrt(dot(psi_in_l, psi_in_l));
psi_in_r = psi_in_l

KPM.kpm_3d!(H_norm, Jx, Jx, Jx, NC, NR, NH, Gamma, psi_in_l, psi_in_r)
ω = 0.1
println("cpge $(KPM.cpge(Gamma, NC, ω))")
