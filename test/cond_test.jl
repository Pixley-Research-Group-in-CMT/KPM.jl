using Test
using LinearAlgebra
using Random
using KPM

# Consistency checks for the conductivity helpers. `dc_cond0` and
# `dc_cond_single(..., Ef=0)` evaluate the same Eq.-25 sum through independent
# code paths (mod-4 index slicing vs closed-form Tn(0) weights), so they must
# agree for arbitrary, non-symmetric moment matrices — the old dc_cond0
# subtracted `onethree` twice, which only cancels for symmetric μ.

@testset "dc_cond0 == dc_cond_single at Ef = 0" begin
    rng = Xoshiro(5)
    NC = 32
    a = 2.3
    mu = randn(rng, ComplexF64, NC, NC)   # generic: μ_nm ≠ μ_mn
    v0 = KPM.dc_cond0(mu, a; NC=NC)
    v1 = KPM.dc_cond_single(mu, a, 0.0; NC=NC)
    @test v0 ≈ v1 atol = 1e-12
end

@testset "Γnmμnmαβ factorized contraction vs direct sum" begin
    rng = Xoshiro(6)
    NC = 24
    mu = randn(rng, ComplexF64, NC, NC)
    for ε in (-0.83, 0.0, 0.37)
        direct = sum(KPM.Γnm.(0:NC-1, (0:NC-1)', ε) .* mu)
        @test KPM.Γnmμnmαβ(mu, ε, NC) ≈ direct rtol = 1e-11
    end
end

@testset "d_dc_cond: zero (not garbage) outside the band" begin
    rng = Xoshiro(8)
    NC = 16
    a = 1.7
    mu = randn(rng, ComplexF64, NC, NC)
    dσ = KPM.d_dc_cond(mu, a, [0.3, 2.5 * a])
    @test isfinite(dσ[1])
    @test dσ[2] == 0.0
end

@testset "kernel application: no-mutate variants match" begin
    NC = 15
    mu2 = ones(NC, NC)
    t_mut = KPM.mu2D_apply_kernel_and_h(copy(mu2), NC, KPM.JacksonKernel)
    t_nomut = KPM.mu2D_apply_kernel_and_h_no_mutate(mu2, NC, KPM.JacksonKernel)
    @test t_mut ≈ t_nomut
    @test all(mu2 .== 1)  # input untouched
end
