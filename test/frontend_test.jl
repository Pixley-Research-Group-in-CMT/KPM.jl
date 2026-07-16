using Test
using Random
using SparseArrays
using LinearAlgebra
using KPM

NH = 64
NR = 4
H = spdiagm(-1 => ones(NH - 1), 1 => ones(NH - 1))
Jx = spdiagm(-1 => -im .* ones(NH - 1), 1 => im .* ones(NH - 1))
Jy = 0.7 .* Jx
h = KPM.rescale(H)

@testset "rescale provenance" begin
    @test Matrix(h.a .* h.H) + h.b * I ≈ Matrix(H)
    @test all((-1 .< eigvals(Hermitian(Matrix(h.H)))) .& (eigvals(Hermitian(Matrix(h.H))) .< 1))

    H_shifted = H + 0.7 * I
    h_centered = KPM.rescale(H_shifted; center=true)
    @test Matrix(h_centered.a .* h_centered.H) + h_centered.b * I ≈ Matrix(H_shifted)
    lambda_centered = eigvals(Hermitian(Matrix(h_centered.H)))
    @test all((-1 .< lambda_centered) .& (lambda_centered .< 1))

    h_fixed = KPM.rescale(H; fixed_a=3)
    @test h_fixed.a === 3.0
    @test h_fixed.b == 0.0
end

@testset "typed DOS moments and reconstruction" begin
    rng = Xoshiro(11)
    psi = exp.(rand(rng, Float64, NH, NR) .* (2im * pi))
    KPM.normalize_by_col(psi, NR)
    m = KPM.dos_moments(h; NC=64, psi_in=copy(psi))
    mu_ref = KPM.kpm_1d(h.H, 64, NR; psi_in=copy(psi))
    @test m.mu == mu_ref
    @test KPM.dos(m) == KPM.dos(mu_ref, h.a; b=h.b)
    @test KPM.dos0(m) == KPM.dos0(mu_ref, h.a)
    @test KPM.dos0(m; dE_order=2) == KPM.dos0(mu_ref, h.a; dE_order=2)

    grid = collect(range(-1.5, 1.5, length=101))
    E, rho = KPM.dos(m; E_grid=grid)
    @test all(isfinite, rho)
    E_range, rho_range = KPM.dos(m; E_range=[-1.5, 1.5], N_tilde=100)
    @test E ≈ E_range
    @test rho ≈ rho_range
    E_explicit, rho_explicit = KPM.dos(m; E_grid=E_range)
    @test E_explicit == E_range
    @test rho_explicit == rho_range
    _, rho_integer_grid = KPM.dos(m; E_grid=collect(-2:2))
    @test eltype(rho_integer_grid) == Float64
    @test all(isfinite, rho_integer_grid)
end

@testset "typed conductivity moments and reconstruction" begin
    rng = Xoshiro(11)
    psi = exp.(rand(rng, Float64, NH, NR) .* (2im * pi))
    KPM.normalize_by_col(psi, NR)
    m2 = KPM.cond_moments(h, Jx, Jy; NC=32, psi_in=copy(psi))
    mu2_ref = KPM.kpm_2d(h.H, Jx, Jy, 32, NR, size(h.H, 1); psi_in=copy(psi))
    @test m2.mu == mu2_ref
    @test KPM.kubo_bastin_cond(m2, 0.1; area=1.0) == KPM.kubo_bastin_cond(mu2_ref, h.a, 0.1; b=h.b, NH=size(h.H, 1), area=1.0)
    @test KPM.dc_cond_single(m2, 0.1) == KPM.dc_cond_single(mu2_ref, h.a, 0.1; b=h.b)
    @test KPM.dc_cond0(m2) == KPM.dc_cond0(mu2_ref, h.a)
    @test KPM.d_dc_cond(m2, [0.0, 0.1]) == KPM.d_dc_cond(mu2_ref, h.a, [0.0, 0.1]; b=h.b)
    @test KPM.d_dc_cond(m2, 0.0:0.05:0.2) == KPM.d_dc_cond(m2, collect(0.0:0.05:0.2))
    @test KPM.d_dc_cond(m2, 0.1) isa Real
    @test KPM.d_dc_cond(m2, 0.1) == only(KPM.d_dc_cond(mu2_ref, h.a, [0.1]; b=h.b))

    @test_throws ArgumentError KPM.kubo_bastin_cond(m2, 0.1; area=1.0, NH=10)
    @test_throws ArgumentError KPM.ConductivityMoments(zeros(ComplexF64, 3, 2), 1.0, 0.0, NH, 1)
end

@testset "centered rescaling parity (b != 0)" begin
    # every energy-dependent delegation must forward the stored b: these
    # comparisons are run away from E = b so a dropped shift changes the numbers
    H_shifted = H + 0.7 * I
    hc = KPM.rescale(H_shifted; center=true)
    @test hc.b != 0.0
    psi = KPM.random_phase_vectors(Xoshiro(21), NH, NR)

    mc = KPM.dos_moments(hc; NC=64, psi_in=copy(psi))
    mu_ref = KPM.kpm_1d(hc.H, 64, NR; psi_in=copy(psi))
    @test mc.mu == mu_ref
    @test KPM.dos(mc) == KPM.dos(mu_ref, hc.a; b=hc.b)
    grid = collect(range(hc.b - 0.5, hc.b + 0.5, length=21))
    @test KPM.dos(mc; E_grid=grid)[2] == KPM.dos(mu_ref, hc.a; b=hc.b, E_grid=grid)[2]

    m2c = KPM.cond_moments(hc, Jx, Jy; NC=32, psi_in=copy(psi))
    mu2_ref = KPM.kpm_2d(hc.H, Jx, Jy, 32, NR, NH; psi_in=copy(psi))
    @test KPM.kubo_bastin_cond(m2c, 0.3; area=1.0) == KPM.kubo_bastin_cond(mu2_ref, hc.a, 0.3; b=hc.b, NH=NH, area=1.0)
    @test KPM.dc_cond_single(m2c, 0.3) == KPM.dc_cond_single(mu2_ref, hc.a, 0.3; b=hc.b)
    @test KPM.d_dc_cond(m2c, [0.3, 0.9]) == KPM.d_dc_cond(mu2_ref, hc.a, [0.3, 0.9]; b=hc.b)
end

@testset "typed front-end validation and reproducibility" begin
    m1 = KPM.dos_moments(h; NC=64, NR=4, rng=Xoshiro(42))
    m2 = KPM.dos_moments(h; NC=64, NR=4, rng=Xoshiro(42))
    m3 = KPM.dos_moments(h; NC=64, NR=4, rng=Xoshiro(43))
    @test m1.mu == m2.mu
    @test m1.mu != m3.mu
    # unit-norm probes give mu_0 = 1 exactly as in the default random path
    @test m1.mu[1] ≈ 1.0

    # rng path reproduces the package's probe recipe exactly (random phase,
    # column-normalized, no mean centering)
    rng_manual = Xoshiro(5)
    psi_manual = exp.(rand(rng_manual, Float64, NH, NR) .* (2im * pi))
    KPM.normalize_by_col(psi_manual, NR)
    @test KPM.random_phase_vectors(Xoshiro(5), NH, NR) == psi_manual

    psi = KPM.random_phase_vectors(Xoshiro(9), NH, NR)
    moments = KPM.dos_moments(h; NC=64, psi_in=copy(psi))
    conductivity_moments = KPM.cond_moments(h, Jx, Jy; NC=32, psi_in=copy(psi))
    @test_throws ArgumentError KPM.dos_moments(h; NC=64, rng=Xoshiro(1), psi_in=psi)
    @test_throws ArgumentError KPM.dos_moments(h; NC=63)
    @test_throws ArgumentError KPM.dos(moments; b=0.2)
    @test_throws ArgumentError KPM.dos_moments(h; NC=64, psi_in=zeros(NH - 1, NR))
    @test_throws ArgumentError KPM.cond_moments(h, Jx, Jy; NC=32, psi_in=zeros(NH - 1, NR))
    @test_throws ArgumentError KPM.kubo_bastin_cond(conductivity_moments, 0.1; area=1.0, b=0.2)

    E, rho = KPM.dos(h; NC=128, NR=4, rng=Xoshiro(7))
    @test all(isfinite, rho)
    integral = sum((rho[1:end-1] .+ rho[2:end]) .* diff(E)) / 2
    @test isapprox(integral, 1.0; rtol=0.15)
    @test occursin("DosMoments", string(moments))
    @test occursin("RescaledHamiltonian", string(h))
end
