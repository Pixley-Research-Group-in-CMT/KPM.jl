using Test
using LinearAlgebra
using SparseArrays
using KPM

include("ed_reference.jl")
using .EDReference

# Units and normalization of the KPM Kubo–Bastin conductivity, validated
# against exact diagonalization on the Haldane model (Chern insulator):
#  - Hall channel, Fermi level in the gap: exact η-free ED reference (TKNN),
#    KPM must give the quantized σ_xy = C e²/h.
#  - Longitudinal channel, Fermi level in the band: ED with Lorentzian
#    broadening of both δ(ε-H) and G(ε), matched to the KPM Lorentz kernel.
#
# The trace is evaluated deterministically (identity probe matrix), so the
# only KPM errors are kernel broadening and quadrature — no stochastic noise.

t2 = 0.2      # gap 6√3 t2 ≈ 2.1 — large, so modest NC resolves it
H, Jx, Jy, area = haldane_model(6, 6; t=1.0, t2=t2, ϕ=π/2, m=0.0)
D = size(H, 1)

a, b, H_norm = KPM.normalizeH(H; center=true)
NC = 96
psi = Matrix{ComplexF64}(I, D, D)          # full deterministic trace
mu2D_xy = KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in=psi)
mu2D_xx = KPM.kpm_2d(H_norm, Jx, Jx, NC, D, D; psi_in=psi)

@testset "quantum Hall: σ_xy quantized and matches ED (units pinned)" begin
    # independent sign anchor: Fukui–Hatsugai–Suzuki Berry flux of the same
    # model's Bloch Hamiltonian — wavefunction overlaps only, no velocity
    # operators, so a simultaneous sign error in the ED and KPM Kubo formulas
    # cannot fool it. TKNN: σ_xy = +C e²/h.
    C_fhs = chern_number_fhs(haldane_bloch(; t=1.0, t2=t2, ϕ=π/2, m=0.0); Nk=30)
    C = round(C_fhs)
    @test abs(C) == 1
    @test C_fhs ≈ C atol = 1e-10

    # exact ED reference: quantized to the FHS Chern number, including sign
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef=0.0)
    @test abs(σxy_ed - C) < 0.02

    # the two independent ED routines agree in the gap
    @test ed_kubo_bastin(H, Jx, Jy, area; Ef=0.0, eta=0.01) ≈ σxy_ed atol = 1e-2
    @test ed_kubo_bastin_broadened(H, Jx, Jy, area; Ef=0.0, eta=0.05) ≈ σxy_ed atol = 1e-2

    # KPM Kubo–Bastin in e²/h units: quantized, correct sign, matches ED
    σxy_kpm = KPM.kubo_bastin_cond(mu2D_xy, a, 0.0; b=b, NH=D, area=area)
    @test σxy_kpm ≈ σxy_ed atol = 0.03
    @test σxy_kpm ≈ C atol = 0.05

    # finite temperature deep in the gap changes nothing (T ≪ gap)
    σxy_T = KPM.kubo_bastin_cond(mu2D_xy, a, 0.0; b=b, NH=D, area=area, beta=50.0)
    @test σxy_T ≈ σxy_kpm atol = 0.01
end

@testset "shifted spectrum: center shift b flows through the Kubo–Bastin chain" begin
    # same Chern insulator, spectrum rigidly shifted: b ≈ shift, and the
    # Fermi function argument f(a x + b) plus x_F = (Ef - b)/a must all be
    # exercised for the plateau to survive at Ef = shift
    shift = 2.5
    Hs = H + shift * sparse(I, D, D)
    as, bs, Hs_norm = KPM.normalizeH(Hs; center=true)
    @test abs(bs - shift) < 0.05

    mu_s = KPM.kpm_2d(Hs_norm, Jx, Jy, NC, D, D; psi_in=psi)
    σxy_s = KPM.kubo_bastin_cond(mu_s, as, shift; b=bs, NH=D, area=area)
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef=0.0)
    @test σxy_s ≈ σxy_ed atol = 0.03
end

@testset "trivial phase: KPM tracks the exact (non-quantized) ED value" begin
    # m > 3√3 t2 sinϕ puts the model in the C = 0 phase. At this small size
    # the gap is only ≈ 0.56, so σ_xy carries visible finite-size corrections
    # (≈ -0.11 e²/h, not exactly 0) — KPM must reproduce the exact ED value,
    # not the idealized thermodynamic-limit one.
    Ht, Jxt, Jyt, areat = haldane_model(6, 6; t=1.0, t2=t2, ϕ=π/2, m=1.6)
    σxy_ed = ed_hall_conductivity_T0(Ht, Jxt, Jyt, areat; Ef=0.0)
    @test abs(σxy_ed) < 0.2   # clearly not ±1

    at, bt, Ht_norm = KPM.normalizeH(Ht; center=true)
    mu_t = KPM.kpm_2d(Ht_norm, Jxt, Jyt, NC, D, D; psi_in=psi)
    σxy_kpm = KPM.kubo_bastin_cond(mu_t, at, 0.0; b=bt, NH=D, area=areat)
    @test σxy_kpm ≈ σxy_ed atol = 0.03
end

@testset "σ_xx: zero in the gap, matches broadened ED in the band" begin
    # insulating: no longitudinal response at Ef in the gap
    σxx_gap = KPM.kubo_bastin_cond(mu2D_xx, a, 0.0; b=b, NH=D, area=area)
    @test abs(σxx_gap) < 1e-4

    # metallic: Lorentz kernel λ ↔ Lorentzian broadening η = a λ √(1-x_F²)/NC
    Ef = -2.0
    xf = (Ef - b) / a
    for λ in (3.0, 4.0)
        η = a * λ * sqrt(1 - xf^2) / NC
        σxx_kpm = KPM.kubo_bastin_cond(mu2D_xx, a, Ef; b=b, NH=D, area=area,
                                       kernel=KPM.LorentzKernels(λ))
        σxx_ed = ed_kubo_bastin_broadened(H, Jx, Jx, area; Ef=Ef, eta=η)
        @test σxx_kpm ≈ σxx_ed rtol = 0.15
    end
end
