using Test
using LinearAlgebra
using SparseArrays
using Random
using KPM

# Deterministic checks of the moment recurrence (including the doubling trick)
# against direct evaluation, plus physics checks of the reconstructed DoS
# against exact diagonalization and the analytic 1D-chain result.

@testset "kpm_1d moments: doubling trick vs direct recurrence" begin
    rng = Xoshiro(42)
    NH = 64
    NC = 32
    A = randn(rng, ComplexF64, NH, NH)
    Hd = Matrix((A + A') / 2)
    a = maximum(abs, eigvals(Hermitian(Hd))) / 0.95
    Hd ./= a
    H = sparse(Hd)

    ψ = randn(rng, ComplexF64, NH)
    ψ ./= norm(ψ)
    psi_in = reshape(ψ, NH, 1)

    mu = KPM.kpm_1d(H, NC, 1; psi_in=psi_in)

    # direct three-term recurrence, no doubling
    mu_direct = zeros(Float64, NC)
    α_prev = copy(ψ)
    α_curr = Hd * ψ
    mu_direct[1] = real(dot(ψ, α_prev))
    mu_direct[2] = real(dot(ψ, α_curr))
    for n in 3:NC
        α_prev, α_curr = α_curr, 2 .* (Hd * α_curr) .- α_prev
        mu_direct[n] = real(dot(ψ, α_curr))
    end
    @test mu ≈ mu_direct atol = 1e-10

    # same moments from exact diagonalization: μn = Σ_i |⟨v_i|ψ⟩|² T_n(λ_i)
    λ, V = eigen(Hermitian(Hd))
    w = abs2.(V' * ψ)
    mu_ed = [sum(w .* cos.(n .* acos.(clamp.(λ, -1, 1)))) for n in 0:(NC - 1)]
    @test mu ≈ mu_ed atol = 1e-8
end

@testset "kpm_1d: NR_parallel=false and avg_output=false" begin
    rng = Xoshiro(1)
    NH = 32
    NC = 16
    NR = 3
    A = randn(rng, ComplexF64, NH, NH)
    H = sparse((A + A') / (2 * NH))  # comfortably inside (-1, 1)

    psi_in = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(psi_in, NR)

    mu_par = KPM.kpm_1d(H, NC, NR; psi_in=copy(psi_in))
    mu_ser = KPM.kpm_1d(H, NC, NR; psi_in=copy(psi_in), NR_parallel=false)
    @test mu_par ≈ mu_ser atol = 1e-12

    mu_all = KPM.kpm_1d(H, NC, NR; psi_in=copy(psi_in), avg_output=false)
    @test size(mu_all) == (NR, NC)
    @test vec(real.(sum(mu_all, dims=1) ./ NR)) ≈ mu_par atol = 1e-12
end

@testset "stochastic trace is unbiased (no mean-centering of probes)" begin
    # H = 0.8 |u><u| with u the uniform state: mean-centered probe vectors
    # are orthogonal to u and give exactly mu_1 = 0 instead of 0.8/D.
    rng = Xoshiro(21)
    D = 8
    u = fill(1 / sqrt(D), D)
    H = 0.8 * u * u'
    NR = 512
    psi_in = exp.(2im * pi * rand(rng, D, NR))
    KPM.normalize_by_col(psi_in, NR)
    mu = KPM.kpm_1d(H, 4, NR; psi_in=psi_in)
    @test isapprox(mu[2], 0.8 / D; rtol=0.3)   # biased estimator gives ~0
end

@testset "kpm_1d moments scale as <psi|psi> for unnormalized input" begin
    rng = Xoshiro(13)
    NH = 32
    NC = 16
    A = randn(rng, ComplexF64, NH, NH)
    H = sparse((A + A') / (2 * NH))
    ψ = randn(rng, ComplexF64, NH)
    ψ ./= norm(ψ)
    mu1x = KPM.kpm_1d(H, NC, 1; psi_in=reshape(ψ, NH, 1))
    mu2x = KPM.kpm_1d(H, NC, 1; psi_in=reshape(2ψ, NH, 1))
    # mu_n = <psi|T_n|psi> scales by |c|^2; the old hardcoded mu_0 = 1 mixed
    # normalized and unnormalized terms in the doubling identities
    @test mu2x ≈ 4 .* mu1x rtol = 1e-10
end

@testset "dos energy derivative (dE_order=1)" begin
    rng = Xoshiro(17)
    NH = 256
    NC = 64
    A = randn(rng, ComplexF64, NH, NH)
    H = sparse((A + A') / 2)
    a, H_norm = KPM.normalizeH(H)
    mu = KPM.kpm_1d(H_norm, NC, 8)

    E_grid = collect(range(-0.4a, 0.4a; length=9))
    _, drho = KPM.dos(mu, a; E_grid=E_grid, N_tilde=length(E_grid), dE_order=1)
    @test all(isfinite, drho)

    # compare against a centered finite difference of the dE_order=0 curve
    h = 1e-4 * a
    _, rp = KPM.dos(mu, a; E_grid=E_grid .+ h, N_tilde=length(E_grid))
    _, rm = KPM.dos(mu, a; E_grid=E_grid .- h, N_tilde=length(E_grid))
    fd = (rp .- rm) ./ (2h)
    @test drho ≈ fd rtol = 1e-5
end

@testset "normalizeH: hermiticity check and center shift" begin
    rng = Xoshiro(3)
    A = sprandn(rng, ComplexF64, 100, 100, 0.05)
    @test_throws ArgumentError KPM.normalizeH(A + 2 * A')

    H = A + A'
    a, H_norm = KPM.normalizeH(H)
    @test maximum(abs, eigvals(Hermitian(Matrix(H_norm)))) < 1

    shift = 0.7
    Hs = H + shift * I
    a, b, Hs_norm = KPM.normalizeH(Hs; center=true)
    ev = eigvals(Hermitian(Matrix(Hs_norm)))
    @test abs(b - shift) < 0.1 * a
    @test minimum(ev) > -1 && maximum(ev) < 1
    # centered rescaling should use the margin symmetrically
    @test abs(maximum(ev) + minimum(ev)) < 0.02
end

@testset "DoS of the 1D chain: analytic form, sum rule, center shift" begin
    # N and NC chosen so the Jackson resolution ~ π·2a/NC ≈ 0.05 stays well
    # above the level spacing ~ 4πt/N ≈ 0.003: the discrete spectrum then
    # reconstructs as the smooth thermodynamic-limit DoS.
    N = 4000
    t = 1.0
    shift = 0.7
    H = spdiagm(0 => fill(shift + 0im, N),
                1 => fill(-t + 0im, N - 1),
                -1 => fill(-t + 0im, N - 1))
    H[1, N] = -t
    H[N, 1] = -t

    a, b, H_norm = KPM.normalizeH(H; center=true)
    @test abs(b - shift) < 0.05

    NC = 256
    NR = 24
    rng = Xoshiro(7)
    psi_in = exp.(2im * pi * rand(rng, N, NR))
    KPM.normalize_by_col(psi_in, NR)
    mu = KPM.kpm_1d(H_norm, NC, NR; psi_in=psi_in)

    E, rho = KPM.dos(mu, a; b=b, N_tilde=1024)

    # sum rule: ∫ρ dE = 1
    dE = E[2] - E[1]
    @test isapprox(sum(rho) * dE, 1.0; atol=2e-2)

    # analytic DoS of the shifted chain, away from the van Hove edges
    rho_exact(e) = abs(e - shift) < 2t ? 1 / (pi * sqrt(4t^2 - (e - shift)^2)) : 0.0
    idx = findall(e -> abs(e - shift) < 1.5 * t, E)
    @test maximum(abs.(rho[idx] .- rho_exact.(E[idx]))) < 0.02
end

@testset "kpm_1d!: workspace reuse allocates no large buffers" begin
    rng = Xoshiro(11)
    NC = 64
    NR = 4

    measure_allocs = function (NH)
        H = spdiagm(1 => fill(-0.4 + 0im, NH - 1), -1 => fill(-0.4 + 0im, NH - 1))
        psi_in = exp.(2im * pi * rand(rng, NH, NR))
        KPM.normalize_by_col(psi_in, NR)
        mu_all = zeros(ComplexF64, NR, NC)
        α_all = zeros(ComplexF64, NH, NR, 2)
        KPM.kpm_1d!(H, NC, NR, NH, mu_all, psi_in; α_all=α_all)  # warm up
        @allocated KPM.kpm_1d!(H, NC, NR, NH, mu_all, psi_in; α_all=α_all)
    end

    NH_small, NH_big = 1024, 4096
    allocs_small = measure_allocs(NH_small)
    allocs_big = measure_allocs(NH_big)

    # per-call overhead (views, broadcast temporaries) is O(NC), independent
    # of the workspace size: a single leaked NH_big×NR buffer would add
    # ≥ (NH_big - NH_small)·NR·16 bytes going from NH_small to NH_big
    @test allocs_big - allocs_small < (NH_big - NH_small) * NR * 16 ÷ 2
end
