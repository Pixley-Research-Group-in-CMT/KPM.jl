# GPU validation of the CUDA extension — run manually on a CUDA machine:
#   julia --project=<env with KPM + CUDA + Arpack> test/gpu_smoke.jl
#
# Not part of the CI suite (needs a functional GPU). Checks that every GPU
# code path produces the same physics as the CPU path on the same inputs:
#   1. kpm_1d moments, GPU vs CPU (deterministic probes)
#   2. DoS reconstruction on the device
#   3. kpm_2d moments + kubo_bastin_cond: quantized Hall plateau from GPU moments
#   4. dc_long
#   5. a timing comparison on a larger chain (informational)

using Test
using LinearAlgebra
using SparseArrays
using Random
using CUDA
using KPM

@assert CUDA.functional() "gpu_smoke.jl needs a functional CUDA GPU"
@assert KPM.whichcore() "CUDA extension did not activate the GPU device"
println("== GPU: ", CUDA.name(CUDA.device()), " | ", Base.format_bytes(CUDA.total_memory()), " ==")

include("ed_reference.jl")
using .EDReference

# helper: evaluate f() with the device temporarily forced to CPU
function on_cpu(f)
    dev = KPM.ACTIVE_DEVICE[]
    KPM.ACTIVE_DEVICE[] = KPM.CPUDevice()
    try
        return f()
    finally
        KPM.ACTIVE_DEVICE[] = dev
    end
end

@testset "kpm_1d: GPU moments == CPU moments" begin
    rng = Xoshiro(42)
    NH = 4096
    NC = 256
    NR = 8
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    psi_in = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(psi_in, NR)

    mu_gpu = KPM.kpm_1d(H, NC, NR; psi_in=copy(psi_in))
    mu_cpu = on_cpu(() -> KPM.kpm_1d(H, NC, NR; psi_in=copy(psi_in)))
    @test mu_gpu ≈ mu_cpu atol = 1e-8
end

@testset "dos: device reconstruction matches host" begin
    rng = Xoshiro(7)
    NH = 4096
    NC = 256
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    psi_in = exp.(2im * pi * rand(rng, NH, 16))
    KPM.normalize_by_col(psi_in, 16)
    mu = on_cpu(() -> KPM.kpm_1d(H, NC, 16; psi_in=psi_in))

    E_g, rho_g = KPM.dos(mu, 1.0; N_tilde=512)
    E_c, rho_c = on_cpu(() -> KPM.dos(mu, 1.0; N_tilde=512))
    @test rho_g ≈ rho_c atol = 1e-8
    dE = E_g[2] - E_g[1]
    @test isapprox(sum(rho_g) * dE, 1.0; atol=2e-2)   # sum rule on device
end

@testset "quantum Hall from GPU kpm_2d moments" begin
    H, Jx, Jy, area = haldane_model(6, 6; t=1.0, t2=0.2, ϕ=π/2, m=0.0)
    D = size(H, 1)
    a, b, H_norm = KPM.normalizeH(H; center=true)
    NC = 96
    psi = Matrix{ComplexF64}(I, D, D)         # deterministic full trace

    mu_gpu = KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in=psi)
    mu_cpu = on_cpu(() -> KPM.kpm_2d(H_norm, Jx, Jy, NC, D, D; psi_in=psi))
    @test mu_gpu ≈ mu_cpu atol = 1e-7

    σxy_gpu = KPM.kubo_bastin_cond(mu_gpu, a, 0.0; b=b, NH=D, area=area)
    σxy_ed = ed_hall_conductivity_T0(H, Jx, Jy, area; Ef=0.0)
    @test σxy_gpu ≈ σxy_ed atol = 0.03        # quantized plateau, GPU end to end
end

@testset "dc_long on GPU" begin
    rng = Xoshiro(11)
    NH = 2048
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    # bond current for the chain: (Jx)_ij = H_ij (i - j)
    Jx = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(0.45 + 0im, NH - 1))
    psi_in = exp.(2im * pi * rand(rng, NH, 4))
    KPM.normalize_by_col(psi_in, 4)

    c_gpu = KPM.dc_long(H, Jx, 1.0, [64, 128], 4, NH; psi_in=copy(psi_in))
    c_cpu = on_cpu(() -> KPM.dc_long(H, Jx, 1.0, [64, 128], 4, NH; psi_in=copy(psi_in)))
    @test c_gpu ≈ c_cpu atol = 1e-6
end

println("\n== timing: 1D chain, NH = 2^17, NC = 1024, NR = 8 ==")
let
    NH = 2^17
    NC = 1024
    NR = 8
    H = spdiagm(1 => fill(-0.45 + 0im, NH - 1), -1 => fill(-0.45 + 0im, NH - 1))
    rng = Xoshiro(1)
    psi_in = exp.(2im * pi * rand(rng, NH, NR))
    KPM.normalize_by_col(psi_in, NR)

    KPM.kpm_1d(H, 64, NR; psi_in=psi_in)                    # GPU warm-up
    t_gpu = @elapsed KPM.kpm_1d(H, NC, NR; psi_in=psi_in)
    on_cpu(() -> KPM.kpm_1d(H, 64, NR; psi_in=psi_in))      # CPU warm-up
    t_cpu = @elapsed on_cpu(() -> KPM.kpm_1d(H, NC, NR; psi_in=psi_in))
    println("kpm_1d: GPU $(round(t_gpu, digits=2)) s | CPU $(round(t_cpu, digits=2)) s ",
            "| speedup ×$(round(t_cpu / t_gpu, digits=1))")
end

println("GPU SMOKE COMPLETE")
