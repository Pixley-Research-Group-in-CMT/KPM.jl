using Test
using SparseArrays
using CUDA
using KPM

const PROJECT_TOML_TEXT = read(joinpath(@__DIR__, "..", "Project.toml"), String)

# CUDA is in the test target, so the extension is loaded and compiled here
# even on CPU-only machines (CUDA.functional() == false). Actual GPU
# execution is only exercised when a functional device is present.
@testset "CUDA extension wiring" begin
    @test occursin("[weakdeps]", PROJECT_TOML_TEXT)
    @test occursin("KPMCUDAExt = \"CUDA\"", PROJECT_TOML_TEXT)
    @test Base.get_extension(KPM, :KPMCUDAExt) !== nothing

    if !CUDA.functional()
        @test KPM.whichcore() == false

        a = [1.0, 2.0]
        @test KPM.maybe_to_device(a) === a
        @test KPM.maybe_to_host(a) === a

        s = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
        @test KPM.maybe_to_device(s) === s
        @test KPM.maybe_to_host(s) === s

        z = KPM.maybe_on_device_zeros(Float64, 2, 2)
        @test z isa Array{Float64, 2}
    else
        @test KPM.whichcore() == true
        # GPU smoke test: moments of a trivial diagonal H, real input matrix
        # (exercises the eltype cast to the complex workspace)
        H = sparse([1, 2], [1, 2], [0.3, -0.3], 2, 2)
        mu = KPM.kpm_1d(H, 8, 2)
        @test isapprox(mu[1], 1.0; atol=1e-6)
    end
end
