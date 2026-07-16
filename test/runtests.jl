using KPM
using Test

@testset "kernels/kernels.jl" begin
    include("test_kernel.jl")
end

@testset "util/chebyshev_iteration.jl" begin
    include("chebyshev_iteration_test.jl")
end

@testset "applications/dos.jl" begin
    include("dos_test.jl")
end

@testset "applications/conductivity.jl" begin
    include("cond_test.jl")
end

@testset "Kubo–Bastin vs exact diagonalization" begin
    include("kubo_bastin_test.jl")
end

@testset "KPM.jl" begin
    include("integration_test.jl")
end

@testset "typed front end (frontend.jl)" begin
    include("frontend_test.jl")
end

@testset "optional CUDA extension migration" begin
    include("optional_cuda_test.jl")
end
