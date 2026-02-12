using KPM
using Test

@testset "kernels/kernels.jl" begin
    include("test_kernel.jl")
end

@testset "util/chebyshev_iteration.jl" begin
    include("chebyshev_iteration_test.jl")
end

@testset "KPM.jl" begin
    include("integration_test.jl")
end

@testset "optional CUDA extension migration" begin
    include("optional_cuda_test.jl")
end
