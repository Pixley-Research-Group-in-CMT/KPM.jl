using KPM
using Test

@testset "kernels/kernels.jl" begin
    include("test_kernel.jl")
end

@testset "KPM.jl" begin
    include("integration_test.jl")
end

@testset "util/chebyshev_iteration.jl" begin
    include("chebyshev_iteration_test.jl")
end

