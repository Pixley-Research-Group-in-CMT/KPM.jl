using KPM
using Test

@testset "util/buffered_sub_arrays.jl" begin
    include("buffered_sub_arrays_test.jl")
end

@testset "kernels/kernels.jl" begin
    include("test_kernel.jl")
end

@testset "util/chebyshev_iteration.jl" begin
    include("chebyshev_iteration_test.jl")
end

@testset "KPM.jl" begin
    include("integration_test.jl")
end
