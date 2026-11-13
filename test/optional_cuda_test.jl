using Test
using SparseArrays

const PROJECT_TOML_TEXT = read(joinpath(@__DIR__, "..", "Project.toml"), String)

@testset "optional CUDA defaults" begin
    @test KPMru.whichcore() == false

    a = [1.0, 2.0]
    @test KPMru.maybe_to_device(a) === a
    @test KPMru.maybe_to_host(a) === a

    s = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
    @test KPMru.maybe_to_device(s) === s
    @test KPMru.maybe_to_host(s) === s

    z = KPMru.maybe_on_device_zeros(Float64, 2, 2)
    @test z isa Array{Float64, 2}
    
    @test occursin("[weakdeps]", PROJECT_TOML_TEXT)
    @test occursin("KPMruCUDAExt = \"CUDA\"", PROJECT_TOML_TEXT)
end