using Test
using SparseArrays

const PROJECT_TOML_TEXT = read(joinpath(@__DIR__, "..", "Project.toml"), String)

@testset "optional CUDA defaults" begin
    @test KPM.whichcore() == false

    a = [1.0, 2.0]
    @test KPM.maybe_to_device(a) === a
    @test KPM.maybe_to_host(a) === a

    s = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
    @test KPM.maybe_to_device(s) === s
    @test KPM.maybe_to_host(s) === s

    z = KPM.maybe_on_device_zeros(Float64, 2, 2)
    @test z isa Array{Float64, 2}
    
    @test occursin("[weakdeps]", PROJECT_TOML_TEXT)
    @test occursin("KPMCUDAExt = \"CUDA\"", PROJECT_TOML_TEXT)
end