using CUDA
using KPM.BufferedSubArrays

## TODO: test memory consumption

N = 16
A = rand(Float64, N, N, N)
Avs_all = [[view(A,1,:,:), view(A,2,:,:)],
       [view(A,:,4,:), view(A,:,2,:)],
       [view(A,2,4,:), view(A,:,2,8)],
       [view(A,:,:,6), view(A,:,:,3)]]

if CUDA.has_cuda()
    sa_types = [SubArray, CuArray]
else
    sa_types = [SubArray]
end

for sa_type in sa_types
    println("test - $(sa_type)")
    for Avs in Avs_all
        println("test - $(typeof(Avs))")
        # Part 1: test working on one BSA
        Avs_bsa = BufferedSubArray{sa_type}.(Avs)
        @test_throws AssertionError get_buffer(Avs_bsa[1])
        load_buffer!(Avs_bsa[1])
        get_buffer(Avs_bsa[1]) .+= 1
        if sa_type == CuArray
            # not yet updated
            @test all(Avs_bsa[1].subarray .<= 1)
            @test all(A .<= 1)
        end
        unload_buffer!(Avs_bsa[1])
        # updated, and reflected on main Array
        @test all(Avs_bsa[1].subarray .>= 1)
        @test any(A .>= 1) & any(A .< 1)
        A .= rand(Float64, N, N, N) # reset back to random ∈ [0,1]


        # Part 2: test two BSA sharing buffer array. When using CPU, no
        # real sharing is happenning. 
        Avs_template = Avs_bsa[1]
        Avs_bsa = BufferedSubArray{sa_type}.(Avs; shared_with=Avs_template)
        # none is checked out, registry empty
        @test !any(ischeckedout, Avs_bsa)
        @test all(x -> !any(x.buffer_registry), Avs_bsa)

        load_buffer!.(Avs_bsa)
        @test all(ischeckedout, Avs_bsa)
        if sa_type == CuArray
            # test sharing
            @test Avs_bsa[1].buffer_registry ≡ Avs_bsa[2].buffer_registry
            @test Avs_bsa[1].buffer ≡ Avs_bsa[2].buffer
            # test non-conflict sharing
            @test Avs_bsa[1].buffer_idx != Avs_bsa[2].buffer_idx
            @test sum(Avs_bsa[1].buffer_registry) == 2
        end

        @test all(Avs_bsa[1].subarray .!== Avs_bsa[2].subarray) # in rare occasion, the randomness may make them the same
        get_buffer(Avs_bsa[1]) .= get_buffer(Avs_bsa[2])
        if sa_type == CuArray
            @test all(Avs_bsa[1].subarray .!== Avs_bsa[2].subarray) # not updated
        end
        unload_buffer!.(Avs_bsa)
        @test all(Avs_bsa[1].subarray .== Avs_bsa[2].subarray) # updated
        A .= rand(Float64, N, N, N) # reset back to random ∈ [0,1]
    end
end
