using CUDA
using Logging
using LoopVectorization


ASA = Array{T} where {T <: SubArray}

"""
evaluate from 3 to n
ψall[:,3] comes from ψall[:,2] and ψall[:,1]
separating into two function might improve performance (or not???)
"""
function chebyshev_iter(H,
                        ψall::Union{Array{T, 2}, CuArray{T, 2}} where T,
                        n::Int64)
    for i in 3:n
        chebyshev_iter_single(H, ψall, i-2, i-1, i)
    end
end

function chebyshev_iter(H,
                        ψviews::Array{T} where {T <: Union{CuArray, SubArray}},
                        n::Int64)
    for i in 3:n
        chebyshev_iter_single(H, ψviews[i-2], ψviews[i-1], ψviews[i])
    end
end



function chebyshev_iter_wrap(H,
                             ψall::Union{Array{T, 2}, CuArray{T, 2}} where T,
                             n::Int64)
    chebyshev_iter_single(H, ψall, n - 1, n, 1)
    chebyshev_iter_single(H, ψall, n, 1, 2)
end

function chebyshev_iter_wrap(H,
                             ψviews::Array{T} where {T <: Union{CuArray, SubArray}},
                             n::Int64)
    chebyshev_iter_single(H, ψviews[n-1], ψviews[n], ψviews[1])
    chebyshev_iter_single(H, ψviews[n], ψviews[1], ψviews[2])
end


chebyshev_iter_wrap(H, ψall) = chebyshev_iter_wrap(H, ψall, size(ψall)[1])

# num indicater ver.
# pp, p -> pp
function chebyshev_iter_single(H,
                               V_all::Union{Array, SubArray, CuArray},
                               i_pp_in::Int64,
                               i_p_in::Int64)
    V_p_in = @view V_all[:, :, i_p_in] # [NH, NR, indexing]
    V_out = @view V_all[:, :, i_pp_in] 
    chebyshev_iter_single(H, V_out, V_p_in)
    return nothing
end

# SubArray and CuArray are all pointer-like
# V_out is V_pp
function chebyshev_iter_single(H,
                               V_pp_in::CuArray,
                               V_p_in::CuArray)
    mul!(V_pp_in, H, V_p_in, 2.0+0im, -1.0+0im)
    return nothing
end

## experimenting multi-threading on CPU. 
function chebyshev_iter_single(H,
                               V_pp_in::SubArray,
                               V_p_in::SubArray)

    Threads.@threads for i = 1:size(V_pp_in, 2)
        @debug "i = $i on thread $(Threads.threadid())"
        mul!(view(V_pp_in, :, i), H, view(V_p_in, :, i), 2.0+0im, -1.0+0im)
    end
    return nothing
end

# for better multi-threading performance: avoid generating views
function chebyshev_iter_single(H,
                               V_pp_in::ASA,
                               V_p_in::ASA)

    Threads.@threads for i = 1:size(V_pp_in)
        @debug "i = $i on thread $(Threads.threadid())"
        mul!(V_pp_in[i], H, V_p_in[i], 2.0+0im, -1.0+0im)
    end
    return nothing
end

# num indicater ver.
# pp, p -> out
function chebyshev_iter_single(H, V_all::Union{Array, SubArray, CuArray}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    (@view V_all[:, :, i_out]) .= (@view V_all[:, :, i_pp_in])
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end
function chebyshev_iter_single(H, V_all::Array{T} where {T <: ASA}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    V_all[i_out] .= V_all[i_pp_in]
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

# SubArray and CuArray are all pointer-like
function chebyshev_iter_single(H,
                               V_pp_in::Union{SubArray, ASA},
                               V_p_in::Union{SubArray, ASA},
                               V_out::Union{SubArray,ASA})
    V_out .= V_pp_in
    chebyshev_iter_single(H, V_out, V_p_in)
end


function chebyshev_iter_single(H,
                               V_pp_in::CuArray,
                               V_p_in::CuArray,
                               V_out::CuArray)
    V_out .= V_pp_in
    chebyshev_iter_single(H, V_out, V_p_in)
end
