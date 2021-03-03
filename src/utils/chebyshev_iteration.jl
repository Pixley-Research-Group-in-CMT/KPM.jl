using CUDA

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
function chebyshev_iter(H,
                        ψviews::Array{T} where {T <: BufferedSubArray},
                        n::Int64)
    load_buffer!(ψviews[1])
    load_buffer!(ψviews[2])
    for i in 3:n
        _chebyshev_iter_single_with_buffer(H, ψviews[i-2], ψviews[i-1], ψviews[i])
    end
    unload_buffer!(ψviews[n-1])
    unload_buffer!(ψviews[n])
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

function chebyshev_iter_wrap(H,
                             ψviews::Array{T} where {T <: BufferedSubArray},
                             n::Int64)
    load_buffer!(ψviews[n-1])
    load_buffer!(ψviews[n])
    _chebyshev_iter_single_with_buffer(H, ψviews[n-1], ψviews[n], ψviews[1])
    _chebyshev_iter_single_with_buffer(H, ψviews[n], ψviews[1], ψviews[2])
    unload_buffer!(ψviews[1])
    unload_buffer!(ψviews[2])
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
# Only on that does mul! (i.e. all others dispatched here)
# `get_buffer` has null effect when passing SubArray or CuArray
# BufferedSubArray must be loaded before calling, and will diverge from
# the base array. Need unload.
function chebyshev_iter_single(H,
                               V_pp_in::Union{SubArray, CuArray, BufferedSubArray},
                               V_p_in::Union{SubArray, CuArray, BufferedSubArray})
    mul!(get_buffer(V_pp_in),
         H,
         get_buffer(V_p_in),
         2.0+0im, -1.0+0im)
    return nothing
end

# num indicater ver.
# pp, p -> out
function chebyshev_iter_single(H, V_all, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    (@view V_all[:, :, i_out]) .= (@view V_all[:, :, i_pp_in])
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

# SubArray and CuArray are all pointer-like
function chebyshev_iter_single(H,
                               V_pp_in::Union{SubArray, CuArray},
                               V_p_in::Union{SubArray, CuArray},
                               V_out::Union{SubArray, CuArray})

    V_out .= V_pp_in
    chebyshev_iter_single(H, V_out, V_p_in)

end

# private, as the operation is highly unsafe.
function _chebyshev_iter_single_with_buffer(H,
                               V_pp_in::BufferedSubArray,
                               V_p_in::BufferedSubArray,
                               V_out::BufferedSubArray)

    # V_out is never seen before, so need checking out
    load_buffer!(V_out)
    get_buffer(V_out) .= get_buffer(V_pp_in)
    # V_pp_in is no longer used, unload
    unload_buffer!(V_pp_in)
    chebyshev_iter_single(H, V_out, V_p_in)

end
