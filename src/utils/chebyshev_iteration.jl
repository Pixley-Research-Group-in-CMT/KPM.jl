using Logging
using LoopVectorization

ASA = Array{T} where {T <: SubArray}

function chebyshev_iter(H, ψall::AbstractArray{T, 2} where T, n::Int64)
    for i in 3:n
        chebyshev_iter_single(H, ψall, i-2, i-1, i)
    end
end

function chebyshev_iter(H, ψviews::Array{T} where {T <: SubArray}, n::Int64)
    for i in 3:n
        chebyshev_iter_single(H, ψviews[i-2], ψviews[i-1], ψviews[i])
    end
end

function chebyshev_iter_wrap(H, ψall::AbstractArray{T, 2} where T, n::Int64)
    chebyshev_iter_single(H, ψall, n - 1, n, 1)
    chebyshev_iter_single(H, ψall, n, 1, 2)
end

function chebyshev_iter_wrap(H, ψviews::Array{T} where {T <: SubArray}, n::Int64)
    chebyshev_iter_single(H, ψviews[n-1], ψviews[n], ψviews[1])
    chebyshev_iter_single(H, ψviews[n], ψviews[1], ψviews[2])
end

chebyshev_iter_wrap(H, ψall) = chebyshev_iter_wrap(H, ψall, size(ψall)[1])

function chebyshev_iter_single(H, V_all::Union{Array, SubArray}, i_pp_in::Int64, i_p_in::Int64)
    V_p_in = @view V_all[:, :, i_p_in]
    V_out = @view V_all[:, :, i_pp_in]
    chebyshev_iter_single(H, V_out, V_p_in)
    return nothing
end

function chebyshev_iter_single(H, V_pp_in::SubArray, V_p_in::SubArray)
    Threads.@threads for i = 1:size(V_pp_in, 2)
        @debug "i = $i on thread $(Threads.threadid())"
        mul!(view(V_pp_in, :, i), H, view(V_p_in, :, i), 2.0+0im, -1.0+0im)
    end
    return nothing
end

function chebyshev_iter_single(H, V_pp_in::ASA, V_p_in::ASA)
    Threads.@threads for i = 1:size(V_pp_in)
        @debug "i = $i on thread $(Threads.threadid())"
        mul!(V_pp_in[i], H, V_p_in[i], 2.0+0im, -1.0+0im)
    end
    return nothing
end

function chebyshev_iter_single(H, V_all::Union{Array, SubArray}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    (@view V_all[:, :, i_out]) .= (@view V_all[:, :, i_pp_in])
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

function chebyshev_iter_single(H, V_all::Array{T} where {T <: ASA}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    V_all[i_out] .= V_all[i_pp_in]
    chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

function chebyshev_iter_single(H, V_pp_in::Union{SubArray, ASA}, V_p_in::Union{SubArray, ASA}, V_out::Union{SubArray,ASA})
    V_out .= V_pp_in
    chebyshev_iter_single(H, V_out, V_p_in)
end
