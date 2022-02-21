using CUDA
using Statistics, LinearAlgebra

"""
Normalize a collection of vectors in an (NH, NR) array `psi_in`,
where each column (that is `psi_in[:, NRi]`) represent a separate
vector.
"""
function normalize_by_col(psi_in, NR; centering=true)
    # TODO: possible GPU optimization
    for NRi in 1:NR
        psi_in_NRi = @view psi_in[:, NRi]
        psi_in_NRi .-= (mean(psi_in_NRi) * centering)
        psi_in_NRi ./= norm(psi_in_NRi)
    end
end


"""
orthonormalize the column vectors of `A`. In-place.

Using classical Gram-Schmit.

When orthogonality is extremely important, applying the same
method twice may help, according to
[this note](http://stoppels.blog/posts/orthogonalization-performance).

"""
function gram_schmidt!(A)
    i_max = size(A, 2)
    Aviews = map(i -> view(A, :, i), 1:i_max)
    Aviews[1] ./= norm(Aviews[1])
    for (i, Aview) in enumerate(Aviews[2:end])
        prev_space = @view A[:,1:i]
        tmp = (prev_space' * Aview)
        # allocates N for number of columns.

        mul!(Aview, prev_space, tmp, -1, 1)
        Aview ./= norm(Aview)
    end
end

"""
orthonormalize the column vectors of `A`. See `gram_schmidt!`, the in-place
version for details.
"""
function gram_schmidt(A)
    A = copy(A)
    gram_schmidt!(A)
    return A
end


"""
Dot product each column of Vls with vector Vr, save in target.
Each view has NR replica of NH. This function take the average.

target: 1D Array (n), n >= NCcols.
Vls: 1D Array of 2D views, shape (n), each view (NH, NR), where n >= NCcols.
Vr: 2D Array, shape NH, NR
NCcols: Integer, number of columns. 
"""
function broadcast_dot_reduce_avg_2d_1d!(target::Union{Array, SubArray},
                                         Vls::Array{T, 1} where {T<:SubArray{Ts, 2} where Ts},
                                         Vr::Array{T, 2} where T,
                                         NR::Int64, NCcols::Int64;
                                         NC0::Int64=1, NCstep::Int64=1
                                        )
    Threads.@threads for i in NC0:NCstep:NCcols
        target[i] = dot(Vls[i], Vr) / NR
    end
    return nothing
end

function broadcast_dot_reduce_avg_2d_1d!(target::Union{Array, SubArray},
                                         Vls::Array{T, 1} where {T<:CuArray{Ts, 2} where Ts},
                                         Vr::CuArray{T, 2} where T,
                                         NR::Int64, NCcols::Int64;
                                         NC0::Int64=1, NCstep::Int64=1
                                        )
    target[NC0:NCstep:NCcols] .= dot.(Vls[NC0:NCstep:NCcols], [Vr])
    target ./= NR
    return nothing
end



"""
Vl and Vr are both [NH, NR] sized array. Each corresponding [:, NR] slice
pair is dotted, saving into the target of [NR], multiplying by alpha and
plus beta. Beta is either a number or vector of [NR]

"""
function broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                              Vl::Union{Array, SubArray},
                              Vr::Union{Array, SubArray},
                              NR::Int64,
                              alpha::Number=1.0,
                              beta::Number=0.0)
    println("deprecated: `broadcast_dot_1d_1d!` with `NR` - 0")
    Threads.@threads for NRi in 1:NR
        target[NRi] = dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta
    end
    return nothing
end

function broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                              Vl::CuArray,
                              Vr::CuArray,
                              NR::Int64,
                              alpha::Number=1.0,
                              beta::Number=0.0)
    println("deprecated: `broadcast_dot_1d_1d!` with `NR` - 1")
    for NRi in 1:NR
        target[NRi] = (dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta)
    end
    return nothing
end

function broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                              Vl::Union{Array, SubArray},
                              Vr::Union{Array, SubArray},
                              NR::Int64,
                              alpha::Number,
                              beta::Union{Array, SubArray})
    println("deprecated: `broadcast_dot_1d_1d!` with `NR` - 2")
    Threads.@threads for NRi in 1:NR
        target[NRi] = dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta[NRi]
    end
    return nothing
end


function broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                              Vl::CuArray,
                              Vr::CuArray,
                              NR::Int64,
                              alpha::Number,
                              beta::CuArray)
    println("deprecated: `broadcast_dot_1d_1d!` with `NR` - 3")
    for NRi in 1:NR
        target[NRi] = (dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha) + beta[NRi]
    end
    return nothing
end

ArrTypes = Union{Array, SubArray, CuArray}
function broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                              Vl_arr::Array{T} where {T <: ArrTypes},
                              Vr_arr::Array{T} where {T <: ArrTypes};
                              alpha::Number=1.0,
                              beta::Union{Number, T} where {T <: ArrTypes}=0.0)
    target .= dot.(Vl_arr, Vr_arr)
    target .*= alpha
    target .+= maybe_to_host(beta)
    return nothing
end

# Pointing
r_i(n) = mod(n - 1, 3) + 1
r_ip(n) = mod(n - 2, 3) + 1
r_ipp(n) = mod(n - 3, 3) + 1

# Pointing - 2 
r2_i(n) = mod(n - 1, 2) + 1
r2_ip(n) = mod(n - 2, 2) + 1
r2_ipp(n) = mod(n - 1, 2) + 1 # using this set of pointing, ipp will always be overwritten by i
