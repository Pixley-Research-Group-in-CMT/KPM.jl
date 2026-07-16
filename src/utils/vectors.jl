using Statistics, LinearAlgebra

"""
Normalize (in place) a collection of vectors in an (NH, NR) array `psi_in`,
where each column `psi_in[:, NRi]` is a separate vector.

`centering=true` subtracts each column's mean first. Note this makes every
vector exactly orthogonal to the uniform state, which biases the stochastic
trace estimator by a rank-one projection (the uniform component of the
spectrum is silently removed) — an O(1/D) effect in general, but exact and
arbitrarily large for operators with weight on the uniform state. The
default is therefore `centering=false`, which keeps E[ψψ†] ∝ I.
"""
function normalize_by_col(psi_in, NR; centering=false)
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


const ArrTypes = Union{Array, SubArray}

"""
Column-pairwise dots: `target[i] = alpha * dot(Vl_arr[i], Vr_arr[i]) + beta[i]`,
with `beta` a number or a vector. `Vl_arr`/`Vr_arr` are arrays of column views.
"""
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

# Ring-buffer index helpers: 3-slot ring ...
r_i(n) = mod(n - 1, 3) + 1
r_ip(n) = mod(n - 2, 3) + 1
r_ipp(n) = mod(n - 3, 3) + 1

# ... and 2-slot ring, where slot ipp is always overwritten by slot i.
r2_i(n) = mod(n - 1, 2) + 1
r2_ip(n) = mod(n - 2, 2) + 1
r2_ipp(n) = mod(n - 1, 2) + 1
