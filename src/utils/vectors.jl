using Statistics, LinearAlgebra

function normalize_by_col(psi_in, NR; centering=true)
    for NRi in 1:NR
        psi_in_NRi = @view psi_in[:, NRi]
        psi_in_NRi .-= (mean(psi_in_NRi) * centering)
        psi_in_NRi ./= norm(psi_in_NRi)
    end
end

function gram_schmidt!(A)
    i_max = size(A, 2)
    Aviews = map(i -> view(A, :, i), 1:i_max)
    Aviews[1] ./= norm(Aviews[1])
    for (i, Aview) in enumerate(Aviews[2:end])
        prev_space = @view A[:,1:i]
        tmp = (prev_space' * Aview)
        mul!(Aview, prev_space, tmp, -1, 1)
        Aview ./= norm(Aview)
    end
end

function gram_schmidt(A)
    A = copy(A)
    gram_schmidt!(A)
    return A
end

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

ArrTypes = Union{Array, SubArray}
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

r_i(n) = mod(n - 1, 3) + 1
r_ip(n) = mod(n - 2, 3) + 1
r_ipp(n) = mod(n - 3, 3) + 1

r2_i(n) = mod(n - 1, 2) + 1
r2_ip(n) = mod(n - 2, 2) + 1
r2_ipp(n) = mod(n - 1, 2) + 1
