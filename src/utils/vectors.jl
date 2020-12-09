using CUDA

"""
Normalize a collection of vectors in an (NH, NR) array `psi_in`,
where each column (that is `psi_in[:, NRi]`) represent a separate
vector.
"""
function normalize_by_col(psi_in, NR)
    # TODO: possible GPU optimization
    for NRi in 1:NR
        psi_in_NRi = @view psi_in[:, NRi]
        psi_in_NRi ./= norm(psi_in_NRi)
    end
end


"""
Dot product each column of Vls with vector Vr, save in target.

target: 2D Array (n, NR), n >= ncols.
Vls: 3D Array, shape (NH, NR, n), where n >= ncols.
Vr: 2D Array, shape NH, NR
ncols: Integer, number of columns. 
"""
function broadcast_dot_2d_1d!(target::Array{T, 2} where T,
                              Vls::Array{T, 3} where T,
                              Vr::Array{T, 2} where T,
                              NR::Int64, ncols::Int64)
    for i in 1:ncols
        for NRi in 1:NR
            target[i, NRi] = dot(view(Vls, :, NRi, i), view(Vr, :, NRi))
        end
    end
    return nothing
end
function broadcast_dot_2d_1d!(target::CuArray{T, 2} where T,
                              Vls::CuArray{T, 3} where T,
                              Vr::CuArray{T, 2} where T,
                              NR::Int64, ncols::Int64)
    #@cuda threads=(ncols, NR) broadcast_dot_2d_1d_gpu!(target, Vls, Vr, NR, ncols)
    target_temp = on_host_zeros(ncols, NR)
    for i in 1:ncols
        for NRi in 1:NR
            target_temp[i, NRi] = dot(view(Vls, :, NRi, i), view(Vr, :, NRi))
        end
    end
    target .= maybe_to_device(target_temp)
    return nothing
end
# TODO: this is not working
#function broadcast_dot_2d_1d_gpu!(target, Vls, Vr, NR, ncols)
    #i = blockIdx().x
    #NRi = blockIdx().y
    ##individual = 
    #@inbounds target[i, NRi] = dot(view(Vls, :, NRi, i), view(Vr, :, NRi))
    #return nothing
#end

"""
Dot product each column of Vls with vector Vr, save in target.
Each view has NR replica of NH. This function take the average.

target: 1D Array (n), n >= ncols.
Vls: 1D Array of 2D views, shape (n), each view (NH, NR), where n >= ncols.
Vr: 2D Array, shape NH, NR
ncols: Integer, number of columns. 
"""
function broadcast_dot_reduce_avg_2d_1d!(target::Array{T, 1} where T,
                                         Vls::Array{T, 1} where {T<:SubArray{Ts, 2} where Ts},
                                         Vr::Array{T, 2} where T,
                                         NR::Int64, ncols::Int64)
    for i in 1:ncols
        target[i] = dot(Vls[i], Vr) / NR
    end
    return nothing
end

function broadcast_dot_reduce_avg_2d_1d!(target::CuArray{T, 1} where T,
                                         Vls::Array{T, 1} where {T<:CuArray{Ts, 2} where Ts},
                                         Vr::CuArray{T, 2} where T,
                                         NR::Int64, ncols::Int64)
    target_temp = on_host_zeros(ncols)
    for i in 1:ncols
        target_temp[i] = dot(Vls[i], Vr) / NR
    end
    target .= maybe_to_device(target_temp)
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
    for NRi in 1:NR
        target[NRi] = dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta
    end
    return nothing
end

function broadcast_dot_1d_1d!(target::CuArray,
                              Vl::CuArray,
                              Vr::CuArray,
                              NR::Int64,
                              alpha::Number=1.0,
                              beta::Number=0.0)
    target_temp = on_host_zeros(NR)
    for NRi in 1:NR
        target_temp[NRi] = dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta
    end
    target .= maybe_to_device(target_temp)
    return nothing
end

function broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                              Vl::Union{Array, SubArray},
                              Vr::Union{Array, SubArray},
                              NR::Int64,
                              alpha::Number,
                              beta::Union{Array, SubArray})
    for NRi in 1:NR
        target[NRi] = dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta[NRi]
    end
    return nothing
end


function broadcast_dot_1d_1d!(target::CuArray,
                              Vl::CuArray,
                              Vr::CuArray,
                              NR::Int64,
                              alpha::Number,
                              beta::CuArray)
    target_temp = on_host_zeros(NR)
    for NRi in 1:NR
        target_temp[NRi] = dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha
    end
    target .= maybe_to_device(target_temp)
    target .+= beta
    return nothing
end
