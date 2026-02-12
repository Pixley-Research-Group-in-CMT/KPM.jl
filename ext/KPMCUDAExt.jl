module KPMCUDAExt

using KPM
using CUDA
using CUDA.CUSPARSE
using SparseArrays

function KPM.whichcore()
    if CUDA.has_cuda()
        println("GPU support for KPM.jl is experimental..")
        return true
    end
    return false
end

function KPM.maybe_to_device(x::Union{SparseMatrixCSC, CuSparseMatrixCSC}, expect_eltype=KPM.dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    if CUDA.has_cuda()
        if x isa CuSparseMatrixCSC
            return x
        else
            return CuSparseMatrixCSC{eltype(x)}(x)
        end
    end
    return x
end

function KPM.maybe_to_device(x::Union{Array, CuArray}, expect_eltype=KPM.dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    if CUDA.has_cuda()
        if x isa CuArray
            return x
        else
            return CuArray{eltype(x)}(x)
        end
    end
    return x
end

KPM.maybe_to_host(x::CuArray) = Array(x)
KPM.maybe_to_host(x::CuSparseMatrixCSC) = SparseMatrixCSC(x)

function KPM.maybe_on_device_rand(args...)
    if CUDA.has_cuda()
        return CUDA.rand(args...)
    end
    return rand(args...)
end

function KPM.maybe_on_device_zeros(args...)
    if CUDA.has_cuda()
        return CUDA.zeros(args...)
    end
    return zeros(args...)
end

# --- CUDA-specialized algorithm overloads (kept out of CPU default path) ---

function KPM.chebyshevT_xn(x_grid::CuArray{T, 1} where {T <: KPM.dt_num}, n_grid::CuArray{Int64, 1})
    Nx = length(x_grid)
    Nn = length(n_grid)
    T_xn = KPM.maybe_on_device_zeros(Nx, Nn)
    @cuda threads=(16, 16) blocks=(8, 8) chebyshevT_xn_cuda!(x_grid, n_grid, T_xn, Nx, Nn)
    return T_xn
end

function chebyshevT_xn_cuda!(x_grid, n_grid, T_xn, Nx, Nn)
    index0_m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index0_n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_m = blockDim().x * gridDim().x
    stride_n = blockDim().y * gridDim().y
    for nx = index0_m:stride_m:Nx
        for nn = index0_n:stride_n:Nn
            @inbounds T_xn[nx, nn] = KPM.chebyshevT_cu(n_grid[nn], x_grid[nx])
        end
    end
    return nothing
end

function KPM.chebyshev_lin_trans(x_grid::CuArray, n_grid::CuArray, mu_tilde::CuArray)
    Nx = length(x_grid)
    Nn = length(n_grid)
    y = complex(x_grid) * 0
    @cuda threads=32 blocks=16 chebyshev_lin_trans_cuda!(x_grid, n_grid, mu_tilde, Nx, Nn, y)
    return y
end

function chebyshev_lin_trans_cuda!(x_grid, n_grid, mu_tilde, Nx, Nn, y)
    index0 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for nx = index0:stride:Nx
        for nn = 1:Nn
            y[nx] += KPM.chebyshevT_cu(n_grid[nn], x_grid[nx]) * mu_tilde[nn]
        end
    end
    return nothing
end

function KPM.chebyshev_iter(H, ψall::CuArray{T, 2} where T, n::Int64)
    for i in 3:n
        KPM.chebyshev_iter_single(H, ψall, i-2, i-1, i)
    end
end

function KPM.chebyshev_iter_single(H, V_all::Union{Array, SubArray, CuArray}, i_pp_in::Int64, i_p_in::Int64)
    V_p_in = @view V_all[:, :, i_p_in]
    V_out = @view V_all[:, :, i_pp_in]
    KPM.chebyshev_iter_single(H, V_out, V_p_in)
    return nothing
end

function KPM.chebyshev_iter_single(H, V_pp_in::CuArray, V_p_in::CuArray)
    mul!(V_pp_in, H, V_p_in, 2.0+0im, -1.0+0im)
    return nothing
end

function KPM.chebyshev_iter_single(H, V_all::Union{Array, SubArray, CuArray}, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    (@view V_all[:, :, i_out]) .= (@view V_all[:, :, i_pp_in])
    KPM.chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

function KPM.chebyshev_iter_single(H, V_pp_in::CuArray, V_p_in::CuArray, V_out::CuArray)
    V_out .= V_pp_in
    KPM.chebyshev_iter_single(H, V_out, V_p_in)
end

function KPM.broadcast_dot_reduce_avg_2d_1d!(target::Union{Array, SubArray},
                                             Vls::Array{T, 1} where {T<:CuArray{Ts, 2} where Ts},
                                             Vr::CuArray{T, 2} where T,
                                             NR::Int64, NCcols::Int64;
                                             NC0::Int64=1, NCstep::Int64=1)
    target[NC0:NCstep:NCcols] .= dot.(Vls[NC0:NCstep:NCcols], [Vr])
    target ./= NR
    return nothing
end

function KPM.broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                                  Vl::CuArray,
                                  Vr::CuArray,
                                  NR::Int64,
                                  alpha::Number=1.0,
                                  beta::Number=0.0)
    for NRi in 1:NR
        target[NRi] = (dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha + beta)
    end
    return nothing
end

function KPM.broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                                  Vl::CuArray,
                                  Vr::CuArray,
                                  NR::Int64,
                                  alpha::Number,
                                  beta::CuArray)
    for NRi in 1:NR
        target[NRi] = (dot(view(Vl, :, NRi), view(Vr, :, NRi)) * alpha) + beta[NRi]
    end
    return nothing
end

KPM.Γnm_cu(n::Int64,m::Int64,ε) = ((ε - 1.0im * m * sqrt(1 - ε^2)) * exp(1.0im * m * CUDA.acos(ε)) * KPM.chebyshevT_cu(n, ε)+
                                    (ε + 1.0im * n * sqrt(1 - ε^2)) * exp(-1.0im * n * CUDA.acos(ε)) * KPM.chebyshevT_cu(m, ε))

function KPM.Γnmμnmαβ(μtilde::CuArray, ε::Float64, NC::Int64)
    temp_result = copy(μtilde)
    @cuda threads=(16, 16) blocks=(8, 8) gamma_nm_mu_nm_ab_kernel!(ε, NC, temp_result)
    return sum(temp_result)
end

function gamma_nm_mu_nm_ab_kernel!(ε, NC, temp_result)
    index0_m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index0_n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_m = blockDim().x * gridDim().x
    stride_n = blockDim().y * gridDim().y
    for m = index0_m:stride_m:NC
        for n = index0_n:stride_n:NC
            @inbounds temp_result[m, n] *= KPM.Γnm_cu(m-1, n-1, ε)
        end
    end
    return nothing
end

function KPM.broadcast_assign!(y_all::CuArray, y_all_views, x::CuArray, c_all::CuArray, idx_max::Int)
    block_count_x = cld(cld(length(x), 32), 512)
    block_count_y = idx_max
    CUDA.@sync @cuda threads=512 blocks=(block_count_x, block_count_y) cu_broadcast_assign!(y_all, x, c_all)
    return nothing
end

function cu_broadcast_assign!(y_all, x, c_all)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    c_idx = blockIdx().y
    x_l = length(x)
    for i = index:stride:x_l
        @inbounds y_all[i + (c_idx - 1) * x_l] += x[i] * c_all[c_idx]
    end
    return nothing
end

end # module
