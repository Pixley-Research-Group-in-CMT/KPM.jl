module KPMCUDAExt

using KPM
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using LinearAlgebra
import LinearAlgebra: mul!

# GPU execution device. All methods below are defined on CUDA-specific types
# (CUDADevice, CuArray, CuSparseMatrixCSR) so nothing in the base package is
# overwritten and the extension precompiles normally.
struct CUDADevice <: KPM.AbstractDevice end

function __init__()
    if CUDA.functional()
        KPM.ACTIVE_DEVICE[] = CUDADevice()
        @info "KPM.jl: CUDA GPU support active (experimental)."
    end
end

# --- device transfer & allocation -----------------------------------------

# Sparse matrices go to CSR (CUSPARSE's preferred layout for SpMM) and are cast
# to `expect_eltype` so mul! against the complex Chebyshev vectors is supported.
KPM.to_device(::CUDADevice, x::SparseMatrixCSC, expect_eltype) = CuSparseMatrixCSR{expect_eltype}(x)
KPM.to_device(::CUDADevice, x::Array, expect_eltype) = CuArray{expect_eltype}(x)
KPM.to_device(::CUDADevice, x::AbstractCuSparseMatrix, expect_eltype) = x
KPM.to_device(::CUDADevice, x::CuArray, expect_eltype) = x

KPM.maybe_to_host(x::CuArray) = Array(x)
KPM.maybe_to_host(x::CuSparseMatrixCSR) = SparseMatrixCSC(x)
KPM.maybe_to_host(x::CuSparseMatrixCSC) = SparseMatrixCSC(x)

KPM.device_zeros(::CUDADevice, args...) = CUDA.zeros(args...)
KPM.device_rand(::CUDADevice, args...) = CUDA.rand(args...)

# --- Chebyshev three-term recurrence ---------------------------------------

# α_{n+1} = 2 H α_n - α_{n-1}, fused into one 5-arg mul! that overwrites the
# α_{n-1} buffer. Contiguous views of CuArrays are CuArrays, so these methods
# catch the whole GPU path of kpm_1d!/kpm_2d!.
function KPM.chebyshev_iter_single(H, V_pp_in::CuArray, V_p_in::CuArray)
    T = eltype(V_pp_in)
    mul!(V_pp_in, H, V_p_in, T(2), T(-1))
    return nothing
end

function KPM.chebyshev_iter_single(H, V_pp_in::CuArray, V_p_in::CuArray, V_out::CuArray)
    V_out .= V_pp_in
    KPM.chebyshev_iter_single(H, V_out, V_p_in)
end

# Index-based variants on a 3D workspace V_all[:, :, i].
function KPM.chebyshev_iter_single(H, V_all::CuArray, i_pp_in::Int64, i_p_in::Int64)
    KPM.chebyshev_iter_single(H, view(V_all, :, :, i_pp_in), view(V_all, :, :, i_p_in))
    return nothing
end

function KPM.chebyshev_iter_single(H, V_all::CuArray, i_pp_in::Int64, i_p_in::Int64, i_out::Int64)
    view(V_all, :, :, i_out) .= view(V_all, :, :, i_pp_in)
    KPM.chebyshev_iter_single(H, V_all, i_out, i_p_in)
end

function KPM.chebyshev_iter(H, ψviews::Array{<:CuArray}, n::Int64)
    for i in 3:n
        KPM.chebyshev_iter_single(H, ψviews[i-2], ψviews[i-1], ψviews[i])
    end
end

function KPM.chebyshev_iter_wrap(H, ψviews::Array{<:CuArray}, n::Int64)
    KPM.chebyshev_iter_single(H, ψviews[n-1], ψviews[n], ψviews[1])
    KPM.chebyshev_iter_single(H, ψviews[n], ψviews[1], ψviews[2])
end

# --- moment reductions ------------------------------------------------------

function KPM.broadcast_dot_1d_1d!(target::Union{Array, SubArray},
                                  Vl_arr::Array{<:CuArray},
                                  Vr_arr::Array{<:CuArray};
                                  alpha::Number=1.0,
                                  beta=0.0)
    target .= dot.(Vl_arr, Vr_arr)
    target .*= alpha
    target .+= KPM.maybe_to_host(beta)
    return nothing
end

function KPM.broadcast_dot_reduce_avg_2d_1d!(target::Union{Array, SubArray},
                                             Vls::Array{T, 1} where {T<:CuArray{Ts, 2} where Ts},
                                             Vr::CuArray{T, 2} where T,
                                             NR::Int64, NCcols::Int64;
                                             NC0::Int64=1, NCstep::Int64=1)
    target[NC0:NCstep:NCcols] .= dot.(Vls[NC0:NCstep:NCcols], (Vr,))
    target ./= NR
    return nothing
end

# --- Chebyshev polynomial evaluation (DoS reconstruction) -------------------

function KPM.chebyshevT_xn(x_grid::CuArray{T, 1} where {T <: KPM.dt_num}, n_grid::CuArray{Int64, 1})
    Nx = length(x_grid)
    Nn = length(n_grid)
    T_xn = CUDA.zeros(KPM.dt_real, Nx, Nn)
    (Nx == 0 || Nn == 0) && return T_xn
    blocks = (min(cld(Nx, 16), 1024), min(cld(Nn, 16), 1024))
    @cuda threads=(16, 16) blocks=blocks chebyshevT_xn_cuda!(x_grid, n_grid, T_xn, Nx, Nn)
    return T_xn
end

function chebyshevT_xn_cuda!(x_grid, n_grid, T_xn, Nx, Nn)
    index0_m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index0_n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_m = blockDim().x * gridDim().x
    stride_n = blockDim().y * gridDim().y
    for nx = index0_m:stride_m:Nx
        for nn = index0_n:stride_n:Nn
            @inbounds T_xn[nx, nn] = cos(n_grid[nn] * acos(x_grid[nx]))
        end
    end
    return nothing
end

function KPM.chebyshev_lin_trans(x_grid::CuArray, n_grid::CuArray, mu_tilde::CuArray)
    Nx = length(x_grid)
    Nn = length(n_grid)
    y = CUDA.zeros(complex(eltype(x_grid)), Nx)
    Nx == 0 && return y  # empty in-band grid: dos() permits this
    threads = 256
    blocks = min(cld(Nx, threads), 1024)
    @cuda threads=threads blocks=blocks chebyshev_lin_trans_cuda!(x_grid, n_grid, mu_tilde, Nx, Nn, y)
    return y
end

function chebyshev_lin_trans_cuda!(x_grid, n_grid, mu_tilde, Nx, Nn, y)
    index0 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for nx = index0:stride:Nx
        acc = zero(eltype(y))
        for nn = 1:Nn
            acc += cos(n_grid[nn] * acos(x_grid[nx])) * mu_tilde[nn]
        end
        y[nx] = acc
    end
    return nothing
end

# --- conductivity helpers ---------------------------------------------------

# Γnm of Garcia et al., PRL 114, 116602 — device-side scalar version.
function _gamma_nm(n, m, ε)
    Tn = cos(n * acos(ε))
    Tm = cos(m * acos(ε))
    return (ε - im * m * sqrt(1 - ε^2)) * exp(im * m * acos(ε)) * Tn +
           (ε + im * n * sqrt(1 - ε^2)) * exp(-im * n * acos(ε)) * Tm
end

# NOTE: ε is restricted to Float64, so d_dc_cond with dE_order >= 1 (which
# feeds ForwardDiff duals through this function) is not supported on GPU;
# compute derivative spectra on the CPU.
function KPM.Γnmμnmαβ(μtilde::CuArray, ε::Float64, NC::Int64)
    temp_result = copy(μtilde)
    blocks = (min(cld(NC, 16), 1024), min(cld(NC, 16), 1024))
    @cuda threads=(16, 16) blocks=blocks gamma_nm_mu_nm_ab_kernel!(ε, NC, temp_result)
    return sum(temp_result)
end

function gamma_nm_mu_nm_ab_kernel!(ε, NC, temp_result)
    index0_m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index0_n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_m = blockDim().x * gridDim().x
    stride_n = blockDim().y * gridDim().y
    for m = index0_m:stride_m:NC
        for n = index0_n:stride_n:NC
            @inbounds temp_result[m, n] *= _gamma_nm(m-1, n-1, ε)
        end
    end
    return nothing
end

# --- dc_long accumulation ---------------------------------------------------

function KPM.broadcast_assign!(y_all::CuArray, y_all_views, x::CuArray, c_all::CuArray, idx_max::Int)
    threads = 512
    block_count_x = min(cld(length(x), threads), 1024)
    CUDA.@sync @cuda threads=threads blocks=(block_count_x, idx_max) cu_broadcast_assign!(y_all, x, c_all)
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
