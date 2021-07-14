using Random
using SparseArrays
using LinearAlgebra
using CUDA
using Logging

function whichcore()
    if CUDA.has_cuda()
        println("GPU support for KPM.jl is experimental..")
        return true
    end
    return false
end
multigpu=0
if whichcore()
    Ngpu = length(collect(devices()))
    if Ngpu > 1
        multigpu = Ngpu
        @info "multi gpu: $(multigpu)"
    else
        multigpu = 0
    end
end



function maybe_to_device(x::Union{SparseMatrixCSC, CUSPARSE.CuSparseMatrixCSC}, expect_eltype=dt_num; multigpu=multigpu)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    if CUDA.has_cuda()
        if multigpu > 0
            # Not already splitted, limited by function signature.
            if typeof(x) <: CUSPARSE.CuSparseMatrixCSC
                @info "Splitting $(typeof(x)). This is inefficient especially if it is already in GPU. Temporarily, we recommend to do it explicitly by `H = KPM._split_sparse_matrix(H; Ns=2)` for example."
                x = maybe_to_host(x)
            end

            x_split = _split_sparse_matrix(x; Ns = multigpu)
            return maybe_to_device(x_split)

        else
            if (typeof(x) <: CUDA.CUSPARSE.CuSparseMatrixCSC)
                return x
            else
                return CUDA.CUSPARSE.CuSparseMatrixCSC{eltype(x)}(x)
            end
        end
    else
        return x
    end
end

function maybe_to_device(x::OpsSplits)
    @assert CUDA.has_cuda() "Do not split operators when not using GPU for now."
    return _broadcast_operator_splits(x)
end

function maybe_to_device(x::Union{Array, CuArray}, expect_eltype=dt_num; multigpu=multigpu, split_hint=nothing)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    if CUDA.has_cuda()# && eltype(x).isbitstype
        if multigpu > 0
            if isnothing(split_hint)
                @info "workspace not created when building unified memory array"
            end
            x = _create_UM_arr(x; split_hint=split_hint)
            return x
        else
            if (typeof(x) <: CUDA.CuArray)
                return x
            else
                return CUDA.CuArray{eltype(x)}(x)
            end
        end
    else
        return x
    end
end


maybe_to_device(x::SubArray, expect_eltype=dt_num) = x # Pushing SubArray to GPU is bad for current CUDA stack.

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::CuArray) = Array(x)
maybe_to_host(x::CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(x)
maybe_to_host(x::Number) = x

function maybe_on_device_rand(args...; multigpu=multigpu, split_hint=nothing)
    if CUDA.has_cuda()
        if multigpu > 0
            return _create_UM_arr(rand(args...); split_hint=split_hint)
        else
            return CUDA.rand(args...)
        end
    else
        return rand(args...)
    end
end


function maybe_on_device_zeros(args...; multigpu=multigpu, split_hint=nothing)
    if CUDA.has_cuda()
        if multigpu > 0
            return _create_UM_arr(rand(args...); split_hint=split_hint)
        else
            return CUDA.zeros(args...)
        end
    else
        return zeros(args...)
    end
end



on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)

