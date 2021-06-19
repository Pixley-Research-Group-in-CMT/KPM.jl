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
whichcore()

function maybe_to_device(x::Union{SparseMatrixCSC, CUSPARSE.CuSparseMatrixCSC})
    if CUDA.has_cuda()
        if (typeof(x) <: CUDA.CUSPARSE.CuSparseMatrixCSC)
            if (eltype(x) == dt_cplx)
                return x
            else
                @info "Inefficient conversion of eltype ($(eltype(x)) to $(dt_cplx)) in $(typeof(x))"
                return maybe_to_device(maybe_to_host(x))
            end
        else
            return CUDA.CUSPARSE.CuSparseMatrixCSC{dt_cplx}(x)
        end
    else
        if eltype(x) == dt_cplx
            return x
        else
            return SparseMatrixCSC{dt_cplx, Int64}(x)
        end
    end
end

function maybe_to_device(x::Union{Array, CuArray})
    if CUDA.has_cuda()# && eltype(x).isbitstype
        if (typeof(x) <: CUDA.CuArray) && (eltype(x) == dt_cplx)
            return x
        else
            return CUDA.CuArray{dt_cplx}(x)
        end
    else
        if eltype(x) == dt_cplx
            return x
        else
            return Array{dt_cplx}(x)
        end
    end
end


maybe_to_device(x::SubArray) = x # Pushing SubArray to GPU is bad for current CUDA stack.

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::CuArray) = Array(x)
maybe_to_host(x::CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(x)
maybe_to_host(x::Number) = x

function maybe_on_device_rand(args...)
    if CUDA.has_cuda()
        return CUDA.rand(args...)
    else
        return rand(args...)
    end
end


function maybe_on_device_zeros(args...)
    if CUDA.has_cuda()
        return CUDA.zeros(args...)
    else
        return zeros(args...)
    end
end



on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)

