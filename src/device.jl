using Random
using SparseArrays
using LinearAlgebra
using CUDA

function whichcore()
    if CUDA.has_cuda()
        println("GPU support for KPM.jl is experimental..")
        return true
    end
    return false
end
whichcore()

@generated function maybe_to_device(x::SparseMatrixCSC{T, Int64} where T<:Number)
    if CUDA.has_cuda()
        return :(CUSPARSE.CuSparseMatrixCSC{dt_cplx}(x))
    else
        return :(SparseMatrixCSC{dt_cplx, Int64}(x))
    end
end
maybe_to_device(x::CUSPARSE.CuSparseMatrixCSC) = x

@generated function maybe_to_device(x::Array{T} where T<:Number)
    if CUDA.has_cuda()# && eltype(x).isbitstype
        return :(CuArray{dt_cplx}(x))
    else
        return :(Array{dt_cplx}(x))
    end
end
maybe_to_device(x::CuArray{T} where T<:Number) = CuArray{dt_cplx}(x)

maybe_to_device(x::SubArray) = x # Pushing SubArray to GPU should never happen

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::CuArray) = Array(x)
maybe_to_host(x::CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(x)
maybe_to_host(x::Number) = x

@generated function maybe_on_device_rand(args...)
    if CUDA.has_cuda()
        return :(maybe_to_device(rand(args...)))
    else
        return :(rand(args...))
    end
end


@generated function maybe_on_device_zeros(args...)
    if CUDA.has_cuda()
        return :(CUDA.zeros(args...))
    else
        return :(zeros(args...))
    end
end



on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)

