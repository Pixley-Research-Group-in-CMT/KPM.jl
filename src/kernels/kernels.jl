# 1/2 factors
hn(n::Integer) = (n!=0) + 1 # h(0) = 1, otherwise 2

include("jackson_kernel.jl")
include("lorentz_kernel.jl")


"""
Create μtilde based on mu, by applying both kernel function and hn(n)
for each direction.

μtilde is ComplexF64 array.
"""

function mu2D_apply_kernel_and_h(mu, NC::Int64, kernel::Function; dims::Array=[1, 2])
    kernels = [kernel, kernel]
    mu2D_apply_kernel_and_h(mu, NC, kernels; dims=dims)
end
function mu2D_apply_kernel_and_h(mu, NC::Int64, kernels::Array{T} where {T<:Function}; dims::Array=[1, 2])
    μtilde = maybe_to_device(complex(copy(mu)))
    #println("size of mu: ", size(μtilde))
    if length(dims) == 0
        return μtilde
    end

    kernel_vec1 = maybe_to_device(kernels[1].(0:NC-1, NC))
    kernel_vec1 .*= hn.(0:NC-1)
    kernel_vec2 = maybe_to_device(kernels[2].(0:NC-1, NC))
    kernel_vec2 .*= hn.(0:NC-1)

    if 1 in dims
        mu2D_apply_kernel_and_h_dims1!(μtilde, NC, kernel_vec1)
    end
    if 2 in dims
        mu2D_apply_kernel_and_h_dims2!(μtilde, NC, kernel_vec2)    
    end

    return μtilde
end


"""
Create μtilde based on mu, by applying both kernel function and hn(n)
for each direction.

μtilde is ComplexF64 array.

No mutate version, takes more memory
"""
function mu2D_apply_kernel_and_h_no_mutate(mu, NC::Int64, kernel::Function; dims::Array=[1, 2])
    kernels = [kernel, kernel]
    mu2D_apply_kernel_and_h_no_mutate(mu, NC, kernels; dims=dims)
end
function mu2D_apply_kernel_and_h_no_mutate(mu, NC::Int64, kernels::Array{Function, 1}; dims::Array=[1, 2])
    μtilde = maybe_to_device(complex(copy(mu)))
    #println("size of mu: ", size(μtilde))
    if length(dims) == 0
        return μtilde
    end

    kernel_vec1 = maybe_to_device(kernel[1].(0:NC-1, NC))
    kernel_vec1 = kernel_vec1 .* hn.(0:NC-1)
    kernel_vec2 = maybe_to_device(kernel[2].(0:NC-1, NC))
    kernel_vec2 = kernel_vec2 .* hn.(0:NC-1)

    if 1 in dims
        μtilde = mu2D_apply_kernel_and_h_dims1(μtilde, NC, kernel_vec1)    
    end
    if 2 in dims
        μtilde = mu2D_apply_kernel_and_h_dims2(μtilde, NC, kernel_vec2)    
    end

    return μtilde
end

mu3D_apply_kernel_and_h(args...; kwargs...) = muND_apply_kernel_and_h(args...; kwargs...)
mu3D_apply_kernel_and_h_no_mutate(args...; kwargs...) = muND_apply_kernel_and_h_no_mutate(args...; kwargs...)

function muND_apply_kernel_and_h(mu, NC::Int64, kernel; dims::Array=[1,2,3])
    μtilde = maybe_to_device(complex(copy(mu)))

    kernel_vec = maybe_to_device(kernel.(0:NC-1, NC))
    kernel_vec = kernel_vec .* hn.(0:NC-1)
    for d in dims
        target_shape = ones(Int64, d)
        target_shape[d] = NC
        μtilde .*= reshape(kernel_vec, target_shape...)
    end
    return μtilde
end

function muND_apply_kernel_and_h_no_mutate(mu, NC::Int64, kernel; dims::Array=[1,2,3])
    #TODO test no mutate
    μtilde = maybe_to_device(complex(copy(mu)))

    kernel_vec = maybe_to_device(kernel.(0:NC-1, NC))
    kernel_vec = kernel_vec .* hn.(0:NC-1)
    for d in dims
        target_shape = ones(Int64, d)
        target_shape[d] = NC
        μtilde = μtilde .* reshape(kernel_vec, target_shape...)
    end
    return μtilde
end


function mu2D_apply_kernel_and_h_dims1(mu, NC::Int64, kernel_vec)
    return mu .* kernel_vec
end
function mu2D_apply_kernel_and_h_dims1!(mu, NC::Int64, kernel_vec)
    mu .*= kernel_vec
    return nothing
end

function mu2D_apply_kernel_and_h_dims2(mu, NC::Int64, kernel_vec)
    return mu .* kernel_vec'
end
function mu2D_apply_kernel_and_h_dims2!(mu, NC::Int64, kernel_vec)
    mu .*= kernel_vec'
    return nothing
end
