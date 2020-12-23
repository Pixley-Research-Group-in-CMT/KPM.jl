# 1/2 factors
hn(n::Integer) = (n!=0) + 1 # h(0) = 1, otherwise 2

include("jackson_kernel.jl")
include("lorentz_kernel.jl")


"""
Create μtilde based on mu, by applying both kernel function and hn(n)
for each direction.

μtilde is ComplexF64 array.
"""
function mu2D_apply_kernel_and_h(mu, NC::Int64, kernel; dims::Array=[1, 2])
    μtilde = maybe_to_device(complex(copy(mu)))
    #println("size of mu: ", size(μtilde))
    if length(dims) == 0
        return μtilde
    end

    kernel_vec = maybe_to_device(kernel.(0:NC-1, NC))
    kernel_vec .*= hn.(0:NC-1)

    if 1 in dims
        mu2D_apply_kernel_and_h_dims1!(μtilde, NC, kernel_vec)    
    end
    if 2 in dims
        mu2D_apply_kernel_and_h_dims2!(μtilde, NC, kernel_vec)    
    end

    return μtilde
end


"""
Create μtilde based on mu, by applying both kernel function and hn(n)
for each direction.

μtilde is ComplexF64 array.

No mutate version, takes more memory
"""
function mu2D_apply_kernel_and_h_no_mutate(mu, NC::Int64, kernel; dims::Array=[1, 2])
    μtilde = maybe_to_device(complex(copy(mu)))
    #println("size of mu: ", size(μtilde))
    if length(dims) == 0
        return μtilde
    end

    kernel_vec = maybe_to_device(kernel.(0:NC-1, NC))
    kernel_vec = kernel_vec .* hn.(0:NC-1)

    if 1 in dims
        μtilde = mu2D_apply_kernel_and_h_dims1(μtilde, NC, kernel_vec)    
    end
    if 2 in dims
        μtilde = mu2D_apply_kernel_and_h_dims2(μtilde, NC, kernel_vec)    
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
