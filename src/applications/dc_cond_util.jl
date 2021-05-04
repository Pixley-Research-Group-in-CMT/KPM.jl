"""
utility functions for conductivity 
"""

"""
Calculate Γnm. Details see Garcia et.al, PRL 114, 116602 (2015)
"""
Γnm(n::Int64,m::Int64,ε) = ((ε - 1.0im * m * sqrt(1 - ε^2)) * exp(1.0im * m * acos(ε)) * chebyshevT(n, ε) +
                                     (ε + 1.0im * n * sqrt(1 - ε^2)) * exp(-1.0im * n * acos(ε)) * chebyshevT(m, ε))


"""
Calculate Γnmμnmαβ.
The input μtilde should be the moment that has already applied kernel and hn
"""
function Γnmμnmαβ(μtilde::Array, ε, NC) 
    #for m in 1:NC
    #    for n in 1:NC
    #        result += Γnm(m-1, n-1, ε) * μtilde[m, n]
    #    end
    #end
    Γnm_matrix = Γnm.(0:NC-1, (0:NC-1)', ε)
    # Note: Γnm_matrix is real and same dimension as μtilde
    @assert size(Γnm_matrix) == size(μtilde)
    result = sum(Γnm_matrix .* μtilde)
    return result
end


function gamma_nm_mu_nm_ab_kernel!(ε, NC, temp_result)
    index0_m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index0_n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_m = blockDim().x * gridDim().x
    stride_n = blockDim().y * gridDim().y
    for m = index0_m:stride_m:NC
        for n = index0_n:stride_n:NC
            @inbounds temp_result[m, n] *= Γnm_cu(m-1, n-1, ε) # * μtilde[m, n]
        end
    end
    return nothing
end
