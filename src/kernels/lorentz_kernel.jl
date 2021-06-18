# TODO doc
"""
LorentzKernels(λ::Float64)

Returns function LorentzKernel(n, N) that evaluates Lorentz kernel with parameter λ, at n-th expansion coefficient with N in total (NC)
"""
function LorentzKernels(λ::Float64) 
    LorentzKernel(n::Integer,N::Integer) = sinh(λ*(1-n/N))/sinh(λ)
    return LorentzKernel
end
