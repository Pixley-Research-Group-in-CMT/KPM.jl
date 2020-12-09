# 1/2 factors
hn(n::Integer) = (n!=0) + 1 # h(0) = 1, otherwise 2

include("jackson_kernel.jl")
include("lorentz_kernel.jl")
