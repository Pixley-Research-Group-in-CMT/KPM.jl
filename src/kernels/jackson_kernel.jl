#TODO docs
"""
JacksonKernel(n::Integer, N::Integer)

Jacksonkernel evaluated at n-th expansion coefficient with N in total (NC)
"""
JacksonKernel(n::Integer, N::Integer) = @. 1/(N+1)*((N+1-n)*cos(pi*n/(N+1))
                                                    + sin(pi*n/(N+1))*cot(pi/(N+1)))
