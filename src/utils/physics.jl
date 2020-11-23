#export sigmaMatrices, sigmaMatricesDot, fermiFunction, fermiFunctions
#export gammaMatrices
#export σx, σy, σz, σ0 
#export σ2dot, σ3dot


"""
    fermiFunction(E, E_f, beta)

calculate Fermi-Dirac function at energy E, Fermi energy μ and temperature β =1/T.
Input and output all Float64.
Infinite β only allowed when accessing fermi energy through fermiFunctions(). [For performance reason for now. TODO: allow β=Inf here withouth perf. reduction. ]

Allow sloppy use of type as long as convertion is available, if using keyword arguments. 
"""
function fermiFunction(E::Float64, E_f::Float64, beta::Float64)
    return 1/(exp((E-E_f)*beta)+1);
end
fermiFunction(; E, E_f, beta) = fermiFunction(Float64(E), Float64(E_f), Float64(beta))

"""
    fermiFunctions(E_f::Float64, beta::Float64)
returns a fermi function with given E_f and beta. 

Allow sloppy use of type as long as convertion is available, if using keyword arguments. 
"""
function fermiFunctions(E_f::Float64, beta::Float64)
    f(x) = fermiFunction(Float64(x), E_f, beta)
    g(x) = (x -> 
            (((x > E_f) + (x >= E_f)) / 2)
           )
    if isinf(beta)
        return g
    else
        return f
    end
end
fermiFunctions(; E_f, beta) = fermiFunctions(Float64(E_f), Float64(beta))
