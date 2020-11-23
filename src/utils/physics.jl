export sigmaMatrices, sigmaMatricesDot, fermiFunction, fermiFunctions
export gammaMatrices
export σx, σy, σz, σ0 
export σ2dot, σ3dot

"""
wi dot σi
i correspond to x,y,z and last one is identity
examples:
    smd3 = sigmaMatricesDot(3)
    smd3([1 0 0 0]) 
        ->
        2×2 Array{ComplexF64,2}:
         0.0+0.0im  1.0+0.0im
         1.0+0.0im  0.0+0.0im
    smd3([0 1 0 0])
        ->
        2×2 Array{ComplexF64,2}:
         0.0+0.0im  0.0-1.0im
         0.0+1.0im  0.0+0.0im
    smd3([1 1 1 1])
        ->
        2×2 Array{ComplexF64,2}:
         2.0+0.0im  1.0-1.0im
         1.0+1.0im  0.0+0.0im
"""
function sigmaMatricesDot(n)
    σi = sigmaMatrices(n)
    function f(w)
        return reduce(+, map(*, w, σi))
    end
    return f
end

function sigmaMatrices()
    return sigmaMatrices(3)
end

function sigmaMatrices(n)
    if n == 2
        σx = [0.0+0im 1.0;
              1.0 0.0]
        σy = [0.0im -1.0im;
              1.0im 0.0im]
        iii = [1.0+0im 0.0;
               0.0 1.0]
        return σx, σy, iii
    end
    if n == 3
        σx = [0.0+0im 1.0;
              1.0 0.0]
        σy = [0.0im -1.0im;
              1.0im 0.0im]
        σz = [1.0+0im 0.0;
              0.0 -1.0]
        iii = [1.0+0im 0.0;
               0.0 1.0]
        return σx, σy, σz, iii
    end
    print("not implemented: ")
    println(n)
    return 0
end


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
σx, σy, σz, σ0 = sigmaMatrices(3)

function gammaMatrices(d; rep=:Dirac)
    @assert (rep==:Dirac) "Gamma matrices: only implemented dirac representation"
    @assert (d==3) "Gamma matrices: only d=3 implemented"
    if d == 3
        if rep==:Dirac
            σx, σy, σz, ii = sigmaMatrices(d)
            zz =  zeros(ComplexF64, 2,2) #00
            αx = [zz σx;
                  σx zz]
            αy = [zz σy;
                  σy zz]
            αz = [zz σz;
                  σz zz]
            β  = [ii zz;
                  zz -ii]
            iii= [ii zz;
                  zz ii]
            return αx, αy, αz, β, iii
        end
    end
end


σ2dot = sigmaMatricesDot(2)
σ3dot = sigmaMatricesDot(3)

