module Utils

export wrapAdd, normalizeH, isNotBoundary, timestamp
using SparseArrays, Arpack, Random
"""
wrapAdd find the sum of x and y, with L+1=1
"""
function wrapAdd(x::Int64,y::Int64,L::Int64)
        return mod(x+y-1,L)+1
end


"""
give 0 for OBC=1 direction if i,i_ is on boundary. Otherwise 1
"""
function isNotBoundary(ijk,ijk_,sizes,OBC)
	f(x,L,OBC) = (abs(div(x,L-1))) & OBC # is on boundary and has OBC requirement
	return 1-reduce(|, map(f,ijk-ijk_,sizes,OBC))
end


"""
Normalize H. If requested, allow renormalizing it to fixed value.
"""
function normalizeH(H::SparseMatrixCSC{ComplexF64,Int64}; eps::Float64=0.1, setA::Float64=0.0)
    
    println("hermitian check: ")
    @assert abs(sum(H-H')) < 1e-16*sqrt(H.n)
    println("pass.")

	if setA==0
    es, vs = eigs(H;tol=0.001,maxiter=300)
#	println(es)
    Emax = maximum(abs.(es))
    Emin = -Emax
    a = (Emax - Emin)/(2 - eps)
	else
    a = setA
	end

    H = H/a
    return a, H

end

function timestamp(text; t = [time(), time()], r = 0, init = false, rank=0)
    curr_time = time()
    if init
        t[1] = curr_time
        t[2] = curr_time
    end 
    println("TIME-",text, ",Δt=", curr_time - t[2], "; total elapsed=", curr_time - t[1],  "at round ", r,"@rank", rank) 
    t[2] = curr_time

end
end
