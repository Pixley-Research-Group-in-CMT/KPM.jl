using SparseArrays, Arpack, Random, LinearAlgebra

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
    normalizeH(H; eps=0.1, fixed_a=0.0, center=false)

Rescale a Hermitian `H` so its spectrum lies inside (-1, 1), as required by
the Chebyshev expansion.

By default returns `(a, H_norm)` with `H_norm = H / a` and
`a = 2 Emax / (2 - eps)`, where `Emax` is the largest eigenvalue magnitude
(Arpack estimate). This assumes a spectrum symmetric about zero; `eps` is the
safety margin keeping eigenvalues away from ±1 (the expansion diverges for
eigenvalues outside [-1, 1]).

With `center=true`, both spectral edges are found and the spectrum is also
shifted to its midpoint: returns `(a, b, H_norm)` with
`H_norm = (H - b I) / a`, `a = (Emax - Emin)/(2 - eps)`, `b = (Emax + Emin)/2`.
Use this for spectra that are not particle-hole symmetric, and pass `b` on to
`dos` so energies are mapped back correctly.

`fixed_a != 0` skips the edge estimation and uses the given scale (no shift);
it is incompatible with `center=true`.
"""
eps_float(H) = eps(real(float(eltype(H))))

function normalizeH(H; eps::Float64=0.1, fixed_a::Number=0.0, center::Bool=false)
    # exact check first; fall back to a roundoff-level tolerance so
    # floating-point construction noise does not reject a physical H
    if !ishermitian(H)
        dev = maximum(abs, H - H')
        scale = maximum(abs, H)
        if !(dev <= sqrt(eps_float(H)) * scale)
            throw(ArgumentError("normalizeH: H must be Hermitian (max |H - H'| = $dev)."))
        end
    end

    if fixed_a != 0
        center && throw(ArgumentError("normalizeH: center=true is incompatible with fixed_a — the center shift is derived from the spectral edges, which fixed_a skips."))
        a = fixed_a
        return a, H / a
    end

    if center
        es_max, _ = eigs(H; nev=1, which=:LR, tol=0.001, maxiter=300)
        es_min, _ = eigs(H; nev=1, which=:SR, tol=0.001, maxiter=300)
        Emax = real(es_max[1])
        Emin = real(es_min[1])
        a = (Emax - Emin) / (2 - eps)
        b = (Emax + Emin) / 2
        return a, b, (H - b * I) / a
    end

    es, _ = eigs(H; tol=0.001, maxiter=300)
    Emax = maximum(abs.(es))
    a = 2 * Emax / (2 - eps)
    return a, H / a
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
