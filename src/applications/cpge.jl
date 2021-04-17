using QuadGK # numerical integral
using FastGaussQuadrature # numerical integral - non adaptive

function cpge(Gamma, NC, ω; beta=Inf, E_f=0.0, kernel=JacksonKernel, δ=1e-5, Ω=ω/20)
    # Equation 45, last term
    # Gamma is calculated using Hamiltonian that is
    # normalized to have energy bounded by [-1, 1]
    #
    # Unit of e^3/Ωħ^3

    @assert (abs.(Ω/ω) < 0.1) "Ω should be much smaller than ω."
    cpge_αβγ = 0.0 + 0im

    Gamma_tilde = mu3D_apply_kernel_and_h(Gamma, NC, kernel)

    # applying specified quad
    nodes, weights = gausschebyshev(NC * 8)
    quad(f) = (
               dot(weights, f.(nodes)),
               nothing # THIS SHOULD BE AN ESTIMATION OF ERROR
              )


    ω₁ = ω
    ω₂ = Ω - ω
    Λnmp_all = map(nmp -> Λnmp(nmp, ω₁, ω₂; δ=δ, E_f=E_f, beta=beta, quad=quad), Iterators.product(0:(NC-1), 0:(NC-1), 0:(NC-1)))

    Gamma_tilde .*= Λnmp_all
    #for n in 1:NC
    #    for m in 1:NC
    #        for p in 1:NC
    #            # note: this loop will be parallized
    #            Gamma_tilde[n, m, p] *= Λnmp([n-1, m-1, p-1], ω; δ=δ, E_f=E_f, beta=beta)
    #        end
    #    end
    #end
   
    return sum(Gamma_tilde) * 1im / (ω₁ * ω₂) * Ω
end

"""
- beta : Inf is zero temperature. beta = 1/T.

- E_f : Fermi energy. Between -1 and 1 because Hamiltonian is normalized. 
"""
function Λnmp(nmp, ω₁, ω₂; E_f=0.0, beta=Inf, δ=1e-5, λ=1e-7, quad=(f->quadgk(f, -1+δ, 1-δ)))
    # Equation 43, Ω = ω1 + ω2, expecting Ω -> 0
    # The integral will cover [-1+δ, 1-δ], where Fermi energy is taken care of by fermi function.
    # λ should be much smaller than δ to ensure the value of gn match with the λ->0
    # but usually not be too small to avoid floating point error dominated by large number near 1
    # future plan: if possible, try taking λ→0 analytically.
    #λ = δ / 100
    f = fermiFunctions(E_f, beta)

    n, m, p = nmp

    # apply cutoffs
    _gn_R(ϵ; n) = gn_R(ϵ; n=n, λ=λ, δ=δ) 
    _gn_A(ϵ; n) = gn_A(ϵ; n=n, λ=λ, δ=δ)
    _Δn(ϵ; n) = Δn(ϵ; n=n, δ=δ)

    f_rr(ϵ) = _gn_R(ϵ + ω₁ + ω₂; n=n) * _gn_R(ϵ + ω₂; n=m) * _Δn(ϵ; n=p)
    f_ar(ϵ) = _gn_R(ϵ + ω₁; n=n) * _Δn(ϵ; n=m) * _gn_A(ϵ - ω₂; n=p)
    f_aa(ϵ) = _Δn(ϵ; n=n) * _gn_A(ϵ - ω₁; n=m) * _gn_A(ϵ - ω₁ - ω₂; n=p)
    Λnmp_integrand(ϵ) = (f_rr(ϵ) + f_ar(ϵ) + f_aa(ϵ)) * f(ϵ)

    #I, E = quadgk(Λnmp_integrand, -1, E_f) # numerical integration, E is error
    I, E = quad(Λnmp_integrand)
    return I
end


function gn_A(ϵ; n, λ=1e-10, δ=1e-5)
    # Equation 36 ctrl+k j3
    # λ is soft cutoff, δ is hard cutoff
    if abs(1 - abs(ϵ)) < δ
        return 0.0
    end
    numerator = 2 * exp(1im * n * acos(ϵ - λ * im)) * 1im
    denominator = sqrt(1 - (ϵ - λ * im)^2)
    return numerator / denominator
end

function gn_R(ϵ; n, λ=1e-10, δ=1e-5)
    # Equation 36 ctrl+k j3
    # λ is soft cutoff, δ is hard cutoff
    if abs(1 - abs(ϵ)) < δ
        return 0.0
    end
    numerator = - 2 * exp(- 1im * n * acos(ϵ + λ * im)) * 1im
    denominator = sqrt(1 - (ϵ + λ * im)^2)
    return numerator / denominator
end

function Δn(ϵ; n, δ=1e-5)
    # Equation 35 ctrl+k D*
    if abs(1 - abs(ϵ)) < δ
        return 0.0
    end
    numerator = 2 * cos(n * acos(ϵ))
    denominator = pi * sqrt(1 - ϵ^2)
    return numerator / denominator
end
