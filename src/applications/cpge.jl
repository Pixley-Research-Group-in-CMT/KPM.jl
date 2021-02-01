using QuadGK # numerical integral

function cpge(Gamma, NC, ω; beta=1000000, E_f=0.0, kernel=JacksonKernel)
    # Equation 45, last term
    # Gamma is calculated using Hamiltonian that is
    # normalized to have energy bounded by [-1, 1]
    #
    # Unit of e^3/Ωħ^3

    cpge_αβγ = 0.0 + 0im

    Gamma_tilde = mu3D_apply_kernel_and_h(Gamma, NC, kernel)

    for n in 1:NC
        for m in 1:NC
            for p in 1:NC
                # note: this loop will be parallized
                Gamma_tilde[n, m, p] *= Λnmp([n, m, p], ω)
            end
        end
    end
   
    return sum(Gamma_tilde) * 1im / ω^2
end

"""
- beta : Inf is zero temperature. beta = 1/T.

- E_f : Fermi energy. Between -1 and 1 because Hamiltonian is normalized. 
"""
function Λnmp(nmp, ω; E_f=0.0, beta=Inf, δ=1e-5)
    # Equation 43, ω1 = -ω2 = ω. ctrl+k w*
    # The integral will cover [-1+δ, E_f]
    # λ should be considerably smaller than δ to ensure the value of gn match with the λ->0
    # but usually not be too small to avoid floating point error dominated by large number near 1
    # future plan: if possible, try taking λ→0 analytically.
    λ = δ / 100
    f = fermiFunctions(E_f, beta)

    n, m, p = nmp

    # apply cutoffs
    _gn_R(ϵ; n=n) = gn_R(ϵ; n=n, λ=λ, δ=δ) 
    _gn_A(ϵ; n=n) = gn_A(ϵ; n=n, λ=λ, δ=δ)

    f_rr(ϵ) = gn_R(ϵ; n=n) * gn_R(ϵ - ω; n=m) * Δn(ϵ; n=p)
    f_ar(ϵ) = gn_R(ϵ + ω; n=n) * Δn(ϵ; n=m) * gn_A(ϵ + ω; n=p)
    f_aa(ϵ) = Δn(ϵ; n=n) * gn_A(ϵ - ω; n=m) * gn_A(ϵ; n=p)
    Λnmp_integrand(ϵ) = (f_rr(ϵ) + f_ar(ϵ) + f_aa(ϵ)) * f(ϵ)

    I, E = quadgk(Λnmp_integrand, -1+δ, E_f) # numerical integration, E is error
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

function Δn(ϵ; n)
    # Equation 35 ctrl+k D*
    numerator = 2 * cos(n * acos(ϵ))
    denominator = pi * sqrt(1 - ϵ^2)
    return numerator / denominator
end
