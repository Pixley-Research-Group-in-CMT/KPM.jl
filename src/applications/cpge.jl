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


function Λnmp(nmp, ω; E_f=0.0, beta=100000)
    # Equation 43, ω1 = -ω2 = ω.
    # TODO
    # Use gn_A and Δn
    # If possible, try taking λ→0 analytically.
    # Any additional option added should be included as optional keyword arguments.
end


function gn_A(ϵ; n)
    # Equation 36
    # TODO
end

function Δn(ϵ; n)
    # Equation 35
    # TODO
end
