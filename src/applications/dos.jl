using DocStringExtensions
using ProgressBars

"""
$(METHODLIST)

Calculate DOS for a fermi energy grid spanning Erange with Ntilde total points.
If Erange is (0,0), automatically set it to be sightly smaller than full size.

When 
"""
function dos end



function dos(μ,
             a::Float64,
             Ntilde::Integer;Erange::Array{T,1}where {T<:Real}=[-1.0,1.0],
             NC::Integer=0,
             kernel = JacksonKernel)
    if length(size(μ)) != 1
        println("The input is not 1D array. Assuming the input is Hamiltonian normalized by a. Running `kpm_1d`...")
        H = μ
        NH = size(H)[1]
        if NC == 0
            println("NC is set to default (1024)")
            NC = 1024
        end
        NR = 21
        println("NR is set to default (21)") #TODO better default NR needed
        μ = kpm_1d(H, NC, NR)
    end

    μ = maybe_to_device(μ)

    if (Erange[1]==0 && Erange[2]==0) 
        Erange=[-a+0.01,a-0.01]
    end
    if (NC==0)
        NC = length(μ)
    else
        NC = min(NC,length(μ))
    end

    Evals_full = collect(((0:(Ntilde)).*(Erange[2]-Erange[1]))./Ntilde .+ Erange[1])
    rhoE_full = similar(Evals_full)
    rhoE_full .= 0


    idx = (abs.(Evals_full) .< abs(a))
    Evals = maybe_to_device(Evals_full[idx])

    n_grid = maybe_to_device(collect(0:(NC-1)))

    hgn = maybe_to_device(kernel.(0:(NC-1),NC) .* hn.(0:(NC-1)))

    rhoE = chebyshev_lin_trans(Evals / a,
                               n_grid,
                               μ .* hgn)

    denom = @. (a*pi*sqrt(1-(Evals/a)^2))
    rhoE ./= denom

    rhoE_full[idx] = maybe_to_host(rhoE)
    return Evals_full, rhoE_full
end
