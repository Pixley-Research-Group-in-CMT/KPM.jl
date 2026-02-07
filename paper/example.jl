using KPM
using LinearAlgebra
using SparseArrays
using Plots
using LaTeXStrings

# Simple dense 1D tight-binding Hamiltonian (periodic)
function tb1dchain(N::Integer; t::Real=1.0)
    H = zeros(Float64, N, N)
    for i in 1:(N-1)
        H[i, i+1] = -t
        H[i+1, i] = -t
    end
    H[1, N] = -t
    H[N, 1] = -t
    return H
end

# Parameters
N = 1000               # system size
NC = 1024               # Chebyshev order
NR = 10               # number of random vectors for stochastic trace
nE = 1000             # output energy grid points

H = tb1dchain(N)
# Rescale H -> (-1, 1)
#Hsparse = sparse(H.*(1+0*1im)) # make the Hamiltonian sparse under complex number
b, H_norm = KPM.normalizeH(H)

# Compute Chebyshev moments (DOS)
mu = KPM.kpm_1d(H_norm, NC, NR)    # returns moments (array-like)

# Reconstruct DOS on a grid and map energies back to physical scale
E, rho1024 = KPM.dos(mu, b;kernel = KPM.JacksonKernel, N_tilde=nE)
E, rho64 = KPM.dos(mu[1:64], b;kernel = KPM.JacksonKernel, N_tilde=nE)
E, rho32 = KPM.dos(mu[1:32], b;kernel = KPM.JacksonKernel, N_tilde=nE)

# Analytical DOS 
rho_exact = zeros(length(E))
mask = abs.(E) .< 2
rho_exact[mask] = 1.0 ./ (π * sqrt.(4 .- E[mask].^2))

# plot the DOS
plot(xlabel=L"E", ylabel="DOS"*L"\;\rho(E)",
        legend = :top, 
        #xlim=[-0.1,0.8586],ylim=[-0.001,0.035],
        framestyle = :box,grid=false,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,
        )
plot!(E, [rho1024 rho64 rho32], lw=[4 3 2],label=[L"N_C=1024" L"N_C=64" L"N_C=32"])
plot!(E, rho_exact, c=:black, ls=:dash, label=L"\mathrm{Analytic}")
savefig("./paper/dosplot.pdf")