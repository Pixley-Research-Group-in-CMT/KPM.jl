using Plots
using LaTeXStrings
using KPM

include("GrapheneModel.jl") # Include the GrapheneLattice function and related structures

function full_optical_condT0(mu1d,mu2d, NC, ω; δ=1e-5, λ=0.0, kernel=KPM.JacksonKernel,
    h = 0.001, Emin= -0.8, Emax = 0.0
    )
    # This function is used to calculate the full optical conductivity
    # by combining the 1D and 2D contributions.
    x_all = collect(Emin:h:Emax)
    y_1 = zeros(ComplexF64, length(x_all))
    y_2 = zeros(ComplexF64, length(x_all))
    mu1d_dev = KPM.maybe_to_device(mu1d[1:NC])
    mu2d_dev = KPM.maybe_to_device(mu2d[1:NC, 1:NC])

    for (i, x) in enumerate(x_all)
        y_1[i] += KPM.d_optical_cond1(mu1d_dev, NC, x; δ=δ, λ=λ, kernel=kernel)
        y_2[i] += KPM.d_optical_cond2(mu2d_dev, NC, ω, x; δ=δ, λ=λ, kernel=kernel)
    end
    return (sum(y_1) * h * (-1im / ω), sum(y_2) * h * (-1im / ω))
    #y_all = y_1 .+ y_2;
    #y_integral = sum(y_all) * h;
    
    #return y_integral*(-1im / ω) # -ie^2 / (ħ^2 * ω)
end

L = 200
Ham, Jx, Jy,Jxx,Jxy,Jyy = GrapheneLattice(L,L);

#heatmap(real.(Matrix(Jy)))
#Es, Vs = eigen(Matrix(Ham))

a = 3.5
H_norm = Ham ./ a
NC = 512 #512
NR = 10
NH = H_norm.n
mu_2d_yy = zeros(ComplexF64, NC, NC)
psi_in_l = exp.(2pi * 1im * rand(NH, NR));
KPM.normalize_by_col(psi_in_l, NR)
psi_in_r = psi_in_l
@time KPM.kpm_2d!(H_norm, Jy, Jy, NC, NR, NH, mu_2d_yy, psi_in_l, psi_in_r; verbose=1);

#mu_1d_xy = zeros(ComplexF64, NR, NC)
mu_1d_yy = KPM.kpm_1d_current(H_norm,Jyy, NC, NR; verbose=1)
#KPM.kpm_1d_current!(H_norm, Jxy, NC, NR, NH, mu_1d_xy, psi_in_l; verbose=1);

#=
ω = 1.0
Gamma = mu_2d_xy
optical_cond1(mu_1d_xy, NC, ω)
optical_cond2(Gamma, NC, ω)

d_optical_cond1(mu_1d_xy, NC, 0.0)
d_optical_cond2(Gamma, NC, ω, 0.0)

ϵ_grid = collect(LinRange(-0.999, 0, 1000))
res = zeros(ComplexF64, length(ϵ_grid))
for (i, ϵ) in enumerate(ϵ_grid)
    res[i] = d_optical_cond(Gamma, NC, ω, ϵ)
end
=#


t = 2.3
μ = 0.466
Ef = μ/t/a
λ = 38.8*10^(-3)/t/a
ωs = collect(LinRange(0.03, 0.982, 100))
res = zeros(ComplexF64, length(ωs))
res2 = zeros(ComplexF64, length(ωs))
for (i, ω) in enumerate(ωs)
    res[i], res2[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    #res[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    println(i)
end
σyyreal = real.(res2)./a
σyyimag = imag.(t*a*res.+res2)./a
ωsreno = ωs*t*a

plot(ylabel = L"\sigma^{yy}/\sigma_0",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:topright,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
         ylim=(-2,8)
        )
scatter!(ωs*t*a, σyyreal, label="real",markerstrokewidth=0.0)
scatter!(ωs*t*a, σyyimag, label="imag",markerstrokewidth=0.0)


#=
using JLD2
fn = "./HPCtemp/mu1D2DsL1000grapheneNR1.jld2"
L = load(fn, "L")
NC = load(fn,"NC")
D = load(fn,"D")
mu2d = load(fn,"mu2d")
mu1d = load(fn,"mu1d")

t = 2.3
μ = 0.466
Ef = μ/t/D
λ = 38.8*10^(-3)/t/D
ωs = collect(LinRange(0.01, 0.982, 100))
res = zeros(ComplexF64, length(ωs))
res2 = zeros(ComplexF64, length(ωs))
Threads.@threads for k in 1:length(ωs)
    ω = ωs[k]
    res[k], res2[k] = full_optical_condT0(mu1d,mu2d, NC, ω;λ=λ,Emax=Ef)
    #res[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    println(k)
end

σyyreal = real.(res2)./D
σyyimag = imag.(t*D*res.+res2)./D
ωsreno = ωs*t*D

plot(ylabel = L"\sigma^{yy}/\sigma_0",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:topright,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
        ylim=(-2,8)
        )
scatter!(ωsreno, [σyyreal σyyimag], label=["real" "imag"],markerstrokewidth=0.0)
#savefig("~/Desktop/OpticalCondGraphene.pdf")
=#
