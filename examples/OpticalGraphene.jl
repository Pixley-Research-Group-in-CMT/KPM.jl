using LinearAlgebra
using SparseArrays
using Plots
using LaTeXStrings
using KPM


struct HoneycombSite
    x::Int
    y::Int
    sublattice::Int  # 1 for A, 2 for B
    pos::Vector{Float64}  # real-space coordinates [rx, ry]
end

function site_position(x, y, subl)
    # Lattice vectors
    rx = [sqrt(3), 0]
    ry = [sqrt(3)/2, 3/2]
    # Sublattice offset
    offset = subl == 1 ? [0.0, 0.0] : [sqrt(3)/2, 1/2] # -δ_3
    # Real-space position
    pos = offset .+ (x-1)*rx .+ (y-1)*ry
    return pos
end

function all_sites(Nx, Ny)
    sites = HoneycombSite[]
    for x in 1:Nx
        for y in 1:Ny
            push!(sites, HoneycombSite(x, y, 1, site_position(x, y, 1)))
            push!(sites, HoneycombSite(x, y, 2, site_position(x, y, 2)))
        end
    end
    return sites
end

function GrapheneLattice(Nx::Int64, Ny::Int64; 
    t::Number = 1.0, # NN hopping
    Δ::Number = 0.0, # On-site energy difference
    bc_factor_x::Number = 1.0+0im, # twisted boundary condition
	bc_factor_y::Number = 1.0+0im,
    )

    N = 2 * Nx * Ny  # Total number of sites
    site_list = all_sites(Nx, Ny)

    function cell_pos(x,y)
		modx = mod(x-1,Nx) + 1 # Periodic Boundary condition
		mody = mod(y-1,Ny) + 1
		return mody + Ny * (modx - 1)
	end

    Ind1, Ind2, Vals = Int64[], Int64[], ComplexF64[]
    function addToList!(pos1,pos2,value,Ind1,Ind2,Vals)
            push!(Ind1, pos1)
            push!(Ind2, pos2)
            push!(Vals, value)
            return
    end
    for x in 1:Nx, y in 1:Ny
        curpos = cell_pos(x,y)
        pbx = cell_pos(x+1,y)
        pby = cell_pos(x,y+1)
        # Add diagonal terms
        addToList!(2*curpos-1, 2*curpos-1, Δ/2, Ind1, Ind2, Vals)
        addToList!(2*curpos-1, 2*curpos, -t, Ind1, Ind2, Vals)
        addToList!(2*curpos, 2*curpos-1, -t, Ind1, Ind2, Vals)
        addToList!(2*curpos, 2*curpos, -Δ/2, Ind1, Ind2, Vals)
        # Add hopping terms
        addToList!(2*curpos, 2*pbx-1, (-t)*bc_factor_x^(x==Nx), Ind1, Ind2, Vals)
        addToList!(2*pbx-1,2*curpos, ((-t)*bc_factor_x^(x==Nx))', Ind1, Ind2, Vals)
        addToList!(2*curpos, 2*pby-1, (-t)*bc_factor_y^(y==Ny), Ind1, Ind2, Vals)
        addToList!(2*pby-1, 2*curpos, ((-t)*bc_factor_y^(y==Ny))', Ind1, Ind2, Vals)
    end
    Ham = sparse(Ind1, Ind2, Vals, N, N)

    #Computing the velocity operator for the graphene lattice. 
    #tilde{h}_{ij} = (i hbar) h_{ij}
    #h_{ij}^mu = (i hbar)^(-1) H_{ij} d_{ij}
    #mu = x, y
    rx = [sqrt(3), 0]
    ry = [sqrt(3)/2, 3/2]
    Valxs, Valys = ComplexF64[], ComplexF64[]
    Valxxs, Valyys, Valxys = ComplexF64[], ComplexF64[], ComplexF64[]
    for k in 1:length(Ind1)
        i = Ind1[k]
        j = Ind2[k]
        dist = site_list[i].pos - site_list[j].pos
        for Py in -1:1, Px in -1:1
            dpos = dist .+ Py * Ny .* ry .+ Px * Nx .* rx
            if norm(dpos) < 1.1
                push!(Valxs,Vals[k]*dpos[1])
                push!(Valys,Vals[k]*dpos[2])
                push!(Valxxs,Vals[k]*dpos[1]*dpos[1])
                push!(Valxys,Vals[k]*dpos[1]*dpos[2])
                push!(Valyys,Vals[k]*dpos[2]*dpos[2])
                break
            end
        end
    end
    Jx = sparse(Ind1, Ind2, Valxs, N, N)
    Jy = sparse(Ind1, Ind2, Valys, N, N)
    Jxx = sparse(Ind1, Ind2, Valxxs, N, N)
    Jxy = sparse(Ind1, Ind2, Valxys, N, N)
    Jyy = sparse(Ind1, Ind2, Valyys, N, N)
    return Ham, Jx, Jy, Jxx, Jxy, Jyy
end

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


Ham, Jx, Jy,Jxx,Jxy,Jyy = GrapheneLattice(1024,1024)

#heatmap(real.(Matrix(Jy)))
#Es, Vs = eigen(Matrix(Ham))

a = 3.5
H_norm = Ham ./ a
NC = 512
NR = 2
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

plot(ylabel = L"\sigma^{yy}/\sigma_0",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:left,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
        ylim=(-10,20)
        )
scatter!(ωs*t*a, real.(res)./a, label="real",markerstrokewidth=0.0)
scatter!(ωs*t*a, imag.(res)./a, label="imag",markerstrokewidth=0.0)



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

