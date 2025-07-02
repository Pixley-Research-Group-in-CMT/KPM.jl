using KPM
using Plots
using LaTeXStrings

include("GrapheneModel.jl") 

function MET0(mu2d, NC, ω; δ=1e-5, λ=0.0, kernel=KPM.JacksonKernel,
    h = 0.001, Emin= -0.8, Emax = 0.0
    )
    x_all = collect(Emin:h:Emax)
    y_2 = zeros(ComplexF64, length(x_all))
    mu2d_dev = KPM.maybe_to_device(mu2d[1:NC, 1:NC])

    for (i, x) in enumerate(x_all)
        y_2[i] += KPM.d_optical_cond2(mu2d_dev, NC, ω, x; δ=δ, λ=λ, kernel=kernel)
    end
    #return x_all, y_2
    return sum(y_2) * h * (1im)
end

L = 200
Ham, Jx, Jy,Jxx,Jxy,Jyy = GrapheneLattice(L,L);
rx, ry = GraphenePositionOps(L,L);
#rz = Matrix{Float64}(I, 2*L^2, 2*L^2) 
rz = sparse(1.0I, 2*L^2, 2*L^2) # Sparse identity matrix

a = 3.5
H_norm = Ham ./ a
NC = 512 #512
NR = 10
NH = H_norm.n
Op1 = ry
Op2 = -Jy
mu_2d_xx = KPM.kpm_2d(H_norm,Op1, Op2, NC, NR, NH; verbose=1);
Op1 = rx
Op2 = Jx
mu_2d_yy = KPM.kpm_2d(H_norm,Op1, Op2, NC, NR, NH; verbose=1);
Op1 = rz
Op2 = (rx*Jy-ry*Jx)./2
mu_2d_zz_1 = KPM.kpm_2d(H_norm,Op1, Op2, NC, NR, NH; verbose=1);
Op1 = ry./2
Op2 = Jx
mu_2d_zz_2 = KPM.kpm_2d(H_norm,Op1, Op2, NC, NR, NH; verbose=1);
Op1 = rx./2
Op2 = Jy
mu_2d_zz_3 = KPM.kpm_2d(H_norm,Op1, Op2, NC, NR, NH; verbose=1);
mu_2d_zz = mu_2d_zz_1 + mu_2d_zz_2 - mu_2d_zz_3;
#mu_2d = mu_2d_xx + mu_2d_yy + mu_2d_zz;
#=
es, test = MET0(mu_2d_xx, NC, 0.2;λ=0.0,Emax=0,Emin=-0.99)
scatter(es,real.(test))
scatter(es,imag.(test))
=#

t = 1
μ = 0
Ef = μ/t/a
λ = 0.0/t/a
ωs = collect(LinRange(0.01, 1, 100))
res = zeros(ComplexF64, length(ωs),3)
for (i, ω) in enumerate(ωs)
    res[i,1]= MET0(mu_2d_xx, NC, ω;λ=λ,Emax=Ef,Emin=-0.99)
    res[i,2]= MET0(mu_2d_yy, NC, ω;λ=λ,Emax=Ef,Emin=-0.99)
    res[i,3]= MET0(mu_2d_zz, NC, ω;λ=λ,Emax=Ef,Emin=-0.99)
    println(i)
end
Gxxreal = real.(sum(res, dims=2))./a
Gxximag = imag.(sum(res, dims=2))./a
ωsreno = ωs*t*a

plot(ylabel = L"\sum_i G_{ii}^{(z)}",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:topleft,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
         #ylim=(-2,8)
        )
scatter!(ωsreno, Gxxreal, label="real",markerstrokewidth=0.0)
scatter!(ωsreno, Gxximag, label="imag",markerstrokewidth=0.0)
#savefig("~/Desktop/fig2.pdf")

plot(ylabel = L"G_{xx}^{(z)}",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:topleft,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
         #ylim=(-2,8)
        )
scatter!(ωsreno, real.(res[:,1])./a, label=L"\mathrm{Re}\;G_{xx}^{(z)}",markerstrokewidth=0.0)
scatter!(ωsreno, real.(res[:,2])./a, label=L"\mathrm{Re}\;G_{yy}^{(z)}",markerstrokewidth=0.0)
scatter!(ωsreno, real.(res[:,3])./a, label=L"\mathrm{Re}\;G_{zz}^{(z)}",markerstrokewidth=0.0)