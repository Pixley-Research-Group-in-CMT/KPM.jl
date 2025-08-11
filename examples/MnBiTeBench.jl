using Plots
using LaTeXStrings

include("MnBiTeModel.jl")

Nx, Ny, Nz = 3,3, 3

Ham, Jx, Jy, Jz = MnBiTeLattice(Nx,Ny,Nz; m=0.0, bc_z=1.0+0im)
ishermitian(Ham)
@time Es, Vs = eigen(Matrix(Ham))
Es = real.(Es)

bx = [1,sqrt(3)/3,0].*(2*pi)
by = [0,2*sqrt(3)/3,0].*(2*pi)
bz = [0,0,1/2].*(2*pi)
Etemp = zeros(8,Nx,Ny,Nz)
for x=1:Nx, y=1:Ny, z=1:Nz
	k1 = (x-1)*bx./Nx #(x+0.5)*bx./Nx # no 2*pi!!
	k2 = (y-1)*by./Ny #(y+0.5)*by./Ny
    k3 = (z-1)*bz./Nz #(z+0.5)*bz./Nz
    ktot = k1 + k2 + k3
    kx, ky, kz = ktot[1], ktot[2], ktot[3]
	Etemp[:,x,y,z] = eigvals(MnBiTeK(kx,ky,kz))
end
Eana = reshape(Etemp, 8*Nx*Ny*Nz)
sort!(Eana)


scatter([Es Eana],
	markerstrokewidth=0,markersize=[3 2],
	label=["Lattice" "Momentum"],
    ylabel = L"E(\mathbf{k})",xlabel = L"\mathrm{eigenstate}\; k",
     framestyle = :box,grid=false,legend=:topleft,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
    )
#savefig("~/Desktop/FigBenchmark.pdf")
maximum(abs.(Es .- Eana))
