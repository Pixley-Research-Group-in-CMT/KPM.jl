using LinearAlgebra
using SparseArrays

# Pauli matrices
σ₀ = Matrix{ComplexF64}(I, 2, 2)
σ₁ = [0 1; 1 0]
σ₂ = [0 -im; im 0]
σ₃ = [1 0; 0 -1]

# 4×4 Gamma matrices (spin/orbital)
Γ₀ = Matrix{ComplexF64}(I, 4, 4)
Γ₁ = kron(σ₁, σ₁)
Γ₂ = kron(σ₂, σ₁)
Γ₃ = kron(σ₃, σ₁)
Γ₄ = kron(σ₀, σ₂)
Γ₅ = kron(σ₀, σ₃)
Γ₁₂ = kron(σ₃, σ₀) # = Γ₁Γ₂/i



# Model parameters
const C₀ = -0.0048
const C₁ = 2.7232
const C₂ = 0.0
const M₀ = -0.1165
const M₁ = 11.9048
const M₂ = 9.4048
const A₁ = 2.7023
const A₂ = 3.1964
const a = 4.334
const a_z = 13.64

const e₀ = C₀ + 2C₁/a_z^2 + 4C₂/a^2
const e₅ = M₀ + 2M₁/a_z^2 + 4M₂/a^2
const t₀ = 2C₂/(3a^2)
const t₀z = C₁/a_z^2
const t₁ = -A₂/(3a)
const t₃z = -A₁/(2a_z)
const t₄ = 0.0 # not given, set to zero or add as needed
const t₅ = 2M₂/(3a^2)
const t₅z = M₁/a_z^2

# Lattice vectors (primitive cell is now 2a_z in z)
a₁ = a * [1.0, 0.0, 0.0]
a₂ = a * [-0.5, sqrt(3)/2, 0.0]
a₄ = 2a_z * [0.0, 0.0, 1.0]  # doubled cell in z

struct MnBiTeBilayerSite
    x::Int
    y::Int
    z::Int
    pos::Vector{Float64}
end

function site_position(x, y, z, orb::Int)
    pos = (x-1)*a₁ + (y-1)*a₂ + (z-1)*a₄
    if orb > 4
        pos[3] += a_z  # Offset for second layer
    end
    return pos
end

function all_sites(Nx, Ny, Nz)
    sites = MnBiTeBilayerSite[]
    for x in 1:Nx, y in 1:Ny, z in 1:Nz
        for orb in 1:8
            push!(sites, MnBiTeBilayerSite(x, y, z, site_position(x, y, z, orb)))
        end
    end
    return sites
end

"""
    MnBiTeLattice8x8(Nx, Ny, Nz; m=0.0)

Constructs the real-space tight-binding Hamiltonian for MnBi₂Te₄ with 8×8 basis per cell.
Returns (Ham, sites).
"""
function MnBiTeLattice(Nx::Int, Ny::Int, Nz::Int; m::Float64=0.0,
    bc_x::Number = 1.0+0im, bc_y::Number = 1.0+0im,bc_z::Number = 1.0+0im)
    Ncell = Nx * Ny * Nz
    N = Ncell * 8 # 8 = 2 layers × 4 orbitals
    site_list = all_sites(Nx, Ny, Nz)

    # Helper to get site index
    function cellidx(x, y, z)
        x = mod(x-1, Nx) + 1 # Periodic Boundary condition
        y = mod(y-1, Ny) + 1
        z = mod(z-1, Nz) + 1
        site = z + Nz * (y - 1) + Nz * Ny * (x - 1)
        return site
    end

    T₀ = kron(σ₀, (e₀*Γ₀ + e₅*Γ₅)) + m*kron(σ₃, Γ₁₂)
    T₁ = kron(σ₀, (-t₀*Γ₀ - im*t₁*Γ₁ - im*t₄*Γ₄ - t₅*Γ₅))
    T₂ = kron(σ₀, (-t₀*Γ₀ - im*t₁*(-Γ₁ + sqrt(3)*Γ₂)/2 - im*t₄*Γ₄ - t₅*Γ₅))
    T₃ = kron(σ₀, (-t₀*Γ₀ - im*t₁*(-Γ₁ - sqrt(3)*Γ₂)/2 - im*t₄*Γ₄ - t₅*Γ₅))
    T₄ = -t₀z * kron(σ₁, Γ₀) + im*t₃z * kron(σ₂, Γ₃) - t₅z * kron(σ₁, Γ₅)
    
    Ind1, Ind2, Vals = Int[], Int[], ComplexF64[]

    function addToList!(pos1, pos2, value, Ind1, Ind2, Vals)
        push!(Ind1, pos1)
        push!(Ind2, pos2)
        push!(Vals, value)
    end

    function addHoppingMat!(cell1, cell2, mat, Ind1, Ind2, Vals)
        pos1 = 8 * (cell1 - 1) .+ (1:8)
        pos2 = 8 * (cell2 - 1) .+ (1:8)
        for i in 1:8, j in 1:8
            if abs(mat[i, j]) > 1e-10  # Avoid adding negligible terms
                addToList!(pos1[i], pos2[j], mat[i, j], Ind1, Ind2, Vals)
            end
        end
    end

    for x in 1:Nx, y in 1:Ny, z in 1:Nz
        curpos = cellidx(x, y, z)
        # Add hopping terms
        pb1 = cellidx(x+1, y, z)
        pb2 = cellidx(x, y+1, z)
        pb3 = cellidx(x-1, y-1, z)
        pbz = cellidx(x, y, z+1)
        addHoppingMat!(curpos, curpos, T₀./2, Ind1, Ind2, Vals) 
        #addHoppingMat!(curpos, curpos, T₀, Ind1, Ind2, Vals) 
        addHoppingMat!(pb1, curpos, T₁*bc_x^(x==Nx), Ind1, Ind2, Vals)
        addHoppingMat!(pb2, curpos, T₂*bc_y^(y==Ny), Ind1, Ind2, Vals)
        addHoppingMat!(pb3, curpos, T₃*bc_x^(x==1)*bc_y^(y==1), Ind1, Ind2, Vals)
        addHoppingMat!(pbz, curpos, T₄*bc_z^(z==Nz), Ind1, Ind2, Vals)
        #=
        addHoppingMat!(curpos, pb1, (T₁*bc_x^(x==Nx))', Ind1, Ind2, Vals)
        addHoppingMat!(curpos, pb2, (T₂*bc_y^(y==Ny))', Ind1, Ind2, Vals)
        addHoppingMat!(curpos, pb3, (T₃*bc_x^(x==1)*bc_y^(y==1))', Ind1, Ind2, Vals)
        addHoppingMat!(curpos, pbz, (T₄*bc_z^(z==Nz))', Ind1, Ind2, Vals)
        =#
    end

    Ham = sparse(Ind1, Ind2, Vals, N, N)
    Ham = Ham + Ham'  


    Valxs, Valys, Valzs = ComplexF64[], ComplexF64[], ComplexF64[]
    for k in 1:length(Ind1)
        i = Ind1[k]
        j = Ind2[k]
        dist = site_list[i].pos - site_list[j].pos
        if abs(bc_z) < 0.99
            all_pos = zeros(Float64, 3,9)
            for Py in -1:1, Px in -1:1
               dpos = dist + Py * Ny * a₁ + Px * Nx * a₂
               all_pos[:,(Py+1)*3 + (Px+2)] = dpos
            end
        else
            all_pos = zeros(Float64, 3,27)
            for Py in -1:1, Px in -1:1, Pz in -1:1
                dpos = dist + Py * Ny * a₁ + Px * Nx * a₂ + Pz * Nz * a₄
                all_pos[:,(Py+1)*9 + (Px+1)*3 + (Pz+2)] = dpos
            end
        end
        ind = sortperm(norm.(eachcol(all_pos)))
        push!(Valxs,Vals[k]*all_pos[1,ind[1]])
        push!(Valys,Vals[k]*all_pos[2,ind[1]])
        push!(Valzs,Vals[k]*all_pos[3,ind[1]])
    end
    Jx = sparse(Ind1, Ind2, Valxs, N, N)
    Jy = sparse(Ind1, Ind2, Valys, N, N)
    Jz = sparse(Ind1, Ind2, Valzs, N, N)
    return Ham, Jx, Jy, Jz
end


function MnBiTeK(kx,ky,kz;m::Float64=0.0)
    k1 = kx
    k2 = (-kx + sqrt(3)*ky)/2
    k3 = (-kx - sqrt(3)*ky)/2
    k4 = kz
    Hk = zeros(ComplexF64, 8, 8)
    Hk += (e₀ - 2*t₀*(cos(k1) + cos(k2) + cos(k3)))*kron(σ₀, Γ₀)
    Hk += -2*t₀z*cos(k4)*kron(σ₁, Γ₀)
    Hk += -t₁*(2*sin(k1)-sin(k2)-sin(k3))*kron(σ₀, Γ₁)
    Hk += -sqrt(3)*t₁*(sin(k2)-sin(k3))*kron(σ₀, Γ₂)
    Hk += 2*t₃z*sin(k4)*kron(σ₂, Γ₃)
    Hk += -2*t₄*(sin(k1) + sin(k2) + sin(k3))*kron(σ₀, Γ₄)
    Hk += (e₅ - 2*t₅*(cos(k1) + cos(k2) + cos(k3)))*kron(σ₀, Γ₅)
    Hk += -2*t₅z*cos(k4)*kron(σ₁, Γ₅)
    Hk += m*kron(σ₃, Γ₁₂)
    return Hk
end

function MnBiTePositionOp(Nx::Int64, Ny::Int64, Nz::Int64)
    Ncell = Nx * Ny * Nz
    N = Ncell * 8 # 8 = 2 layers × 4 orbitals
    site_list = all_sites(Nx, Ny, Nz)
    X_pos = spzeros(Float64, N, N)
    Y_pos = spzeros(Float64, N, N)
    Z_pos = spzeros(Float64, N, N)

    for ind in 1:N
        X_pos[ind, ind] = site_list[ind].pos[1]
        Y_pos[ind, ind] = site_list[ind].pos[2]
        Z_pos[ind, ind] = site_list[ind].pos[3]
    end
    
    return X_pos, Y_pos, Z_pos
end