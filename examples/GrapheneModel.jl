using LinearAlgebra
using SparseArrays

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

function GraphenePositionOps(Nx::Int64, Ny::Int64)
    N = 2 * Nx * Ny
    site_list = all_sites(Nx, Ny)
    
    # Position operators
    X_pos = spzeros(Float64, N, N)
    Y_pos = spzeros(Float64, N, N)
    for i in 1:N
        X_pos[i, i] = site_list[i].pos[1]
        Y_pos[i, i] = site_list[i].pos[2]
    end
    
    return X_pos, Y_pos
end