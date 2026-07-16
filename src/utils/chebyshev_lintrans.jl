"""
Evaluate the Chebyshev series Σ_n μ̃_n T_n(x) on a point or grid of x values.
For moderate sizes the T_{x,n} matrix is materialized and applied as one
matrix-vector product; above ~1 GB it falls back to a per-point loop.
"""
function chebyshev_lin_trans(x::Real, n_grid::Array, mu_tilde::Array)
    T_xn = chebyshevT_xn(x, n_grid)
    res = T_xn * mu_tilde
    return sum(res)
end

function chebyshev_lin_trans(x_grid::Array, n_grid::Array, mu_tilde::Array)
    Nx = length(x_grid)
    Nn = length(n_grid)
    est_size = Nx * Nn / 65536  # in MB, at 16 bytes/entry
    if est_size < 1000
        T_xn = chebyshevT_xn(x_grid, n_grid)
        return T_xn * mu_tilde
    end

    # otherwise use less memory
    y = zeros(complex(eltype(x_grid)), Nx)
    Threads.@threads for nx = 1:Nx
        y[nx] = dot(chebyshevT.(n_grid, x_grid[nx]), mu_tilde)
    end
    return y
end
