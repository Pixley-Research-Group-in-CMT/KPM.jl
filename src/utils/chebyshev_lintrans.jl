function chebyshev_lin_trans(x::Real, n_grid::Array, mu_tilde::Array)
    #@assert length(n_grid) == length(mu_tilde)
    T_xn = chebyshevT_xn(x, n_grid)
    res = T_xn * mu_tilde
    #@assert length(res) == 1
    return sum(res)
end



function chebyshev_lin_trans(x_grid::Array, n_grid::Array, mu_tilde::Array)
    Nx = length(x_grid)
    Nn = length(n_grid)
    est_size = Nx * Nn / 65536  # in MB
    # This holds ~4096Nc, 16384Ntilde for example.
    if est_size < 1000
        T_xn = chebyshevT_xn(x_grid, n_grid)
        return T_xn * mu_tilde
    end

    # otherwise use less memory
    # TODO multithreading?
    y = complex(x_grid) * 0
    for nx = ProgressBar(1:Nx)
        y[nx] = dot(chebyshevT.(n_grid, x_grid[nx]), mu_tilde)
    end
    return y
end



