
struct pde_mesh        
    X::Array{Float64}
    T::Array{Float64}
end

struct uniform_pde_mesh
    dT::Float64
    dX::Float64
    Nt::Int
    Nx::Int
    X::Array{Float64}
end


struct uniform_lrm_mesh
    umesh::uniform_pde_mesh
    lrm_idx::Int
end

function create_uniform_mesh(S0, sigma, T, width, Nt)::uniform_pde_mesh
    V = sigma*sigma*T
    xMin = (log(S0) - width*sqrt(V))
    xMax = (log(S0) + width*sqrt(V))
    dx = (xMax - xMin)/Nx
    dT = T/Nt
    X = range(xMin, xMax, step=dx)
    # The -dT is due to a sign error in my derivation
    # where I assumed forward stepping, I will correct
    return uniform_pde_mesh(dT, dx, Nt, length(X), X)
end


function create_uniform_lrm_mesh(S0, sigma, T, width, Nt)::uniform_lrm_mesh
# creates a uniform grid, but making sure dV is on the grid
# by scaling dx such that n*dx = dV for some n 
# should probably do some error checking, but not right now

    umesh = create_uniform_mesh(S0, sigma, T, width, Nt)
    log_dS = sqrt(sigma*sigma*umesh.dT)
    n = Int(round(log_dS ÷ umesh.dX))
    new_dx = log_dS/n 
    mid_idx = umesh.Nx ÷ 2 + 1
    new_x = range(umesh.X[mid_idx] - (mid_idx - 1)*new_dx, umesh.X[mid_idx] + (mid_idx - 1)*new_dx, step=new_dx)
    return uniform_lrm_mesh(
        uniform_pde_mesh(umesh.dT, new_dx, Nt, length(new_x), new_x),
        n)
end

function create_lrm_mesh(S0, sigma, T, width, Nt)::pde_mesh
    # create the finite difference mesh which adds two points to the x-direction
    # in order to take advantage of the LRM delta formula

    # just for testing purposes, create a uniform grid.
    umesh = create_uniform_mesh(S0, sigma, T, width, Nt)
    T = range(0, T, step=T/Nt)
    return pde_mesh(umesh.X, T)
end

