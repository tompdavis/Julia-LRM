using Plots
using SparseArrays

struct pde_mesh        
    S_vect::Array{Float64}
    t_vect::Array{Float64}
end

struct uniform_pde_mesh
    dT::Float64
    dX::Float64
    Nt::Int
    Nx::Int
    X::Array{Float64}
end


function create_uniform_mesh(S0, sigma, T, width, Nt)::uniform_pde_mesh
    V = sigma*sigma*T
    xMin = (log(S0) - width*sqrt(V))
    xMax = (log(S0) + width*sqrt(V))
    dx = (xMax - xMin)/Nx
    dT = T/Nt
    X = range(xMin, xMax, step=dx)
    T = range(0, T, step=dT)
    return uniform_pde_mesh(-dT, dx, Nt, length(X), X)
end


function create_lrm_mesh(dx, dt, v)::pde_mesh
    # create the finite difference mesh which adds two points to the x-direction
    # in order to take advantage of the LRM delta formula

end

function bs_pde_fi( S0, T, r, q, sigma, payoff, umesh::uniform_pde_mesh)
    M = umesh.Nx
    Nt = umesh.Nt
    dx = umesh.dX
    dT = umesh.dT
    C = Array{Float64}(undef, Nt, M)
    S = map(exp, umesh.X)
    C[1,:] = map(payoff, S)
    cNaughtLast = payoff(exp(umesh.X[1] - dx))
    cMaxLast = payoff(exp(umesh.X[end] + dx))
  
    A = spzeros(Float64, M, M)
    g = spzeros(Float64, M, 1)
    a = (r - 0.5*sigma*sigma)*dT/dx
    b = 0.5*sigma*sigma*dT/dx/dx
    for i = 1:M
        if i == 1
            A[i, 1    ] = (1 - r*dT - 2*b - (2*a - 2*b)*(-a/2 + b)/(1 - r*dT - 3/2*a + b)) 
            A[i, 2    ] = (a/2 + b - (a/2 + b)^2/(1 - r*dT - 3/2*a + b))
       elseif i == M
            A[i, M - 1] = (-a/2 + b - (a/2 + b)^2/(1 - r*dT + 3/2*a + b))
            A[i, M    ] = (1 -r*dT - 2*b +(a/2 + b)*(2*a + 2*b)/(1 - r*dT + 3/2*a + b))
       else
            A[i, i - 1] = (-a/2 + b)
            A[i, i    ] = (1 - r*dT - 2*b)
            A[i, i + 1] = (a/2 + b)
        end
    end 
    As = sparse(A)
    for alpha = 2:Nt
        g[1] = (-a/2 + b)/(1 - r*dT - 3/2*a + b)*cNaughtLast
        g[end] = (a/2 + b)/(1 - r*dT + 3/2*a + b)*cMaxLast
        C[alpha, :] = As\(C[alpha - 1, :] - g)
        cNaughtLast = (cNaughtLast - (2*a - 2*b)*C[alpha,1] - (-a/2 + b)*C[alpha,2])/(1 - r*dT - 3/2*a + b);
        cMaxLast = (cMaxLast + (2*a + 2*b)*C[alpha,end] - (a/2 + b)*C[alpha,end-1])/(1 - r*dT + 3/2*a + b);
    end
    return [S, C]
end


function bs_pde_cn( S0, T, r, q, sigma, payoff, umesh::uniform_pde_mesh)
    M = umesh.Nx
    Nt = umesh.Nt
    dx = umesh.dX
    dT = umesh.dT
    C = Array{Float64}(undef, Nt, M)
    S = map(exp, umesh.X)
    C[1,:] = map(payoff, S)
    cNaughtLast = payoff(exp(umesh.X[1] - dx))
    cMaxLast = payoff(exp(umesh.X[end] + dx))
    
    A = spzeros(Float64, M, M)
    B = spzeros(Float64, M, M)
    g = spzeros(Float64, M, 1)
    a = (r - 0.5*sigma*sigma)*dT/dx
    b = 0.5*sigma*sigma*dT/dx/dx
    c1 = (-a/4 + b/2)/(1 - r*dT/2 - 3/4*a + b)
    c2 = (a/4 + b/2)/(1 - r*dT/2 + 3/4*a + b)
    for i = 1:M
        if i == 1
            A[i, 1    ] = (1 - r*dT/2 - b + (-a + 5/2*b)*c1)
            A[i, 2    ] = (a/4 + b/2 + (a/4 - 2*b)*c1)
            A[i, 3    ] = 1/2*b*c1
            B[i, 1    ] = (1 + r*dT/2 + b - (-a + 5/2*b)*c1)
            B[i, 2    ] = (-a/4 - b/2 - (1/4*a - 2*b)*c1)
            B[i, 3    ] = -1/2*b*c1
        elseif i == M
            A[i, M - 2] = 1/2*b*c2
            A[i, M - 1] = (-a/4 + b/2 + (-1/4*a - 2*b)*c2)
            A[i, M    ] = (1 - r*dT/2 - b + (a + 5/2*b)*c2)
            B[i, M - 2] = -1/2*b*c2
            B[i, M - 1] = (a/4 - b/2 - (-a/4 - 2*b)*c2)
            B[i, M    ] = (1 + r*dT/2 + b - (a + 5/2*b)*c2)
        else
            A[i, i - 1] = (-a/4 + b/2)
            A[i, i    ] = (1 - r*dT/2 - b)
            A[i, i + 1] = (a/4 + b/2)
            B[i, i - 1] = (a/4 - b/2)
            B[i, i    ] = (1 + r*dT/2 + b)
            B[i, i + 1] = (-a/4 - b/2)
        end
    end 
    As = sparse(A)
    Bs = sparse(B)
    for alpha = 2:Nt
        g[1] = cNaughtLast*(a/4 - b/2 - (1 + r*dT/2 + 3/4*a - b)*c1)
        g[end] = cMaxLast*(-a/4 - b/2 - (1 + r*dT/2 - 3/4*a - b)*c2) 
        
        C[alpha, :] = As\(Bs*C[alpha-1, :] + g)

        cNaughtLast = (cNaughtLast*(1 + r*dT/2 + 3/4*a - b) + 
                      C[alpha, 1]*(-a + 5/2*b) + C[alpha, 2]*(a/4 - 2*b) + 
                      C[alpha, 3]*1/2*b + C[alpha-1, 1]*(-a + 5/2*b) + 
                      C[alpha-1, 2]*(a/4 - 2*b) + 
                      C[alpha-1, 3]*1/2*b)/(1 - r*dT/2 - 3/4*a + b)
        cMaxLast = (cMaxLast*(1 + r*dT/2 - 3/4*a - b) + C[alpha, end-2]*1/2*b + 
                   C[alpha, end-1]*(-a/4 - 2*b) + C[alpha, end]*(a + 5/2*b) + 
                   C[alpha-1, end-2]*1/2*b + C[alpha-1, end-1]*(-a/4 - 2*b) + 
                   C[alpha-1, end]*(a + 5/2*b))/(1 - r*dT/2 + 3/4*a + b)
    end
    return [S, C]
end

K = 100
f = x -> max(x - K, 0)
T = 1.0
r = 0.05
q = 0.0
sigma = 0.2
S0 = 90
Nt = 1000
Nx = 1000

umesh = create_uniform_mesh(S0, sigma, T, 4, Nt)
S, C = @time bs_pde_cn(S0, T, r, q, sigma, f, umesh)
S1, C1 = @time bs_pde_fi(S0, T, r, q, sigma, f, umesh)

Plots.plot(S, C[2,:])
Plots.plot(S, C[end,:])