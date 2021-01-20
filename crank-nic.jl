include("mesh.jl")


function bs_pde_cn( S0, T, r, q, sigma, payoff, umesh::uniform_pde_mesh)
    M = umesh.Nx
    Nt = umesh.Nt
    dx = umesh.dX
    dT = -umesh.dT
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
    for alpha = 2:Nt
        g[1] = cNaughtLast*(a/4 - b/2 - (1 + r*dT/2 + 3/4*a - b)*c1)
        g[end] = cMaxLast*(-a/4 - b/2 - (1 + r*dT/2 - 3/4*a - b)*c2) 
        
        C[alpha, :] = A\(B*C[alpha-1, :] + g)

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


function bs_pde_cn( S0, T, r, q, sigma, payoff, genmesh::pde_mesh)
    M = length(genmesh.X)
    Nt = length(genmesh.T)
    C = Array{Float64}(undef, Nt, M)
    S = map(exp, genmesh.X) 
    C[1,:] = map(payoff, S)
    # Need to choose a dx for the initial BCs
    dx = genmesh.X[2] - genmesh.X[1]
    cNaughtLast = payoff(exp(genmesh.X[1] - dx))
    dx = genmesh.X[end] - genmesh.X[end - 1]
    cMaxLast = payoff(exp(genmesh.X[end] + dx))
    
    A = spzeros(Float64, M, M)
    B = spzeros(Float64, M, M)
    g = spzeros(Float64, M, 1)
    a = 0
    b = 0
    c1 = 0
    c2 = 0
    for alpha = 2:Nt
        dT = genmesh.T[alpha - 1] - genmesh.T[alpha]
        for i = 1:M
            if i == 1
                dx = genmesh.X[i + 1] - genmesh.X[i]
                a = (r - 0.5*sigma*sigma)*dT/dx
                b = 0.5*sigma*sigma*dT/dx/dx
                c1 = (-a/4 + b/2)/(1 - r*dT/2 - 3/4*a + b)
                c2 = (a/4 + b/2)/(1 - r*dT/2 + 3/4*a + b)               
                A[i, 1    ] = (1 - r*dT/2 - b + (-a + 5/2*b)*c1)
                A[i, 2    ] = (a/4 + b/2 + (a/4 - 2*b)*c1)
                A[i, 3    ] = 1/2*b*c1
                B[i, 1    ] = (1 + r*dT/2 + b - (-a + 5/2*b)*c1)
                B[i, 2    ] = (-a/4 - b/2 - (1/4*a - 2*b)*c1)
                B[i, 3    ] = -1/2*b*c1
            elseif i == M
                dx = genmesh.X[i] - genmesh.X[i - 1]
                a = (r - 0.5*sigma*sigma)*dT/dx
                b = 0.5*sigma*sigma*dT/dx/dx
                c1 = (-a/4 + b/2)/(1 - r*dT/2 - 3/4*a + b)
                c2 = (a/4 + b/2)/(1 - r*dT/2 + 3/4*a + b)              
                A[i, M - 2] = 1/2*b*c2
                A[i, M - 1] = (-a/4 + b/2 + (-1/4*a - 2*b)*c2)
                A[i, M    ] = (1 - r*dT/2 - b + (a + 5/2*b)*c2)
                B[i, M - 2] = -1/2*b*c2
                B[i, M - 1] = (a/4 - b/2 - (-a/4 - 2*b)*c2)
                B[i, M    ] = (1 + r*dT/2 + b - (a + 5/2*b)*c2)
            else
                dx = genmesh.X[i + 1] - genmesh.X[i]
                a = (r - 0.5*sigma*sigma)*dT/dx
                b = 0.5*sigma*sigma*dT/dx/dx
                c1 = (-a/4 + b/2)/(1 - r*dT/2 - 3/4*a + b)
                c2 = (a/4 + b/2)/(1 - r*dT/2 + 3/4*a + b)             
                A[i, i - 1] = (-a/4 + b/2)
                A[i, i    ] = (1 - r*dT/2 - b)
                A[i, i + 1] = (a/4 + b/2)
                B[i, i - 1] = (a/4 - b/2)
                B[i, i    ] = (1 + r*dT/2 + b)
                B[i, i + 1] = (-a/4 - b/2)
            end
        end
        g[1] = cNaughtLast*(a/4 - b/2 - (1 + r*dT/2 + 3/4*a - b)*c1)
        g[end] = cMaxLast*(-a/4 - b/2 - (1 + r*dT/2 - 3/4*a - b)*c2) 
        
        C[alpha, :] = A\(B*C[alpha-1, :] + g)

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

