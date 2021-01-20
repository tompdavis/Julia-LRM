include("pde_mesh.jl")


function bs_pde_fi( S0, T, r, q, sigma, payoff)
    umesh = create_uniform_mesh(S0, sigma, T, 4, 1000)
    return bs_pde_fi(S0, T, r, q, sigma, payoff, umesh)
end

function bs_pde_fi( S0, T, r, q, sigma, payoff, lrm_mesh::uniform_lrm_mesh)
    return bs_pde_fi(S0, T, r, q, sigma, payoff, lrm_mesh.umesh)
end


function bs_pde_fi( S0, T, r, q, sigma, payoff, umesh::uniform_pde_mesh)
    M = umesh.Nx
    Nt = umesh.Nt
    dx = umesh.dX
    dT = umesh.dT
   
    C = Array{Float64}(undef, Nt, M)
    S = map(exp, umesh.X)
    C[end,:] = map(payoff, S)
    cNaughtLast = payoff(exp(umesh.X[1] - dx))
    cMaxLast = payoff(exp(umesh.X[end] + dx))
  
    A = spzeros(Float64, M, M)
    g = spzeros(Float64, M, 1)
    sqrt_dV = sqrt(sigma*sigma*dT)
    # Nearest node to the LRM point
    lrm_idx = Int(round(sqrt_dV ÷ dx))
    lrm_dx1 = sqrt_dV - (lrm_idx - 1)*dx
    lrm_dx2 = (lrm_idx + 1)*dx - sqrt_dV
    ecc = lrm_dx2/lrm_dx1
    mid_idx = M ÷ 2 + 1
    Lone = zeros(Float64, Nt, 1)
    Ltwo = zeros(Float64, Nt, 1)
    Lone[end] = payoff(exp(umesh.X[mid_idx - lrm_idx] - lrm_dx1))
    Ltwo[end] = payoff(exp(umesh.X[mid_idx + lrm_idx] + lrm_dx1))
    # With the new definition, ahat and bhat are equal to a and b
    a = (r - 0.5*sigma*sigma)*dT/dx
    xbar = 1/2*(lrm_dx1 + lrm_dx2)
    a_hat = (r - 0.5*sigma*sigma)*dT/xbar
    b = 0.5*sigma*sigma*dT/dx/dx
    b_hat = 0.5*sigma*sigma*dT/xbar/xbar
    delta = 0
    for i = 1:M
        if i == 1
            A[i, 1    ] = (1 + r*dT + 2*b + (a/2 - b)*(2*a - 2*b)/(1 + r*dT + 3/2*a - b))
            A[i, 2    ] = ((-a/2 - b) + (a/2 - b)*(-a/2 + b)/(1 + r*dT + 3/2*a - b))
       elseif i == M  
            A[i, M - 1] = ((a/2 - b) + (a/2 + b)*(-a/2 - b)/(1 + r*dT - 3/2*a - b))
            A[i, M    ] = (1 + r*dT + 2*b + (-2*a - 2*b)*(-a/2 - b)/(1 + r*dT - 3/2*a - b))
       else
            A[i, i - 1] = (a/2 - b)
            A[i, i    ] = (1 + r*dT + 2*b)
            A[i, i + 1] = (-a/2 - b)
        end
    end 
    for alpha = Nt:-1:2
        g[1] = (a/2 - b)/(1 + r*dT + 3/2*a - b)*cNaughtLast
        g[end] = (-a/2 - b)/(1 + r*dT - 3/2*a - b)*cMaxLast
        C[alpha - 1, :] = A\(C[alpha, :] - g)
        cNaughtLast = (cNaughtLast + (2*a - 2*b)*C[alpha - 1, 1] 
                   + (-a/2 + b)*C[alpha - 1, 2])/(1 + r*dT + 3/2*a - b);
        cMaxLast = (cMaxLast - (2*a + 2*b)*C[alpha - 1, end] 
                    + (a/2 + b)*C[alpha - 1, end - 1])/(1 + r*dT - 3/2*a - b)

        Lone[alpha - 1] = (Lone[alpha] + (a_hat/2*ecc + b_hat/2*(1+ecc))*C[alpha - 1, mid_idx - lrm_idx + 1]
                + (-a_hat/2/ecc + b_hat/2*(1 + 1/ecc))*C[alpha - 1, mid_idx - lrm_idx - 1])/(1 + r*dT 
                - a_hat/2*(1/ecc - ecc) + b_hat*(1 + 1/2*ecc + 1/2/ecc))

        Ltwo[alpha - 1] = (Ltwo[alpha] + (a_hat/2/ecc + b_hat*(1/2 + 1/2/ecc))*C[alpha - 1, mid_idx + lrm_idx + 1]
                + (-a_hat/2*ecc + b_hat*(1/2 + 1/2*ecc))*C[alpha - 1, mid_idx + lrm_idx - 1])/(1 + r*dT 
                - a_hat/2*(ecc - 1/ecc) + b_hat*(1 + 1/2*ecc + 1/2/ecc))
        if alpha == 3
            delta = exp(-r*dT)*0.5/S0/sqrt_dV*(Ltwo[alpha - 1] - Lone[alpha - 1])
        end
        # Lone = (Lone + C[alpha - 1, mid_idx - lrm_idx - 1]*(a_hat*lrm_dx2/lrm_dx1 - 2*b_hat/lrm_dx1) 
        #         + C[alpha - 1, mid_idx - lrm_idx]*(-a_hat*lrm_dx1/lrm_dx2 -2*b_hat/lrm_dx2))/(1 + r*dT - a*lrm_dx1/lrm_dx2 + a*lrm_dx2/lrm_dx1 - 2*b_hat*(1/lrm_dx1 + 1/lrm_dx2))
        # Ltwo = (Ltwo + C[alpha - 1, mid_idx + lrm_idx]*(-a_hat*lrm_dx2/lrm_dx1 -2*b_hat/lrm_dx1 )  
        #         + C[alpha - 1, mid_idx + lrm_idx + 1]*(-a_hat*lrm_dx1/lrm_dx2 -2*b_hat/lrm_dx2))/(1 + r*dT + a_hat*(lrm_dx2/lrm_dx1 - lrm_dx1/lrm_dx2) - 2*b_hat*(1/lrm_dx1 + 1/lrm_dx2))
    end
    
    return [S, C, delta, Lone, Ltwo]
end

