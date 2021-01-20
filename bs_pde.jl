using SparseArrays
using Plots
using Distributions
include("pde_implicit.jl")


function bs_option(S, K, T, sigma, r, eta::Float64 = +1.0)
    F = exp(r*T)*S
    V = sigma*sigma*T
    d1 = (log(F/K) + 0.5*V)/sqrt(V)
    d2 = d1 - sqrt(V)
    d = Distributions.Normal(0,1)
    return [exp(-r*T)*eta*(F*cdf(d,eta*d1) - K*cdf(d,eta*d2)),
            eta*cdf(d, eta*d1),
            pdf(d, d1)/S/(sqrt(V))]
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
S1, C1, d, Lone, Ltwo = bs_pde_fi(S0, T, r, q, sigma, f, umesh)
# print("lrm: ", d)
# print("\nth:", bs_option(S0, K, T, sigma, r)[2])
lrm_mesh = create_uniform_lrm_mesh(S0, sigma, T, 4, Nt)
S2, C2, d2 = bs_pde_fi(S0, T, r, q, sigma, f, lrm_mesh)
Plots.plot([Lone Ltwo])
print("ghost node L2: ", Ltwo[1])
S3, C3 = bs_pde_fi(S0*exp(sqrt(sigma*sigma*umesh.dT)), T - umesh.dT, r, q, sigma, f)
print("\nfrom pde: ", C3[1,length(S3) ÷ 2 + 1])

# bds = x -> bs_option(x, K, T, sigma, r)[1]
# C_th1 = map(bds, S1)
# C_th2 = map(bds, S2)
err_bps(a,b) = [100*100*abs(x - y)/y for (x, y) in zip(a, b)]
# # Plots.plot([S1 S2], [err_bps(C1[1,:], C_th1) err_bps(C2[1,:], C_th2)], yaxis=:log)
# Plots.plot([S1 S2 S1], [C1[1,:] C2[1,:] C_th1])

