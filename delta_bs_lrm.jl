using Plots
using Distributions

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

function bs_digital(S,K,T,sigma,r,eta::Float64 = -1.0)
    F = exp(r*T)*S
    V = sigma*sigma*T
    d1 = (log(F/K) - 0.5*V)/sqrt(V)
    d = Distributions.Normal(0,1)
    return [exp(-r*T)*cdf(d, eta*d1),
            eta*exp(-r*T)*pdf(d, eta*d1)/S/sqrt(V),
            -eta*exp(-r*T)*pdf(d, eta*d1)*(log(F/K) + 0.5*V)/sqrt(V)/S^2/V]
            # -exp(-r*T)*d1*pdf(d, eta*d1)/S^2/sqrt(V) - exp(-r*T)*1/S^2/sqrt(V)*pdf(d, eta*d1)]
end

function bump_dg(bound_fn, x, eps)
    p_up = bound_fn(x + eps)
    p = bound_fn(x)
    p_dn = bound_fn(x - eps)
    return [p, (p_up-p_dn)/2/eps, (p_up - 2*p + p_dn)/eps/eps]
end

T = 180/365
σ = 0.3
V = σ^2*T
r = 0.05
K = 90.0
S = 100
N = 1000
ϵ = 0.0001

# eVect = 2*dS:-0.01:0.5*dS
# d_b = Array{Float64}(undef, length(eVect))
# d_th = Array{Float64}(undef, length(eVect))
# bound_bs = x -> bs_option(x, K, T, σ, r)
# for i = 1:length(eVect)
#     bdg = bump_dg(bound_bs, S, eVect[i])
#     d_b[i] = bdg[2][1]
#     d_th[i] = bdg[1][2]
# end
# Plots.plot(eVect, 100*100*(d_th - d_b)./d_th)

dT = 0.001
R = r*dT
t = T + dT
dV = σ^2*dT
dS = sqrt(dV)

Cup = bs_option(S*exp(dS), K, t, σ, r)[1]
Cdn = bs_option(S*exp(-dS), K, t, σ, r)[1]
d_try = exp(-R)*0.5/S/dS*(Cup - Cdn)

p = 1/2 + 1/2*(R - 1/2*V)/√V
q = 1 - p
C_try = exp(-R)*(p*Cup + q*Cdn)
C = bs_option(S, K, T, σ, r, 1.0)[1]
d_th = bs_option(S, K, T, σ, r, 1.0)[2]

println("One step tree price: ", C_try)
println("Theoretical price: ", C)
println("Absolute error (bps): ", 100*100*(C - C_try)/C)

println("LRM delta: ", d_try)
println("Theoretical delta: ", d_th)
println("Absolute error(bps): ", 100*100*(d_th-d_try)/d_th)

# Now I can try extrapolating to get the real delta 


# bound_bs = x -> bs_option(x, K, T, σ, r, 1.0)
# dg = bump_dg(bound_bs, S, dS)
# d_bump = dg[2][1]
