include("byrnegreenwood.jl")
using Plots

T = 180/365
σ = 0.3
V = σ^2*T
r = 0.05
K = 90.0
S = 100
N = 1000
ϵ = 0.0001

# Now I can try extrapolating to get the real delta 
d_th = bs_option(S, K, T, σ, r, 1.0)[2]
bound_bs(x,t) = bs_option(x, K, t, σ, r, 1.0)
dt1 = 0.005
dt2 = 0.001
d1 = lrm_bump(bound_bs, S, T, r, σ, dt1)[1] 
d2 = lrm_bump(bound_bs, S, T, r, σ, dt2)[1] 

extrap_d = (d2*dt1 - d1*dt2)/(dt1-dt2)

println(extrap_d)
println(d_th)
println(100*100*(extrap_d-d_th)/d_th)

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

# t = T + dT
# dV = σ^2*dT
# dS = sqrt(dV)

# Cup = bs_option(S*exp(dS), K, t, σ, r)[1]
# Cdn = bs_option(S*exp(-dS), K, t, σ, r)[1]
# d_try = exp(-R)*0.5/S/dS*(Cup - Cdn)

# p = 1/2 + 1/2*(R - 1/2*V)/√V
# q = 1 - p
# C_try = exp(-R)*(p*Cup + q*Cdn)
# C = bs_option(S, K, T, σ, r, 1.0)[1]
# d_th = bs_option(S, K, T, σ, r, 1.0)[2]

# println("One step tree price: ", C_try)
# println("Theoretical price: ", C)
# println("Absolute error (bps): ", 100*100*(C - C_try)/C)

# println("LRM delta: ", d_try)
# println("Theoretical delta: ", d_th)
# println("Absolute error(bps): ", 100*100*(d_th-d_try)/d_th)



# bound_bs = x -> bs_option(x, K, T, σ, r, 1.0)
# dg = bump_dg(bound_bs, S, dS)
# d_bump = dg[2][1]
