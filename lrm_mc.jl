include("byrnegreenwood.jl")
# import ForwardDiff: Dual

S = 100
K = 90
sigma = 0.3
r = 0.05
t = 180/365
N = 1000
P = 100000

# Cam = lsmc_am_put(Dual(S, 1.0), K, r, sigma, t, N, P)
# println(Cam)
# Clrm = lsmc_am_put_lrm(S, K, r, sigma, t, N, P)
# println(Clrm)

call_payoff = x -> max(x - K, 0)

C_th = bs_option(S, K, t, sigma, r, 1.0)

dt1 = 0.005
dt2 = 0.001

C_mc1 = mc_eur(call_payoff, S, r, sigma, t, P, dt1)
C_mc2 = mc_eur(call_payoff, S, r, sigma, t, P, dt2)
d1 = C_mc1[2]
d2 = C_mc2[2]

# Extrapolation of monte carlo doesn't work, even if the seed is fixed.
extrap_d = (d2*dt1 - d1*dt2)/(dt1-dt2)

println("error in d1:", d1-extrap_d)
println("error in d2:", d2-extrap_d)
println(extrap_d)
println(C_th[2])
