include("byrnegreenwood.jl")
using Plots

# import ForwardDiff: Dual

folder = "Graphs"

function bump_dg(bound_fn, x, eps)
    p_up = bound_fn(x + eps)
    p = bound_fn(x)
    p_dn = bound_fn(x - eps)
    return [p, (p_up-p_dn)/2/eps, (p_up - 2*p + p_dn)/eps/eps]
end

function heaviside(t)
   0.5 * (sign(t) + 1)
end

# todo: what does gamma look like for a european digital put using Dual numbers?
T = 180/365
σ = 0.3
V = σ^2*T
r = 0.05
K = 90.0
S = 100
N = 1000
ϵ = 0.0001
#like for a digital european?
payoff = x -> heaviside(K - x)
bound_crr = x -> crr_eur(x, K, r, σ, T, N, payoff)
