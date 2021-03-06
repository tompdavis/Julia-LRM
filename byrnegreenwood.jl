import ForwardDiff: Dual
using Distributions
using Statistics
using Random

seed = 43123

function crr_am(S, K, r, σ, t, N, payoff)
    Δt = t/N
    R = r*Δt
    V = σ^2*Δt
    p = 1/2 + 1/2*(R - 1/2*V)/√V
    q = 1 - p
    Z = [payoff(S*exp((2*i - N)*√V)) for i = 0:N]
    d = 0
    for n = N-1:-1:0
        for i = 0:n
            x = payoff(S*exp((2*i - n)*√V))
            y = (q*Z[i+1] + p*Z[i+2])/exp(R)
            d = n == 0 ? exp(-R)*(0.5/sqrt(V)/S*Z[i + 2] - 0.5/sqrt(V)/S*Z[i + 1]) : 0
            Z[i + 1] = max(x, y)
        end
    end
    return [Z[1], d]
end


function crr_eur(S, K, r, σ, t, N, payoff)
    Δt = t/N
    R = r*Δt
    V = σ^2*Δt
    p = 1/2 + 1/2*(R - 1/2*V)/√V
    q = 1 - p
    Z = [payoff(S*exp((2*i - N)*√V)) for i = 0:N]
    d = 0
    for n = N-1:-1:0
        for i = 0:n
            y = (q*Z[i+1] + p*Z[i+2])/exp(R)
            d = n == 0 ? exp(-R)*((0.5/sqrt(V)/S*Z[i + 2] - 0.5/sqrt(V)/S*Z[i + 1])) : 0
            Z[i + 1] = y
        end
    end
    return [Z[1], d]
end

# function crr_central(S, K, r, σ, t, N, payoff)
#     Δt = t/N
#     R = r*Δt
#     V = σ^2*Δt
#     p = 1/2
#     q = 1/2
#     Z = [payoff(S*exp((r-1/2*σ^2)*t + (2*i - N)*√V)) for i = 0:N]
#     d = 0
#     g = 0
#     for n = N-1:-1:0
#         for i = 0:n
#             y = (q*Z[i+1] + p*Z[i+2])/exp(R)
#             d = n == 0 ? exp(-R)*((0.5/sqrt(V)/S*Z[i + 2] - 0.5/sqrt(V)/S*Z[i + 1])) : 0
#             g = n == 0 ? exp(-R)*(1/2*(-1/sqrt(V)/S/S)*Z[i + 2] + 1/2*(1/sqrt(V)/S/S)*Z[i + 1]) : 0
#             Z[i + 1] = y
#         end
#     end
#     return [Z[1], d, g]
# end

function lsmc_am_put(S, K, r, σ, t, N, P)
    Δt = t/N
    R = exp(r*Δt)
    T = typeof(S*exp(-σ^2*Δt/2 + σ*sqrt(Δt)*0.1)/R)
    rng = MersenneTwister(seed)
    X = Array{T}(undef, N+1, P)
    for p = 1:P
        X[1, p] = x = S
        for n = 1:N
            x *= R*exp(-σ^2*Δt/2 + σ*sqrt(Δt)*randn(rng))
            X[n+1, p] = x
        end
    end

    V = [max(K - x, 0)/R for x in X[N+1, :]]

    for n = N-1:-1:1
        I = V .!= 0
        A = [x^d for d = 0:3, x in X[n+1, :]]
        β = A[:, I]' \ V[I]
        cV = A'*β
        for p in 1:P
            ev = max(K - X[n+1, p], 0)
            if I[p] && cV[p] < ev
                V[p] = ev/R
            else
                V[p] = V[p]/R
            end
        end
    end
    return max(mean(V), K - S)
end

function lsmc_am_put_lrm(S, K, r, σ, t, N, P)
    Δt = t/N
    R = exp(r*Δt)
    T = typeof(S*exp(-σ^2*Δt/2 + σ*sqrt(Δt)*0.1)/R)
    dt = Δt/100
    dV = σ^2*dt

    rng = MersenneTwister(seed)
    X = Array{T}(undef, N+1, P)
    Xup = Array{T}(undef, 1, P)
    Xdn = Array{T}(undef, 1, P)
    for p = 1:P
        X[1, p] = x = S
        for n = 1:N
            rn = randn(rng)
            x *= R*exp(-σ^2*Δt/2 + σ*sqrt(Δt)*rn)
            if n == 1
                Xup[1,p] = exp(r*(Δt-dt))*exp(sqrt(dV))*X[n,p]*exp(r*(Δt-dt))*exp(-σ^2*(Δt - dt)/2 + σ*sqrt(Δt - dt)*rn)
                Xdn[1,p] = exp(r*(Δt-dt))*exp(-sqrt(dV))*X[n,p]*exp(r*(Δt-dt))*exp(-σ^2*(Δt - dt)/2 + σ*sqrt(Δt - dt)*rn)
            end
            X[n+1, p] = x
        end
    end

    V = [max(K - x, 0)/R for x in X[N+1, :]]
    Vup = [max(K - x, 0)/R for x in exp(sqrt(dV))*X[N+1, :]]
    Vdn = [max(K - x, 0)/R for x in exp(-sqrt(dV))*X[N+1, :]]

    for n = N-1:-1:1
        I = V .!= 0
        Iup = Vup .!= 0
        Idn = Vdn .!= 0
        A = [x^d for d = 0:3, x in X[n+1, :]]
        β = A[:, I]' \ V[I]
        cV = A'*β
        if n > 1  
            Aup = [x^d for d = 0:3, x in exp(sqrt(dV))*X[n+1, :]]
            βup = Aup[:, Iup]' \ Vup[Iup]
            Adn = [x^d for d = 0:3, x in exp(-sqrt(dV))*X[n+1, :]]
            βdn = Adn[:, Idn]' \ Vdn[Idn]
            cVup = Aup'*βup
            cVdn = Adn'*βdn
        else
            Aup = [x^d for d = 0:3, x in Xup[1, :]]
            βup = Aup[:, Iup]' \ Vup[Iup]
            Adn = [x^d for d = 0:3, x in Xdn[1, :]]
            βdn = Adn[:, Idn]' \ Vdn[Idn]
            cVup = Aup'*βup
            cVdn = Adn'*βdn
        end
        for p in 1:P
            ev = max(K - X[n+1, p], 0)
            if I[p] && cV[p] < ev
                V[p] = ev/R
            else
                V[p] = V[p]/R
            end
            if n > 1
                ev = max(K - exp(sqrt(dV))*X[n+1,p], 0)
            else
                ev = max(K - Xup[1, p], 0)
            end
            if Iup[p] && cVup[p] < ev
                Vup[p] = ev/R
            else
                Vup[p] = Vup[p]/R
            end
            if n > 1
                ev = max(K - exp(-sqrt(dV))*X[n+1,p], 0)
            else
                ev = max(K - Xdn[1, p], 0)
            end
            if Idn[p] && cVdn[p] < ev
                Vdn[p] = ev/R
            else
                Vdn[p] = Vdn[p]/R
            end
        end
    end
    D = exp(r*dt)*0.5/S/sqrt(dV)*[vup - vdn for (vup, vdn) in zip(Vup, Vdn)]
    return max(mean(V), K - S), mean(D), std(D)/sqrt(P)
end

function mc_eur(payoff, S, r, σ, t, N, dt=0.001)
    rng = MersenneTwister(seed)
    R = exp(r*t)
    T = typeof(S*exp(-σ^2*t/2 + σ*sqrt(t)*0.1)/R)
    X = Array{T}(undef, N)
    Xup = Array{T}(undef, N)
    Xdn = Array{T}(undef, N)
    dV = σ^2*dt
    dR = exp(r*dt)
    for n = 1:N
        rn = randn(rng)
        X[n] = S*exp(-σ^2*t/2 + σ*sqrt(t)*rn)*R
        Xup[n] = (S*exp(sqrt(dV)))*exp(-σ^2*(t-dt)/2 + σ*sqrt(t-dt)*rn)*R/dR
        Xdn[n] = (S*exp(-sqrt(dV)))*exp(-σ^2*(t-dt)/2 + σ*sqrt(t-dt)*rn)*R/dR
    end
    V = [payoff(x)/R for x in X]
    D = 0.5/S/sqrt(dV)/R*[payoff(xup) - payoff(xdn) for (xup, xdn) in zip(Xup, Xdn)]
    return mean(V), mean(D)
end

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

function lrm_bump(bound_fn, S, t, r, sigma, dt)
   # the bound function now has to take in S and t 
    dV = sigma^2*dt
    Cup = bound_fn(S*exp(sqrt(dV)), t - dt)
    Cdn = bound_fn(S*exp(-sqrt(dV)), t - dt)
    return exp(-r*dt)*0.5/S/sqrt(dV)*(Cup - Cdn) 
end

