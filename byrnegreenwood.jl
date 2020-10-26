import ForwardDiff: Dual
using Distributions

function crr_am_put(S, K, r, σ, t, N, payoff)
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


function crr_eur_put(S, K, r, σ, t, N, payoff)
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

function crr_central(S, K, r, σ, t, N, payoff)
    Δt = t/N
    R = r*Δt
    V = σ^2*Δt
    p = 1/2
    q = 1/2
    Z = [payoff(S*exp((r-1/2*σ^2)*t + (2*i - N)*√V)) for i = 0:N]
    d = 0
    g = 0
    for n = N-1:-1:0
        for i = 0:n
            y = (q*Z[i+1] + p*Z[i+2])/exp(R)
            d = n == 0 ? exp(-R)*((0.5/sqrt(V)/S*Z[i + 2] - 0.5/sqrt(V)/S*Z[i + 1])) : 0
            g = n == 0 ? exp(-R)*(1/2*(-1/sqrt(V)/S/S)*Z[i + 2] + 1/2*(1/sqrt(V)/S/S)*Z[i + 1]) : 0
            Z[i + 1] = y
        end
    end
    return [Z[1], d, g]
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
