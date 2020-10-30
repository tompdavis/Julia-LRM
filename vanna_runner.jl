include("./byrnegreenwood.jl")
using Plots
using Distributions

function bs_option_g(S, K, T, sigma, r, eta::Float64 = +1.0)
    F = exp(r*T)*S
    V = sigma*sigma*T
    d1 = (log(F/K) + 0.5*V)/sqrt(V)
    d2 = d1 - sqrt(V)
    d = Distributions.Normal(0,1)
    return [exp(-r*T)*eta*(F*cdf(d,eta*d1) - K*cdf(d,eta*d2)),
            eta*cdf(d, eta*d1),
            pdf(d, d1)/S/(sqrt(V)),
            S*pdf(d, -d1)*sqrt(T),
            -d2/sigma*pdf(d, -d1)]
end


folder = "Likely"
T = 180/365
σ = 0.3
V = σ^2*T
r = 0.05
K = 90.0
S = 100
N = 1000
ϵ = 0.0001

S0 = Dual(S, 1, 0)
sigma = Dual(σ, 0, 1)

C = bs_option_g(S0, K, T, sigma, r, -1.0)
vega_ad = C[1].partials[2]
vanna_ad = C[2].partials[2]

payoff = x -> max(K - x, 0)

C_lrm = crr_eur_put(S0, K, r, sigma, T, N, payoff)

S_vect = [S*exp(z*sqrt(V)) for z in -2.5:0.10:2.5]
S_zoom = 95.0:0.05:105.0

function make_vega_vanna_vect(svect, K, T, sigma, r, N, payoff)
    payoff = x -> max(K - x, 0)
    vega_bs = Array{Float64}(undef, length(svect))
    vanna_bs = Array{Float64}(undef, length(svect))
    vega_ad = Array{Float64}(undef, length(svect))
    vanna_mixed = Array{Float64}(undef, length(svect))
    for i in 1:length(svect)
        sdual = Dual(svect[i], 1, 0)
        vol = Dual(sigma, 0, 1)
        C = bs_option_g(sdual, K, T, vol, r)
        vega_bs[i] = C[4].value
        vanna_bs[i] = C[5].value
        C_crr = crr_eur_put(sdual, K, r, vol, T, N, payoff)
        vega_ad[i] = C_crr[1].partials[2]
        vanna_mixed[i] = C_crr[2].partials[2]
    end
    return [vega_bs, vanna_bs, vega_ad, vanna_mixed]
end

res = make_vega_vanna_vect(S_vect, K, T, σ, r, N, payoff)
vega_bs = res[1]
vanna_bs = res[2]
vega_ad = res[3]
vanna_mixed = res[4]

lw = 3
folder = "Likely"
Plots.plot(S_vect, [vega_bs, vega_ad],
                   labels=["Closed Form" "AD"],
                   ylabel="Vega",
                   xlabel="S",
                   linewidth=lw)
fig_name = "vega.png"
Plots.savefig(string(folder, "\\", fig_name))

Plots.plot(S_vect, [vanna_bs, vanna_mixed],
                   labels=["Closed Form" "LRM + AD"],
                   ylabel="Vanna",
                   xlabel="S",
                   linewidth=lw)
fig_name = "vanna.png"
Plots.savefig(string(folder, "\\", fig_name))

res = make_vega_vanna_vect(S_zoom, K, T, σ, r, N, payoff)
vega_bs = res[1]
vanna_bs = res[2]
vega_ad = res[3]
vanna_mixed = res[4]

Plots.plot(S_zoom, [vega_bs, vega_ad],
                   labels=["Closed Form" "AD"],
                   ylabel="Vega",
                   xlabel="S",
                   linewidth=lw)
fig_name = "vega-zoom.png"
Plots.savefig(string(folder, "\\", fig_name))

Plots.plot(S_zoom, [vanna_bs, vanna_mixed],
                   labels=["Closed Form" "LRM + AD"],
                   ylabel="Vanna",
                   xlabel="S",
                   linewidth=lw)
fig_name = "vanna-zoom.png"
Plots.savefig(string(folder, "\\", fig_name))


function makeplots(S_vect, exercise="Eur", ext="")
    if exercise == "Eur"
        bound_tree = x -> crr_eur(x, K, r, sigma, T, N, payoff)
    elseif exercise == "Am"
        bound_tree = x -> crr_am(x, K, r, sigma, T, N, payoff)
    end

    C = [bound_tree(Dual(x, 1.0, 0.0)) for x in S_vect]
    bs = [bs_option_g(x, K, T, σ, r, -1.0) for x in S_vect]

    price_tree = [C[n][1].value for n in 1:length(S_vect)]
    price_bs = [bs[n][1] for n in 1:length(S_vect)]
    delta_bs = [bs[n][2] for n in 1:length(S_vect)]
    gamma_bs = [bs[n][3] for n in 1:length(S_vect)]

    delta_lrm = [C[n][2].value for n in 1:length(S_vect)]
    delta_ad = [C[n][1].partials[1] for n in 1:length(S_vect)]
    gamma_mixed = [C[n][2].partials[1] for n in 1:length(S_vect)]

    if exercise == "Eur"
        to_plot = [price_bs, price_tree]
        labels = ["Closed Form" "CRR"]
    elseif exercise == "Am"
        to_plot = [price_tree]
        labels = "CRR"
    end

    Plots.plot(S_vect, to_plot,
                       labels=labels,
                       linewidth=lw,
                       xlabel="S",
                       ylabel="C")
    fig_name = "Price-Vanilla"
    Plots.savefig(string(folder, "\\", fig_name, exercise, ext, ".png"))

    if exercise == "Eur"
        to_plot = [delta_bs, delta_lrm, delta_ad]
        labels = ["Closed Form" "LRM" "AD"]
    elseif exercise == "Am"
        to_plot = [delta_lrm, delta_ad]
        labels = ["LRM" "AD"]
    end

    Plots.plot(S_vect, to_plot,
                        labels=labels,
                        xlabel="S",
                        ylabel="Δ",
                        linewidth=lw)
    fig_name = "Delta-Vanilla"
    Plots.savefig(string(folder, "\\", fig_name, exercise, ext, ".png"))

    Plots.plot(S_vect, 100*100*[(delta_bs - delta_lrm)./delta_bs,
                                 (delta_bs - delta_ad)./delta_bs],
                                 labels=["LRM Relative Error" "AD Relative Error"],
                                 xlabel="S",
                                 ylabel="bps",
                                 linewidth=lw)
    fig_name = "DeltaRel-Vanilla"
    Plots.savefig(string(folder, "\\", fig_name, exercise, ext, ".png"))

    Plots.plot(S_vect, 100*100*[(price_bs - price_tree)./price_bs,
                                (delta_bs - delta_lrm)./delta_lrm],
                        ylabel="bps",
                        xlabel="S",
                        linewidth=lw,
                        labels=["Price Relative Error" "LRM Δ Relative Error"])
    fig_name = "PriceRel-Vanilla"
    Plots.savefig(string(folder, "\\", fig_name, exercise, ext, ".png"))

    if exercise == "Eur"
        to_plot =[gamma_bs, gamma_mixed]
        labels = ["Closed Form" "LRM + AD"]
    elseif exercise == "Am"
        to_plot = gamma_mixed
        labels = "LRM + AD"
    end

    Plots.plot(S_vect, to_plot,
                        labels=labels,
                        xlabel="S",
                        ylabel="Γ",
                        linewidth=lw)
    fig_name = "Gamma-Vanilla"
    Plots.savefig(string(folder, "\\", fig_name, exercise, ext, ".png"))

    Plots.plot(S_vect, 100*100*[(delta_bs - delta_lrm)./delta_bs,
                                (gamma_bs - gamma_mixed)./delta_bs],
                        labels=["Δ Relative Error" "Γ Relative Error"],
                        xlabel="S",
                        ylabel="bps",
                        linewidth=lw)
    fig_name = "GreeksRel-Vanilla"
    Plots.savefig(string(folder, "\\", fig_name, exercise, ext, ".png"))

end

makeplots(S_vect)
makeplots(S_zoom, "Eur", "-zoom")
makeplots(S_vect, "Am")
makeplots(S_zoom, "Am", "-zoom")
