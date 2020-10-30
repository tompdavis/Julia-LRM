include("byrnegreenwood.jl")
using Plots

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

function make_plots(T, σ, r, K, S, N, ϵ, ex = "Eur", type = "Vanilla", zoom=true)


    if type == "Vanilla"
        payoff = x -> max(0, K - x)
        if ex == "Eur"
            bound_bs = x -> bs_option(x, K, T, σ, r, -1.0)
            bound_crr = x -> crr_eur(x, K, r, σ, T, N, payoff)
            suffix = "VanillaEur"
        elseif ex == "Am"
            bound_crr = x -> crr_am(x, K, r, σ, T, N, payoff)
            suffix = "VanillaAm"
        end
    elseif type == "Digital"
        payoff = x -> heaviside(K - x)
        if ex == "Eur"
            bound_bs = x -> bs_digital(x, K, T, σ, r, -1.0)
            bound_crr = x -> crr_eur(x, K, r, σ, T, N, payoff)
            suffix = "DigitalEur"
        elseif ex == "Am"
            bound_crr = x -> crr_am(x, K, r, σ, T, N, payoff)
            suffix = "DigitalAm"
        end
    end

    if zoom == true
        Svect = 95.0:0.05:105.0
        suffix = string(suffix, "-zoom")
    else
        Svect = [S*exp(z*sqrt(V)) for z in -2.5:0.10:2.5]
    end

    C_at_S = Array{Float64}(undef, length(Svect))
    # dual_delt_at_S = Array{Float64}(undef, length(Svect))
    lrm_delt_at_S = Array{Float64}(undef, length(Svect))
    price_on_tree = Array{Float64}(undef, length(Svect))
    eur_delta_at_S = Array{Float64}(undef, length(Svect))
    eur_gamma_at_S = Array{Float64}(undef, length(Svect))
    lrm_bump_gamma = Array{Float64}(undef, length(Svect))
    # lrm_dual_gamma = Array{Float64}(undef, length(Svect))
    bump_delta = Array{Float64}(undef, length(Svect))
    bump_gamma = Array{Float64}(undef, length(Svect))
    # dual_gamma = Array{Float64}(undef, length(Svect))

    for n in 1:length(Svect)
        # Szd = Dual(Svect[n], 1)
        if ex == "Eur"
            eur_cf = bound_bs(Svect[n])
            C_at_S[n] = eur_cf[1]
            eur_delta_at_S[n] = eur_cf[2]
            eur_gamma_at_S[n] = eur_cf[3]
        end
        bdg = bump_dg(bound_crr, Svect[n], ϵ)
        price_on_tree[n] = bdg[1][1]
        lrm_delt_at_S[n] = bdg[1][2]
        # dual_delt_at_S[n] = bdg[1][1].partials[1]
        lrm_bump_gamma[n] = bdg[2][2]
        # lrm_dual_gamma[n] =
        bump_delta[n] = bdg[2][1]
        bump_gamma[n] = bdg[3][1]
        # dual_gamma[n] = bdg[2][1].partials[1]
    end

    if ex == "Am"
        to_plot = price_on_tree
       labels = "CRR"
   elseif ex == "Eur"
       to_plot = [C_at_S,
                   price_on_tree]
       labels=["Closed Form" "CRR"]
   end
   lw = 3
    Plots.plot(Svect, to_plot,
                       xlabel="S",
                       ylabel="C",
                       labels=labels,
                       linewidth=lw)
    fig_name = string("Price-", suffix, ".png")
    Plots.savefig(string(folder, "\\", fig_name))

    if ex == "Am"
        to_plot = [lrm_delt_at_S, bump_delta]
        labels = ["LRM" "Bump"]
    elseif ex == "Eur"
        if type == "Vanilla"
            to_plot = [eur_delta_at_S, lrm_delt_at_S, bump_delta]
            labels=["Closed Form" "LRM" "Bump"]
        elseif type == "Digital"
            to_plot = [eur_delta_at_S, lrm_delt_at_S]
            labels=["Closed Form" "LRM"]
        end
    end
    Plots.plot(Svect, to_plot,
                       xlabel = "S",
                       ylabel = "Δ",
                       linewidth=lw,
                       labels = labels)
    fig_name = string("Delta-", suffix, ".png")
    Plots.savefig(string(folder, "\\", fig_name))
    if ex == "Eur"
        if type == "Vanilla"
            to_plot = 100*100*[(eur_delta_at_S - lrm_delt_at_S)./eur_delta_at_S,
                                (eur_delta_at_S - bump_delta)./eur_delta_at_S]
            labels = ["LRM" "Bump"]
        elseif type == "Digital"
            to_plot = 100*100*[(eur_delta_at_S - lrm_delt_at_S)./eur_delta_at_S]
            labels = "LRM"
        end
        Plots.plot(Svect, to_plot ,
                           xlabel="S",
                           ylabel="bps",
                           labels=labels,
                           linewidth=lw)
        fig_name = string("DeltaRel-", suffix, ".png")
        Plots.savefig(string(folder, "\\", fig_name))
    end

    if ex == "Am"
        to_plot = lrm_bump_gamma
        labels = "LRM + Bump"
    elseif ex == "Eur"
        to_plot = [eur_gamma_at_S, lrm_bump_gamma]
        labels=["Closed Form" "LRM + Bump"]
    end

    Plots.plot(Svect, to_plot,
                       xlabel="S",
                       ylabel="Γ",
                       labels=labels,
                       linewidth=lw)
    fig_name = string("Gamma-", suffix, ".png")
    Plots.savefig(string(folder, "\\", fig_name))

    if ex == "Eur" && type == "Vanilla"
        Plots.plot(Svect, 100*100*[(lrm_bump_gamma - eur_gamma_at_S)./eur_gamma_at_S,
                                    (eur_delta_at_S - lrm_delt_at_S)./eur_delta_at_S],
                          xlabel="S",
                          ylabel="bps",
                          labels=["Γ Relative Error" "Δ Relative Error"])
        fig_name = string("GreeksRel-", suffix, ".png")
        Plots.savefig(string(folder, "\\", fig_name))

        Plots.plot(Svect, [(C_at_S - price_on_tree)./C_at_S,
                           (eur_delta_at_S - lrm_delt_at_S)./eur_delta_at_S]*100*100,
                           labels=["Price Relative Error" "Δ Relative Error"],
                           xlabel="S",
                           ylabel="bps",
                           linewidth=lw)
        fig_name = string("PriceRel-", suffix, ".png")
        Plots.savefig(string(folder, "\\", fig_name))
    end
end

T = 180/365
σ = 0.3
V = σ^2*T
r = 0.05
K = 90.0
S = 100
N = 1000
ϵ = 0.0001
# #
make_plots(T, σ, r, K, S, N, ϵ, "Am", "Vanilla", false)
make_plots(T, σ, r, K, S, N, ϵ, "Am", "Vanilla", true)
make_plots(T, σ, r, K, S, N, ϵ, "Eur", "Vanilla", true)
make_plots(T, σ, r, K, S, N, ϵ, "Eur", "Vanilla", false)
make_plots(T, σ, r, K, S, N, ϵ, "Eur", "Digital", false)
make_plots(T, σ, r, K, S, N, ϵ, "Eur", "Digital", true)
make_plots(T, σ, r, K, S, N, ϵ, "Am", "Digital", true)
make_plots(T, σ, r, K, S, N, ϵ, "Am", "Digital", false)

