function baseline_malware!(du, u, p, t)
    S, I, R = u
    β, γ = p

    du[1] = -β * S * I       ##The number of health machine alwas decreases
    du[2] = β * S * I - γ * I ##The infection engine
    du[3] = γ * I             ##Number of recovered machine always increase
end
