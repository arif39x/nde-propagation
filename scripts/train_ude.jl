using CSV, DataFrames, ComponentArrays, Optimization, OptimizationOptimJL, OptimizationOptimisers, DifferentialEquations, Lux, SciMLSensitivity, Random, Zygote, Plots

include("../src/models/physics.jl")
include("../src/models/nural.jl")
include("../src/solvers.jl")
include("../src/training.jl")

data = CSV.read("data/malware_data.csv", DataFrame)
t_steps = data.timestamp
observed_I = data.infected_noisy

nn = create_nn()
rng = Random.default_rng()
ps, st = Lux.setup(rng, nn)

u0 = Float32[0.99, 0.01, 0.0]
tspan = (0.0f0, 50.0f0)

p_init = ComponentVector(
    physics=Float32[0.5, 0.15],
    neural=ps
)

function loss(p, _)
    sol = predict(u0, p, tspan, nn, st)

    if sol.retcode != ReturnCode.Success
        return 1e7f0  # if the sover falles it return high penalty not inf and keep the gradient defined
    end

    pred_I = sol[2, :]

    data_loss = sum(abs2, pred_I .- observed_I) #meansquare error standard

    reg_loss = 0.01f0 * sum(abs2, p.neural)

    return data_loss + reg_loss
end

adtype = Optimization.AutoZygote()
optf = OptimizationFunction(loss, adtype)
optprob = OptimizationProblem(optf, p_init)

println("Stabilizing with Adam arounf 200 iterations...")
res_adam = solve(optprob, OptimizationOptimisers.Adam(0.005f0), maxiters=200)

println("Refining with BFGS...")
optprob_bfgs = remake(optprob, u0=res_adam.u)
result = solve(optprob_bfgs, BFGS(), maxiters=100)

p_trained = result.u
final_sol = predict(u0, p_trained, tspan, nn, st)

p_plot = plot(t_steps, observed_I, seriestype=:scatter, label="Noisy Data (Input)", mc=:black, ms=3, ma=0.4)
plot!(final_sol.t, final_sol[2, :], label="UDE Learned Curve", lw=3, lc=:red)
title!("Malware Dynamics")
xlabel!("Time")
ylabel!("Infected Ratio")

savefig(p_plot, "plots/ude_results.png")
println("Training complete.")
