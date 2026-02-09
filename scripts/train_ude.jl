using CSV, DataFrames, ComponentArrays, Optimization, OptimizationOptimJL, OptimizationOptimisers, DifferentialEquations, Lux, SciMLSensitivity, Random, Zygote, Plots

# Ensure paths are relative to the script location
script_dir = @__DIR__
include(joinpath(script_dir, "../src/models/physics.jl"))
include(joinpath(script_dir, "../src/models/nural.jl"))
include(joinpath(script_dir, "../src/training.jl"))

# Load data - Using joinpath to prevent UndefVarError if path is wrong
data_path = joinpath(script_dir, "../data/malware_data.csv")
data = CSV.read(data_path, DataFrame)

t_steps = Float32.(data.timestamp)
observed_I = Float32.(data.infected_noisy)

nn = create_nn()
rng = Random.default_rng()
ps, st = Lux.setup(rng, nn)
ps = ComponentVector(ps) .* 0.01f0

u0 = Float32[0.99, 0.01, 0.0]
p_init = ComponentVector(physics=Float32[0.5, 0.15], neural=ps)

callback = function (p, l)
    println("Current loss: $l")
    return false
end

# Integrated solver logic to resolve scope issues
function predict_at_times(u0, p, tspan, nn, st, t_save)
    prob = ODEProblem((du, u, p, t) -> ude_model!(du, u, p, t, nn, st), u0, tspan, p)
    sol = solve(prob, AutoTsit5(Rosenbrock23()),
        saveat=t_save,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        abstol=1e-6, reltol=1e-6,
        isoutofdomain=(u, p, t) -> any(x -> x < 0.0, u),
        verbose=false)
    return sol
end

function loss(p, tspan_current)
    sol = predict_at_times(u0, p, tspan_current, nn, st, t_steps)

    if sol.retcode != ReturnCode.Success || size(sol, 2) != length(t_steps)
        return 1.0f7
    end

    pred_I = sol[2, :]
    neg_penalty = sum(abs2, min.(0.0f0, sol))

    return sum(abs2, pred_I .- observed_I) + 0.01f0 * sum(abs2, p.neural) + 10.0f0 * neg_penalty
end

println("Short-term training (t=0 to 10)")
tspan_short = (0.0f0, 10.0f0)
optf = OptimizationFunction((p, _) -> loss(p, tspan_short), Optimization.AutoZygote())
optprob = OptimizationProblem(optf, p_init)
res_short = solve(optprob, OptimizationOptimisers.Adam(1.0f-3), callback=callback, maxiters=150)

println("Full-term training (t=0 to 50)")
tspan_full = (0.0f0, 50.0f0)
optf_full = OptimizationFunction((p, _) -> loss(p, tspan_full), Optimization.AutoZygote())
optprob_full = remake(optprob, f=optf_full, u0=res_short.u)
res_adam = solve(optprob_full, OptimizationOptimisers.Adam(5.0f-4), callback=callback, maxiters=200)

println("BFGS Refinement")
optprob_bfgs = remake(optprob_full, u0=res_adam.u)
result = solve(optprob_bfgs, OptimizationOptimJL.BFGS(initial_stepnorm=0.01), callback=callback, maxiters=100)

p_trained = result.u
final_sol = predict_at_times(u0, p_trained, tspan_full, nn, st, t_steps)
p_plot = plot(t_steps, observed_I, seriestype=:scatter, label="Data", title="Malware UDE Refined")
plot!(final_sol.t, final_sol[2, :], label="UDE Prediction", lw=3, color=:red)
savefig(joinpath(script_dir, "../plots/ude_results.png"))
