using CSV, DataFrames, ComponentArrays, Optimization, OptimizationOptimJL, OptimizationOptimisers, DifferentialEquations, Lux, SciMLSensitivity, Random, Zygote, Plots

include("../src/models/physics.jl")
include("../src/models/nural.jl")
include("../src/solvers.jl")
include("../src/training.jl")

data = CSV.read("data/malware_data.csv", DataFrame)
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

function predict_at_times(u0, p, tspan, nn, st, t_save)
    prob = ODEProblem((du, u, p, t) -> ude_model!(du, u, p, t, nn, st), u0, tspan, p)
    sol = solve(prob, AutoTsit5(Rosenbrock23()),
        saveat=t_save,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        abstol=1e-6, reltol=1e-6)
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
savefig("plots/ude_results.png")

function ude_model!(du, u, p, t, nn, st)    ##nn the lux neural net structure and st the lux model (generally empty for simple layer)
    S, I, R = u

    nn_val, _ = nn([I], p.neural, st)    ##hetwork looks at the current number of infected machine which is I and passess this through the weight stored in p.neural

    correction = tanh(nn_val[1])  ## nOw this will prevents the nn from suggesting a billion percent infection rate


    infection_term = p.physics[1] * S * I + nn_val[1]    #p.physics[1] * S * I  is infection term suspected node ,infected node ...beta scales contagious the diseas is

    du[1] = -infection_term
    du[2] = infection_term - p.physics[2] * I
    du[3] = p.physics[2] * I
end
