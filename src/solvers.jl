
function predict(u0, p, tspan, nn, st, t_save)
    prob = ODEProblem((du, u, p, t) -> ude_model!(du, u, p, t, nn, st), u0, tspan, p)

    sol = solve(
        prob,
        AutoTsit5(Rosenbrock23()),
        saveat=t_save,
        sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
        abstol=1e-6,
        reltol=1e-6,
        isoutofdomain=(u, p, t) -> any(x -> x < 0.0, u),
        verbose=false
    )

    return sol
end
