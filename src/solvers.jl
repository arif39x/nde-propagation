using DifferentialEquations #To solve the calculas it gives the algorythm like Tsit5 that step through time to see hoe malware spreads

function predict(u0, p, tspan, nn, st)    #u0 initial condition, p combined parametes(Phy+neuralnet) tspan the time window
    prob = ODEProblem((du, u, p, t) -> ude_model!(du, u, p, t, nn, st), u0, tspan, p) #The one of the most intersting logic that i wrote ok let me explain
    # it creates a hybrid math model ehre the system changes over time is defined by combining physics laws and the neural network's real time prediction....
    return solve(prob, AutoTsit5(Rosenbrock23()), saveat=0.5, reltol=1e-6, abstol=1e-6)
end
