using SciMLSensitivity, Lux, ComponentArrays    #SciMlSensitivity is a library that allows to calculate gradient through the ODE  and the ComponentArrays are container utility...to treat a giant vector of parameters as a structured objeets...

function ude_model!(du, u, p, t, nn, st)    ##nn the lux neural net structure and st the lux model (generally empty for simple layer)
    S, I, R = u

    nn_guess, _ = nn([I], p.neural, st)    ##hetwork looks at the current number of infected machine which is I and passess this through the weight stored in p.neural

    correction = tanh(nn_val[1])  ## nOw this will prevents the nn from suf=ggesting a billion percent infection rate


    infection_term = p.physics[1] * S * I + nn_guess[1]    #p.physics[1] * S * I  is infection term suspected node ,infected node ...beta scales contagious the diseas is

    du[1] = -infection_term
    du[2] = infection_term - p.physics[2] * I
    du[3] = p.physics[2] * I
end
