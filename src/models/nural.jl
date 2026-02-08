using Lux, Random

function create_nn()     # Macro Stck layer
    nn = Chain(
        Dense(1 => 16, tanh), # it takes 1 features and expands to 16 neurons
        Dense(16 => 16, tanh),# a hidden layer to add ddepth for learning commplex non-linear patterns
        Dense(16 => 1)         ##Compress back the 16 features back to 1 value
    )
    return nn
end
