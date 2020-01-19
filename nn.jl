using Random 
using LinearAlgebra
using Statistics
using Plots 
using DelimitedFiles
pyplot()

# helper functions
function sigmoid(X)
    sigma = 1 ./ (1 .+ exp.(.-X))
    return sigma, X
end

function relu(X)
    rel = max.(0, X)
    return rel, X
end

# initialise parameters
function init_params(layer_dim)
    param = Dict()

    for l = 1:length(layer_dim)-1
        param[string("W_", string(l))] = rand(layer_dim[l+1], layer_dim[l]) * 0.1
        param[string("b_", string(l))] = zeros(layer_dim[l+1], 1)
    end

    return  param
end

# Forward propogation
function forward(A, w, b)

    Z = w*A .+ b
    cache = (A, w, b)

    return Z, cache
    
end

function calc_act_forward(A_pre, W, b, func_type)
    if (func_type == "sigmoid")
        Z, linear_step_cache = forward(A_pre, W, b)
        A, activation_step_cache = sigmoid(Z)
    elseif (func_type == "relu")
        Z, linear_step_cache = forward(A_pre, W, b)
        A, activation_step_cache = relu(Z)
    end

    cache = (linear_step_cache, activation_step_cache)
    return A, cache
    
end

function model_forward_step(X, params)
    all_caches = []
    A = X
    L = length(params)/2

    for l=1:L-1
        A_pre = A
        A, cache = calc_act_forward(
            A_pre, 
            params[string("W_" , string(Int(l)))], 
            params[string("b_" , string(Int(l)))],
            "relu"
            )
        push!(all_caches , cache)
    end

    A_1, cache = calc_act_forward(
        A,
        params[string("W_" , string(Int(L)))],
        params[string("b_" , string(Int(L)))], 
        "sigmoid"
    )

    push!(all_caches, cache)

    return A_1, all_caches
    
end

# Cost: Binary Cross-Entropy
function cost_function(AL, Y)
    cost = -mean(Y.*log(AL) + (1 .- Y).*log(1 .- AL))
    
    return cost
    
end

# Back propogation
function backward_linear_step(dZ, cache)
    A_prev, W, b = cache

    m = size(A_prev)[2]

    dW = dZ * (A_prev')/m
    db = sum(dZ, dims = 2)/m
    dA_prev = (W')* dZ
    
    return dW, db, dA_prev
    
end

function backward_relu(dA, cache_activation)
    return dA.*(cache_activation.>0)
end

function backward_sigmoid(dA, cache_activation)
    return dA.*(sigmoid(cache_activation)[1].*(1 .- sigmoid(cache_activation)[1]))    
end

function backward_activation_step(dA, cache, activation)
    linear_cache, cache_activation = cache

    if (activation == "relu")
        dZ = backward_relu(dA, cache_activation)
        dW, db, dA_prev = backward_linear_step(dZ, linear_cache)    
    elseif (activation = "sigmoid")
        dZ = backward_sigmoid(dA, cache_activation)
        dW, db, dA_prev = backward_linear_step(dZ, linear_cache)
    end
    
    return dW, db, dA_prev
    
end

function model_backward_step(A_1, Y, caches)
    grads = Dict()
    L = length(caches)
    m = size(A_1)[2]
    
    Y = reshape(Y, size(A_1))
    dA_1 = (-(Y./A_1) .+ ((1 .- Y) ./ (1 .- A_1)))
    current_cache = caches[L]
    grads[string("dW_", string(L))], grads[string("db_" , string(L))], grads[string("dA_" , string(L-1))] = backward_activation_step(dA_1, current_cache, "sigmoid")
    
    for l=reverse(0:L-2)
        current_cache = caches[l+1]
        grads[string("dW_", string(l+1))], grads[string("db_" , string(l+1))], grads[string("dA_" , string(l-1))] = backward_activation_step(dA_1, current_cache, "sigmoid")
    end

    return grads

end

function update_params(parameters, grads, learning_rate)
    L = Int(length(parameters)/2)

    for l=0:(L-1)
        parameters[string("W_" , string(l+1))] -= learning_rate.*grads[string("dW_" , string(l+1))]
        parameters[string("b_",string(l+1))] -= learning_rate.*grads[string("db_",string(l+1))]
    end

    return parameters

end

function check_accuracy(A_L , Y)
    A_L = reshape(A_L , size(Y))
    return sum((A_L.>0.5) .== Y)/length(Y)
end 

function train_nn(layers_dimensions , X , Y , learning_rate , n_iter)

    params = init_params(layers_dimensions)
    costs = []
    iters = []
    accuracy = []
    for i=1:n_iter
        A_l , caches  = model_forward_step(X , params)
        cost = cost_function(A_l , Y)
        acc = check_accuracy(A_l , Y)
        grads  = model_backwards_step(A_l , Y , caches)
        params = update_param(params , grads , learning_rate)
        println("Iteration ->" , i)
        println("Cost ->" , cost)
        println("Accuracy -> " , acc)
        push!(iters , i)
        push!(costs , cost)
        push!(accuracy , acc)
        
    end 
    plt = plot(iters , costs ,title =  "Cost Function vs Number of Iterations" , lab ="J")
    xaxis!("N_Iterations")
    yaxis!("J")
    plt_2 = plot(iters , accuracy ,title =  "Accuracy vs Number of Iterations" , lab ="Acc" , color = :green)
    xaxis!("N_Iterations")
    yaxis!("Accuracy")
    plot(plt , plt_2 , layout = (2,1))
    savefig("cost_plot_rand.pdf")
    return params , costs 

end

X = collect(1:10)
Y = vcat(zeros(5), zeros(5) .+ 1)

using Debugger

# @enter train_nn(10 , X , Y , 0.01 , 100)
train_nn([10, 10, 1] , X , Y , 0.01 , 100)
