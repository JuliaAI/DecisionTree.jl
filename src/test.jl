julia

Pkg.activate("DecisionTree.jl")
using Revise
using Pkg

import Random
# Random.GLOBAL_RNG
my_rng = Random.MersenneTwister(1)

using Logging

using BenchmarkTools
using DecisionTree
using DecisionTree.ModalLogic

n_samp = 50
N = 3

X = Array{Int,3}(undef, n_samp, 1, N);
Y = Array{Int,1}(undef, n_samp);
for i in 1:n_samp
	instance = fill(2, 3)
	y = rand(my_rng, 0:1)
	if y == 0
		instance[3] = 1
	else
		instance[3] = 2
	end
	X[i,1,:] .= instance
 Y[i] = y
end

spl = floor(Int, n_samp*.8)
X_train = X[1:spl,:,:]
Y_train = Y[1:spl]
X_test  = X[spl+1:end,:,:]
Y_test  = Y[spl+1:end]



global_logger(ConsoleLogger(stderr, Logging.Warn))
# global_logger(ConsoleLogger(stderr, Logging.Info))

# 484.136 μs (4368 allocations: 256.34 KiB)
# 517.471 μs (4368 allocations: 256.38 KiB)
@btime T2 = build_tree(Y_train, X_train; ontology = ModalLogic.IntervalOntology, rng = my_rng)

T2 = build_tree(Y_train, X_train; rng = my_rng)

preds = apply_tree(T2, X_test)

if Y_test == preds
	print("Yeah!")
else
	@error "Predictions don't match expected values" Y_test preds
end
