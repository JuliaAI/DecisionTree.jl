julia

using Revise
using Pkg
Pkg.activate("DecisionTree.jl")

n_samp = 50
N = 3

X = Array{Int,3}(undef, n_samp, 1, N);
Y = Array{Int,1}(undef, n_samp);
for i in 1:n_samp
	instance = fill(2, 3)
	y = rand(0:1)
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

using Logging

import Random
using BenchmarkTools
using DecisionTree
using DecisionTree.ModalLogic

global_logger(ConsoleLogger(stderr, Logging.Warn))
# global_logger(ConsoleLogger(stderr, Logging.Info))

# @btime T = DecisionTree.treeclassifier.fit(
# 	X = OntologicalDataset{Int,1}(IntervalAlgebra,X_train),
# 	Y = Y_train,
# 	W = nothing,
# 	loss = DecisionTree.util.entropy,
# 	max_features = 1,
# 	max_depth = 2,
# 	min_samples_leaf = 1,
# 	min_samples_split = 2,
# 	min_purity_increase = 0.0,
# 	rng = Random.GLOBAL_RNG)
# T2 = DecisionTree._convert(T.root, T.list, Y[T.labels])
# DecisionTree.print_tree(T2)

# @btime
T2 = build_tree(Y_train, X_train) # 122.872 μs (1472 allocations: 85.31 KiB)

preds = apply_tree(T2, X_test)

if Y_test == preds
	print("Yeah!")
else
	@error "Predictions don't match expected values" Y_test preds
end
