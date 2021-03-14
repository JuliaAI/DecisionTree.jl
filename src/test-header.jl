using Pkg
Pkg.activate("..")
using Revise

using DecisionTree
using DecisionTree.ModalLogic

import Random
my_rng() = Random.MersenneTwister(1) # Random.GLOBAL_RNG

using Logging
using IterTools

using BenchmarkTools
using ScikitLearnBase
using Statistics
using Test
using Profile
using PProf

include("example-datasets.jl")

function testDataset((name,dataset), timeit::Integer = 2; debugging_level = DecisionTree.DTOverview, post_pruning_purity_thresholds = [], args = (), kwargs = ())
	println("Benchmarking dataset '$name'...")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	length(dataset) == 4 || error(length(dataset))
	X_train, Y_train, X_test, Y_test = dataset
	
	println("args = ", args)
	println("kwargs = ", kwargs)

	# println(" n_samples = $(size(X_train)[end-1])")
	println(" train size = $(size(X_train))")
	# global_logger(ConsoleLogger(stderr, Logging.Info))
	global_logger(ConsoleLogger(stderr, debugging_level))
	# global_logger(ConsoleLogger(stderr, DecisionTree.DTDebug))
	if timeit == 0
		T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());
	elseif timeit == 1
		@time T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());
	elseif timeit == 2
		@btime build_tree($Y_train, $X_train, args...; kwargs..., rng = my_rng());
		T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());
	end

	# if timeit != 2
	# 	print(T)
	# end

	println(" test size = $(size(X_test))")
	for pruning_purity_threshold in sort(unique([(Float64.(post_pruning_purity_thresholds))...,1.0]))
		println(" Purity threshold $pruning_purity_threshold")
		
		global_logger(ConsoleLogger(stderr, Logging.Warn));

		T_pruned = prune_tree(T, pruning_purity_threshold)
		preds = apply_tree(T_pruned, X_test);
		cm = confusion_matrix(Y_test, preds)
		# @test cm.accuracy > 0.99
		println("  acc: ", round(cm.accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
		for (i,row) in enumerate(eachrow(cm.matrix))
			for val in row
				print(lpad(val,3," "))
			end
			println("  " * "$(round(100*row[i]/sum(row), digits=2))%")
		end

		global_logger(ConsoleLogger(stderr, Logging.Info));

		# println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
	end
	return T;
end