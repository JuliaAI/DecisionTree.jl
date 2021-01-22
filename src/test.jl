julia

using Pkg
Pkg.activate("DecisionTree.jl")
using Revise

import Random
my_rng() = Random.MersenneTwister(1) # Random.GLOBAL_RNG

using Logging

using BenchmarkTools
using DecisionTree
using DecisionTree.ModalLogic
using ScikitLearnBase
using Statistics
using Test

include("example-datasets.jl")

function testDataset((name,dataset), timeit::Bool = true; pruning_purity_thresholds = [0.7,0.8,0.9])
	println("Testing dataset '$name'")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	length(dataset) == 4 || error(length(dataset))
	X_train, Y_train, X_test, Y_test = dataset
	if timeit
		@btime build_tree($Y_train, $X_train; ontology = $(ModalLogic.getIntervalOntologyOfDim(X_train)), rng = my_rng());
	end

	# global_logger(ConsoleLogger(stderr, Logging.Info))
	T = build_tree(Y_train, X_train; ontology = ModalLogic.getIntervalOntologyOfDim(X_train), rng = my_rng());

	if !timeit
		print(T)
	end

	for pruning_purity_threshold in sort(unique([(Float64.(pruning_purity_thresholds))...,1.0]))
		println(" Purity threshold $pruning_purity_threshold")
		
		global_logger(ConsoleLogger(stderr, Logging.Warn));

		T_pruned = prune_tree(T, pruning_purity_threshold)
		preds = apply_tree(T_pruned, X_test);
		cm = confusion_matrix(Y_test, preds)
		# @test cm.accuracy > 0.99

		if Y_test == preds
			println("  Accuracy: 100% baby!")
		else
			println("  Accuracy: ", round(cm.accuracy*100, digits=2), "%")
			println("  Kappa: ", round(cm.kappa*100, digits=2), "%")
			println("  Matrix: ")
			display(cm.matrix)
		end;

		global_logger(ConsoleLogger(stderr, Logging.Info));

		if timeit
			println("  (nodes,height): ($(num_nodes(T_pruned)),$(height(T_pruned)))")
		end
	end

	# model = fit!(DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold), X_train, Y_train)
	# cm = confusion_matrix(Y_test, predict(model, X_test))
	# @test cm.accuracy > 0.99

	T
end

testDatasets(d, timeit::Bool = true) = map((x)->testDataset(x, timeit), d);

datasets = Tuple{String,Tuple{Array,Array,Array,Array}}[
	("simpleDataset",traintestsplit(simpleDataset(200,50,my_rng())...,0.8)),
	("simpleDataset2",traintestsplit(simpleDataset2(200,5,my_rng())...,0.8)),
	("Eduard-5",EduardDataset(5)),
	("Eduard-10",EduardDataset(10)),
	# ("PaviaDataset",traintestsplit(SampleLandCoverDataset(100, 3, "Pavia", 5, my_rng())...,0.8)),
	# ("IndianPinesCorrectedDataset",traintestsplit(SampleLandCoverDataset(100, 3, "IndianPinesCorrected", 5, my_rng())...,0.8)),
	# ("PaviaDataset",traintestsplit(SampleLandCoverDataset(100, 5, "Pavia", 5, my_rng())...,0.8)),
	# ("PaviaDataset",traintestsplit(SampleLandCoverDataset(100, 7, "Pavia", 5, my_rng())...,0.8)),
];

# T = testDataset(datasets[1], false)
# T = testDataset(datasets[2], false)
testDatasets(datasets);


