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

include("example-datasets.jl")


function testDataset((name,dataset))
	println("Testing dataset '$name'")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	length(dataset) == 4 || error(length(dataset))
	X_train, Y_train, X_test, Y_test = dataset
	@btime build_tree($Y_train, $X_train; ontology = ModalLogic.IntervalOntology);

	# global_logger(ConsoleLogger(stderr, Logging.Info))
	T = build_tree(Y_train, X_train; rng = my_rng);

	preds = apply_tree(T, X_test);

	if Y_test == preds
		println("  Accuracy: 100% baby!")
	else
		println("  Accuracy: ", round((sum(Y_test .== preds)/length(preds))*100, digits=2), "%")
	end;

	global_logger(ConsoleLogger(stderr, Logging.Info));

	T;
end

datasets = Tuple{String,Tuple{Array,Array,Array,Array}}[
	("simpleDataset",traintestsplit(simpleDataset(200,50)...,0.8)),
	("Eduard-5",EduardDataset(5)),
	("Eduard-10",EduardDataset(10)),
];

# for d in datasets
	# testDataset(d)
# end

T = testDataset(datasets[2])
#=

Testing dataset 'simpleDataset'
  121.186 ms (2252270 allocations: 194.33 MiB)
	Accuracy: 100% baby!
Testing dataset 'Eduard-5'
	1.366 s (16365958 allocations: 642.61 MiB)
	Accuracy: 84.44%
Testing dataset 'Eduard-10'
	6.415 s (69301692 allocations: 2.67 GiB)
	Accuracy: 84.44%

=#
