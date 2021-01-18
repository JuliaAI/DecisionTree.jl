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

function testDataset((name,dataset), timeit::Bool = true)
	println("Testing dataset '$name'")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	length(dataset) == 4 || error(length(dataset))
	X_train, Y_train, X_test, Y_test = dataset
	if timeit
		@btime build_tree($Y_train, $X_train; ontology = ModalLogic.IntervalOntology);
	end

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

testDatasets(d, timeit::Bool = true) = map((x)->testDataset(x, timeit), d);


datasets = Tuple{String,Tuple{Array,Array,Array,Array}}[
	("simpleDataset",traintestsplit(simpleDataset(200,50)...,0.8)),
	("Eduard-5",EduardDataset(5)),
	("Eduard-10",EduardDataset(10)),
];

testDatasets(datasets);

# T = testDataset(datasets[1])
# T = testDataset(datasets[2])
# T = testDataset(datasets[3])

#=

Without Xf(... best feature)
Testing dataset 'simpleDataset'
  146.995 ms (2251482 allocations: 194.09 MiB)
  Accuracy: 100% baby!
Testing dataset 'Eduard-5'
  6.307 s (39708278 allocations: 1.50 GiB)
  Accuracy: 91.11%
Testing dataset 'Eduard-10'
  12.520 s (102004178 allocations: 3.93 GiB)
  Accuracy: 84.44%

using Views with smarter Xf:
Testing dataset 'simpleDataset'
  150.841 ms (2251328 allocations: 194.08 MiB)
  Accuracy: 100% baby!
Testing dataset 'Eduard-5'
  4.617 s (39847067 allocations: 1.50 GiB)
  Accuracy: 90.0%
Testing dataset 'Eduard-10'
  12.352 s (102190347 allocations: 3.94 GiB)
  Accuracy: 84.44%

using Views:
Testing dataset 'simpleDataset'
  129.927 ms (2251644 allocations: 194.31 MiB)
  Accuracy: 100% baby!
Testing dataset 'Eduard-5'
  5.084 s (39690013 allocations: 1.50 GiB)
  Accuracy: 91.11%
Testing dataset 'Eduard-10'
  11.421 s (102032965 allocations: 3.94 GiB)
  Accuracy: 84.44%

not using Views:
Testing dataset 'simpleDataset'
  151.523 ms (2251644 allocations: 194.31 MiB)
  Accuracy: 100% baby!
Testing dataset 'Eduard-5'
  5.356 s (39781460 allocations: 1.50 GiB)
  Accuracy: 90.0%
Testing dataset 'Eduard-10'
  16.321 s (101172782 allocations: 3.90 GiB)
  Accuracy: 85.56%

=#
 

