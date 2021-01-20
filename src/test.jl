julia

Pkg.activate("DecisionTree.jl")
using Revise
using Pkg

import Random
# Random.GLOBAL_RNG
my_rng() = Random.MersenneTwister(1)

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
		@btime build_tree($Y_train, $X_train; ontology = ModalLogic.IntervalOntology, rng = my_rng());
	end

	# global_logger(ConsoleLogger(stderr, Logging.Info))
	T = build_tree(Y_train, X_train; rng = my_rng());

	preds = apply_tree(T, X_test);

	if Y_test == preds
		println("  Accuracy: 100% baby!")
	else
		println("  Accuracy: ", round((sum(Y_test .== preds)/length(preds))*100, digits=2), "%")
	end;

	global_logger(ConsoleLogger(stderr, Logging.Info));

	if timeit
		println("  Nodes: $(num_nodes(T))")
		println("  Height: $(height(T))")
	else
		print(T)
	end
end

testDatasets(d, timeit::Bool = true) = map((x)->testDataset(x, timeit), d);

datasets = Tuple{String,Tuple{Array,Array,Array,Array}}[
	("simpleDataset",traintestsplit(simpleDataset(200,50,my_rng())...,0.8)),
	("simpleDataset2",traintestsplit(simpleDataset2(200,5,my_rng())...,0.8)),
	("Eduard-5",EduardDataset(5)),
	("Eduard-10",EduardDataset(10)),
	# ("PaviaDataset",traintestsplit(SampleLandCoverDataset(100, 3, "Pavia", my_rng())...,0.8)),
	# ("IndianPinesCorrectedDataset",traintestsplit(SampleLandCoverDataset(100, 3, "IndianPinesCorrected", my_rng())...,0.8)),
];

testDatasets(datasets);

# T = testDataset(datasets[1], false)

# X = ModalLogic.OntologicalDataset{Int64,0}(ModalLogic.IntervalOntology,Array{Int64,2}(undef, 20, 10))
# X = ModalLogic.OntologicalDataset{Float32,0}(ModalLogic.IntervalOntology,rand(Float32, 20, 10))
# XX = ModalLogic.OntologicalDataset{Int64,1}(ModalLogic.IntervalOntology,X_train)

#=

Testing dataset 'simpleDataset'
  120.604 ms (2251653 allocations: 194.33 MiB)
  Accuracy: 100% baby!
  Nodes: 3
  Height: 1
Testing dataset 'simpleDataset2'
  851.795 μs (20017 allocations: 1.15 MiB)
  Accuracy: 30.0%
  Nodes: 1
  Height: 0
Testing dataset 'Eduard-5'
  4.502 s (39457913 allocations: 1.49 GiB)
  Accuracy: 88.89%
  Nodes: 167
  Height: 10
Testing dataset 'Eduard-10'
  11.129 s (102020927 allocations: 3.93 GiB)
  Accuracy: 84.44%
  Nodes: 221
  Height: 10

=#
 

