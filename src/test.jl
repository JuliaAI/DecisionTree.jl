# julia-1.5.3
# julia






using Pkg
Pkg.activate("..")
using Revise

using DecisionTree
using DecisionTree.ModalLogic

import Random
my_rng() = Random.MersenneTwister(1) # Random.GLOBAL_RNG

using Logging

using BenchmarkTools
using ScikitLearnBase
using Statistics
using Test
using Profile
using PProf

include("example-datasets.jl")

function testDataset((name,dataset), timeit::Integer = 2; post_pruning_purity_thresholds = [], args = (), kwargs = ())
	println("Benchmarking dataset '$name'...")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	length(dataset) == 4 || error(length(dataset))
	X_train, Y_train, X_test, Y_test = dataset
	
	println("args = ", args)
	println("kwargs = ", kwargs)

	# println(" n_samples = $(size(X_train)[end-1])")
	println(" train size = $(size(X_train))")
	# global_logger(ConsoleLogger(stderr, Logging.Info))
	if timeit == 0
		T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());
	elseif timeit == 1
		@time T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());
	elseif timeit == 2
		@btime build_tree($Y_train, $X_train, args...; kwargs..., rng = my_rng());
		T = build_tree(Y_train, X_train, args...; kwargs..., rng = my_rng());
	end

	if timeit != 2
		print(T)
	end

	println(" test size = $(size(X_test))")
	for pruning_purity_threshold in sort(unique([(Float64.(post_pruning_purity_thresholds))...,1.0]))
		println(" Purity threshold $pruning_purity_threshold")
		
		global_logger(ConsoleLogger(stderr, Logging.Warn));

		T_pruned = prune_tree(T, pruning_purity_threshold)
		preds = apply_tree(T_pruned, X_test);
		cm = confusion_matrix(Y_test, preds)
		# @test cm.accuracy > 0.99

		print("  acc: ", round(cm.accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
		display(cm.matrix)

		global_logger(ConsoleLogger(stderr, Logging.Info));

		println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
	end
	return T;
end

# testDatasets(d, timeit::Bool = true) = map((x)->testDataset(x, timeit), d);

datasets = Tuple{String,Tuple{Array,Array,Array,Array}}[
	# ("simpleDataset",traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng())...,0.8)),
	# ("simpleDataset2",traintestsplit(simpleDataset2(200,n_variables = 5,rng = my_rng())...,0.8)),
	# ("Eduard-5",EduardDataset(5)),
	# ("Eduard-10",EduardDataset(10)),
	("PaviaDataset, 1x1",                           traintestsplit(SampleLandCoverDataset(9*30,  1, "Pavia", rng = my_rng())...,0.8)),
	("PaviaDataset, 3x3 flattened",                 traintestsplit(SampleLandCoverDataset(9*30,  3, "Pavia", flattened = true, rng = my_rng())...,0.8)),
	("PaviaDataset, 3x3",                           traintestsplit(SampleLandCoverDataset(9*30,  3, "Pavia", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 1x1",            traintestsplit(SampleLandCoverDataset(16*30, 1, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 3x3 flattened",  traintestsplit(SampleLandCoverDataset(16*30, 3, flattened = true, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 3x3",            traintestsplit(SampleLandCoverDataset(16*30, 3, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	("PaviaDataset, 5x5",                           traintestsplit(SampleLandCoverDataset(9*30,  5, "Pavia", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 5x5",            traintestsplit(SampleLandCoverDataset(16*30, 5, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	("PaviaDataset, 7x7",                           traintestsplit(SampleLandCoverDataset(9*30,  7, "Pavia", rng = my_rng())...,0.8)),
	("IndianPinesCorrectedDataset, 7x7",            traintestsplit(SampleLandCoverDataset(16*30, 7, "IndianPinesCorrected", rng = my_rng())...,0.8)),
	# ("PaviaDataset, 4x4",traintestsplit(SampleLandCoverDataset(9*30, 4, "Pavia", rng = my_rng())...,0.8)),
	# ("IndianPinesCorrectedDataset, 4x4",traintestsplit(SampleLandCoverDataset(16*30, 4, "IndianPinesCorrected", rng = my_rng())...,0.8)),
];

args = (
	max_depth=-1,
	min_samples_leaf=4,
	min_samples_split=8,
	# min_purity_increase=0.02,
	# max_purity_split=1.0, # TODO there's something wrong here, I think this sets min_purity_increase.
)
kwargs = (
	initCondition=DecisionTree.startAtCenter,
	# initCondition=DecisionTree.startWithRelationAll,
	# ontology=getIntervalOntologyOfDim(Val(2))
	ontology=getIntervalTopologicalOntologyOfDim(Val(2)),
	# test_operators=[ModalLogic.TestOpLes],
	test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLes],
	# test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLes, ModalLogic.TestOpGeq075, ModalLogic.TestOpLes075],
)
# timeit = 0
# T = testDataset(("PaviaDataset, sample", traintestsplit(SampleLandCoverDataset(9*5, 3, "Pavia", n_variables = 1, rng = my_rng())...,0.8)), timeit, args=args, kwargs=kwargs);
# T = testDataset(("PaviaDataset, sample", traintestsplit(SampleLandCoverDataset(9*5, 1, "Pavia", n_variables = 1, rng = my_rng())...,0.8)), timeit, args=args, kwargs=kwargs);
timeit = 2
T = testDataset(datasets[3], false, args=args, kwargs=kwargs);

# exit()

T = testDataset(("PaviaDataset, sample", traintestsplit(SampleLandCoverDataset(9*5, 3, "Pavia", n_variables = 10, rng = my_rng())...,0.8)), timeit, args=args, kwargs=kwargs);
T = testDataset(datasets[1], timeit, args=args, kwargs=kwargs);
T = testDataset(datasets[2], timeit, args=args, kwargs=kwargs);
T = testDataset(datasets[3], timeit, args=args, kwargs=kwargs);
T = testDataset(datasets[4], timeit, args=args, kwargs=kwargs);
T = testDataset(datasets[5], timeit, args=args, kwargs=kwargs);
T = testDataset(datasets[6], timeit, args=args, kwargs=kwargs);


timeit = 1
for min_purity_increase in [0.0, 0.02]
	for max_purity_split in [1.0, 0.9]
		for ontology in [getIntervalTopologicalOntologyOfDim(Val(2)), getIntervalOntologyOfDim(Val(2))]
			for initCondition in [DecisionTree.startAtCenter,DecisionTree.startWithRelationAll]
				args = (
					max_depth=-1,
					min_samples_leaf=4,
					min_samples_split=8,
					min_purity_increase=min_purity_increase,
					max_purity_split=max_purity_split, # TODO there's something wrong here, I think this sets min_purity_increase.
				)
				kwargs = (
					initCondition=initCondition,
					ontology=ontology,
					# test_operators=[ModalLogic.TestOpLes],
					test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLes],
					# test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLes, ModalLogic.TestOpGeq075, ModalLogic.TestOpLes075],
				)

				T = testDataset(datasets[1], timeit, args=args, kwargs=kwargs);
				T = testDataset(datasets[3], timeit, args=args, kwargs=kwargs);
				T = testDataset(datasets[4], timeit, args=args, kwargs=kwargs);
				T = testDataset(datasets[6], timeit, args=args, kwargs=kwargs);

				T = testDataset(datasets[7], timeit, args=args, kwargs=kwargs);
				T = testDataset(datasets[8], timeit, args=args, kwargs=kwargs);
				T = testDataset(datasets[9], timeit, args=args, kwargs=kwargs);
				T = testDataset(datasets[10], timeit, args=args, kwargs=kwargs);
			end
		end
	end
end
# run(`say 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'`)

# @profview T = testDataset(datasets[2], false)
# T = testDataset(datasets[1], false)
# @profile T = testDataset(datasets[1], false)
# pprof()
# Profile.print()
# testDatasets(datasets);

# X_train, Y_train, X_test, Y_test = traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng())...,0.8)
# model = fit!(DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold), X_train, Y_train)
# cm = confusion_matrix(Y_test, predict(model, X_test))
# @test cm.accuracy > 0.99

# for relations in [ModalLogic.TopoRelations, ModalLogic.IA2DRelations]
# 	for (X,Y) in Iterators.product(4:6,4:9)
# 		sum = 0
# 		for rel in relations
# 			sum += (ModalLogic.enumAcc(S, rel, X,Y) |> collect |> length)
# 			end
# 		# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
# 		@assert sum == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)
# 	end
# 	for (X,Y) in Iterators.product(4:6,4:9)
# 		sum = 0
# 		for rel in relations
# 			sum += (ModalLogic.enumAcc(S, rel, X,Y) |> distinct |> collect |> length)
# 			end
# 		# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
# 		@assert sum == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)
# 	end
# end

# S = [ModalLogic.Interval2D((2,3),(3,4))]
# S = [ModalLogic.Interval2D((2,4),(2,4))]
S = [ModalLogic.Interval2D((2,3),(2,3))]
relations = ModalLogic.TopoRelations
(X,Y) = (3,3)
SUM = 0
for rel in relations
	println(rel)
	map(ModalLogic.print_world, ModalLogic.enumAcc(S, rel, X,Y) |> collect)
	global SUM
	SUM += (ModalLogic.enumAcc(S, rel, X,Y) |> collect |> length)
end
# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
@assert SUM == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)

# Test that T = testDataset(datasets[1], timeit, args=args, kwargs=kwargs); with test_operators=[ModalLogic.TestOpLes] and without is equivalent
