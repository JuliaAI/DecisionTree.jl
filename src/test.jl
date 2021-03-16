# julia-1.5.4
# julia

include("test-header.jl")

# testDatasets(d, timeit::Bool = true) = map((x)->testDataset(x, timeit), d);

datasets = Tuple{String,Tuple{Tuple{Array,Vector},Tuple{Array,Vector},Vector{String}}}[
	# ("simpleDataset",traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng()),0.8)),
	# ("simpleDataset2",traintestsplit(simpleDataset2(200,n_variables = 5,rng = my_rng()),0.8)),
	# ("Eduard-5",EduardDataset(5)),
	# ("Eduard-10",EduardDataset(10)),
	("PaviaDataset, 1x1",                           traintestsplit(SampleLandCoverDataset(9*70,  1, "Pavia", rng = my_rng()),0.5)),
	("PaviaDataset, 3x3 flattened",                 traintestsplit(SampleLandCoverDataset(9*70,  3, "Pavia", flattened = true, rng = my_rng()),0.5)),
	("PaviaDataset, 3x3",                           traintestsplit(SampleLandCoverDataset(9*70,  3, "Pavia", rng = my_rng()),0.5)),
	("PaviaDataset, 5x5 flattened",                 traintestsplit(SampleLandCoverDataset(9*70,  5, "Pavia", flattened = true, rng = my_rng()),0.5)),
	("PaviaDataset, 5x5",                           traintestsplit(SampleLandCoverDataset(9*70,  5, "Pavia", rng = my_rng()),0.5)),
	("IndianPinesCorrectedDataset, 1x1",            traintestsplit(SampleLandCoverDataset(16*40, 1, "IndianPinesCorrected", rng = my_rng()),0.5)),
	("IndianPinesCorrectedDataset, 3x3 flattened",  traintestsplit(SampleLandCoverDataset(16*40, 3, "IndianPinesCorrected", flattened = true, rng = my_rng()),0.5)),
	("IndianPinesCorrectedDataset, 3x3",            traintestsplit(SampleLandCoverDataset(16*40, 3, "IndianPinesCorrected", rng = my_rng()),0.5)),
	("IndianPinesCorrectedDataset, 5x5 flattened",  traintestsplit(SampleLandCoverDataset(16*40, 5, "IndianPinesCorrected", flattened = true, rng = my_rng()),0.5)),
	("IndianPinesCorrectedDataset, 5x5",            traintestsplit(SampleLandCoverDataset(16*70, 5, "IndianPinesCorrected", rng = my_rng()),0.5)),
	# ("PaviaDataset, 7x7",                           traintestsplit(SampleLandCoverDataset(9*70,  7, "Pavia", rng = my_rng()),0.5)),
	# ("IndianPinesCorrectedDataset, 7x7",            traintestsplit(SampleLandCoverDataset(16*70, 7, "IndianPinesCorrected", rng = my_rng()),0.5)),
	# ("PaviaDataset, 4x4",traintestsplit(SampleLandCoverDataset(9*30, 4, "Pavia", rng = my_rng()),0.8)),
	# ("IndianPinesCorrectedDataset, 4x4",traintestsplit(SampleLandCoverDataset(16*30, 4, "IndianPinesCorrected", rng = my_rng()),0.8)),
];

args = (
	loss = DecisionTree.util.entropy,
	# loss = DecisionTree.util.gini,
	# loss = DecisionTree.util.zero_one,
	# max_depth = -1,
	# min_samples_leaf = 4,
	# min_purity_increase = 0.02, # TODO check this
	# min_loss_at_leaf = 1.0, # TODO check there's something wrong here, I think this sets min_purity_increase.
)

# TODO add parameter: allow relationAll at all levels? Maybe it must be part of the relations... I don't know
kwargs = (
	initCondition = DecisionTree.startAtCenter,
	# initCondition = DecisionTree._startAtWorld(ModalLogic.Interval2D((1,3),(3,4))),
	# initCondition = DecisionTree.startWithRelationAll,
	
	# ontology = getIntervalOntologyOfDim(Val(2)),
	# ontology = Ontology(ModalLogic.Interval2D,setdiff(Set(ModalLogic.RCC8Relations),Set([ModalLogic.Topo_PO]))),
	# ontology = Ontology(ModalLogic.Interval2D,[ModalLogic._IA2DRel(i,j) for j in [ModalLogic.IA_O,ModalLogic.IA_Oi] for i in [ModalLogic.IA_O,ModalLogic.IA_Oi]]),
	ontology = getIntervalRCC8OntologyOfDim(Val(2)),
	# ontology = getIntervalRCC5OntologyOfDim(Val(2)),

	# ontology=Ontology(ModalLogic.Interval2D,ModalLogic.AbstractRelation[]),
	useRelationId = true,
	# useRelationId = false,
	# useRelationAll = true,
	useRelationAll = false,
	# test_operators = [ModalLogic.TestOpGeq],
	# test_operators = [ModalLogic.TestOpLeq],
	test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
	# test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq, ModalLogic.TestOpGeq_85, ModalLogic.TestOpLeq_85],
	# test_operators = [ModalLogic.TestOpGeq_75, ModalLogic.TestOpLeq_75],
	# test_operators = [ModalLogic.TestOpGeq_85, ModalLogic.TestOpLeq_85],
	# test_operators = [ModalLogic.TestOpGeq_75],
)
# timeit = 2
# timeit = 0

# T = testDataset(("PaviaDataset, 3x3",                           traintestsplit(SampleLandCoverDataset(9*30,  3, "Pavia", n_variables = 1, rng = my_rng()),0.8)), timeit, args=args, kwargs=kwargs);

# T = testDataset(datasets[1], timeit, args=args, kwargs=kwargs);
# T = testDataset(datasets[2], timeit, debugging_level=DecisionTree.DTOverview, args=args, kwargs=kwargs);
# T = testDataset(datasets[3], timeit, args=args, kwargs=kwargs);

# exit()

# timeit = 0

# for databatch in [0,1]
# 	for loss in [DecisionTree.util.entropy]
# 		for min_samples_leaf in [1, 2, 4, 6]
# 			for min_purity_increase in [0.01, 0.05]
# 				for min_loss_at_leaf in [0.4, 0.45, 0.5, 0.55, 0.6]
# 					cur_args = merge(args, (loss=loss,
# 																	min_samples_leaf=min_samples_leaf,
# 																	min_purity_increase=min_purity_increase,
# 																	min_loss_at_leaf=min_loss_at_leaf,
# 																	))
# 					cur_kwargs = merge(kwargs, ())
# 					testDataset(datasets[databatch*5+1], timeit, debugging_level = Logging.Warn, args=cur_args, kwargs=cur_kwargs);
# 					# testDataset(datasets[databatch*3+2], timeit, debugging_level = Logging.Warn, args=args, kwargs=kwargs);

# 				end
# 			end
# 		end
# 	end

# 	# relations = [ModalLogic.RCC8Relations, subsets(ModalLogic.RCC8Relations,1), subsets(ModalLogic.RCC8Relations,2)...]
# 	# i_relation = 1
# 	# while i_relation <= length(relations)
# 		# relation = relations[i_relation]
# 		# # relation = ModalLogic._IA2DRel(ModalLogic.RelationId , ModalLogic.IA_O)
# 		# println(relation)
# 		# cur_kwargs = merge(kwargs, (ontology=Ontology(ModalLogic.Interval2D,relation),))
# 		# T = testDataset(datasets[databatch*3+3], timeit, debugging_level = Logging.Warn, args=args, kwargs=cur_kwargs);
# 		# println(T)
# 		# i_relation+=1
# 	# end
# end

timeit = 0

loss = DecisionTree.util.entropy
min_samples_leaf = 2
min_purity_increase = 0.01
min_loss_at_leaf = 0.55

selected_args = merge(args, (loss = loss,
															min_samples_leaf = min_samples_leaf,
															min_purity_increase = min_purity_increase,
															min_loss_at_leaf = min_loss_at_leaf,
															))

for ontology in [getIntervalRCC8OntologyOfDim(Val(2)), getIntervalRCC5OntologyOfDim(Val(2))]
	for databatch in [0,1]
		for useRelationAll in [true]
			for initCondition in [DecisionTree.startAtCenter]
				for test_operators in [[ModalLogic.TestOpGeq, ModalLogic.TestOpLeq,
																ModalLogic.TestOpGeq_60, ModalLogic.TestOpLeq_60,
																ModalLogic.TestOpGeq_70, ModalLogic.TestOpLeq_70,
																ModalLogic.TestOpGeq_80, ModalLogic.TestOpLeq_80,
																ModalLogic.TestOpGeq_90, ModalLogic.TestOpLeq_90]]
						cur_args = selected_args
						cur_kwargs = merge(kwargs, (
							ontology = ontology,
							useRelationAll = useRelationAll,
							initCondition = startAtCenter,
							test_operators = test_operators,
							))
						testDataset(datasets[databatch*5+3], timeit, debugging_level = DecisionTree.DTOverview, args=cur_args, kwargs=cur_kwargs);
						exit()
						testDataset(datasets[databatch*5+1], timeit, debugging_level = Logging.Warn, args=cur_args, kwargs=cur_kwargs);
						testDataset(datasets[databatch*5+2], timeit, debugging_level = Logging.Warn, args=cur_args, kwargs=cur_kwargs);
						testDataset(datasets[databatch*5+3], timeit, debugging_level = Logging.Warn, args=cur_args, kwargs=cur_kwargs);
						testDataset(datasets[databatch*5+4], timeit, debugging_level = Logging.Warn, args=cur_args, kwargs=cur_kwargs);
						testDataset(datasets[databatch*5+5], timeit, debugging_level = Logging.Warn, args=cur_args, kwargs=cur_kwargs);
				end
			end
		end
	end
end

exit()

T = testDataset(datasets[3], 0, args=args, kwargs=kwargs);
T = testDataset(datasets[6], 0, args=args, kwargs=kwargs);
T = testDataset(("PaviaDataset, 3x3",                           traintestsplit(SampleLandCoverDataset(9*30,  3, "Pavia", n_variables = 1, rng = my_rng()),0.8)), timeit, args=args, kwargs=kwargs);

exit()


# T = testDataset(("PaviaDataset, sample", traintestsplit(SampleLandCoverDataset(9*5, 3, "Pavia", n_variables = 1, rng = my_rng()),0.8)), timeit, args=args, kwargs=kwargs);
# T = testDataset(("PaviaDataset, sample", traintestsplit(SampleLandCoverDataset(9*5, 1, "Pavia", n_variables = 1, rng = my_rng()),0.8)), timeit, args=args, kwargs=kwargs);


# TODO test the same with window 3x5; also test with a specific initial world. One that allows During, for example, or one on the border
relations = [ModalLogic.RCC8Relations...,ModalLogic.IA2DRelations...,]
i_relation = 1
while i_relation <= length(relations)
	relation = relations[i_relation]
	# relation = ModalLogic._IA2DRel(ModalLogic.RelationId , ModalLogic.IA_O)
	println(relation)
	kwargs = (
		initCondition=DecisionTree._startAtWorld(ModalLogic.Interval2D((1,3),(3,4))),
		# initCondition=DecisionTree._startAtWorld(ModalLogic.Interval2D((1,4),(1,6))),
		useRelationId = false,
		useRelationAll = false,
		test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
		ontology=Ontology(ModalLogic.Interval2D,[relation]),
		# ontology=Ontology(ModalLogic.Interval2D,relation_set),
	)
	T = testDataset(("PaviaDataset, 3x3 mod",                           traintestsplit(SampleLandCoverDataset(9*30, (3,5), "Pavia", rng = my_rng()),0.8)), 0, args=args, kwargs=kwargs);
	println(T)
	i_relation+=1
end



# exit()

T = testDataset(("PaviaDataset, sample", traintestsplit(SampleLandCoverDataset(9*5, 3, "Pavia", n_variables = 10, rng = my_rng()),0.8)), timeit, args=args, kwargs=kwargs);
post_pruning_purity_thresholds = [0.7, 0.8, 0.9]
T = testDataset(datasets[1], timeit, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[2], timeit, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[3], timeit, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[4], timeit, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[5], timeit, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[6], timeit, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);


timeit = 1
for min_purity_increase in [0.0, 0.02]
	for max_purity_split in [1.0, 0.9]
		for ontology in [getIntervalRCC8OntologyOfDim(Val(2)), getIntervalOntologyOfDim(Val(2))]
			for initCondition in [DecisionTree.startAtCenter,DecisionTree.startWithRelationAll]
				args = (
					max_depth=-1,
					min_samples_leaf=4,
					min_purity_increase=min_purity_increase,
					max_purity_split=max_purity_split, # TODO there's something wrong here, I think this sets min_purity_increase.
				)
				kwargs = (
					initCondition=initCondition,
					ontology=ontology,
					# test_operators=[ModalLogic.TestOpLeq],
					test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
					# test_operators=[ModalLogic.TestOpGeq, ModalLogic.TestOpLeq, ModalLogic.TestOpGeq075, ModalLogic.TestOpLeq075],
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

# X_train, Y_train, X_test, Y_test = traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng()),0.8)
# model = fit!(DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold), X_train, Y_train)
# cm = confusion_matrix(Y_test, predict(model, X_test))
# @test cm.accuracy > 0.99

# for relations in [ModalLogic.RCC8Relations, ModalLogic.IA2DRelations]
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
relations = ModalLogic.RCC8Relations
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

# Test that T = testDataset(datasets[1], timeit, args=args, kwargs=kwargs); with test_operators=[ModalLogic.TestOpLeq] and without is equivalent
