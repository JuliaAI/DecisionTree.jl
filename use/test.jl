# julia-1.5.4
# julia

include("scanner.jl")

# testDatasets(d, timing_mode::Bool = true) = map((x)->testDataset(x, timing_mode), d);

rng = my_rng()

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
	# ontology = getIntervalRCC8OntologyOfDim(Val(1)),
	ontology = getIntervalRCC8OntologyOfDim(Val(2)),
	# ontology = getIntervalRCC5OntologyOfDim(Val(2)),

	# ontology=Ontology(ModalLogic.Interval2D,ModalLogic.AbstractRelation[]),
	useRelationId = true,
	# useRelationId = false,
	# useRelationAll = true,
	useRelationAll = false,
	# test_operators = [TestOpGeq],
	# test_operators = [TestOpLeq],
	test_operators = [TestOpGeq, TestOpLeq],
	# test_operators = [TestOpGeq, TestOpLeq, TestOpGeq_85, TestOpLeq_85],
	# test_operators = [TestOpGeq_70, TestOpLeq_70],
	# test_operators = [TestOpGeq_85, TestOpLeq_85],
	# test_operators = [TestOpGeq_75],
	# rng = my_rng,
	# rng = DecisionTree.mk_rng(123),
)

# rng_i = DecisionTree.mk_rng(124)
rng_i = DecisionTree.mk_rng(1)

# timing_mode = :btime
# timing_mode = :none

# T = testDataset(("Pavia, 3x3",                           traintestsplit(SampleLandCoverDataset("Pavia", 30,  3, n_variables = 1, rng = my_rng),0.8)), timing_mode, args=args, kwargs=kwargs);

# T = testDataset(datasets[1], timing_mode, args=args, kwargs=kwargs);
# T = testDataset(datasets[2], timing_mode, log_level=DecisionTree.DTOverview, args=args, kwargs=kwargs);
# T = testDataset(datasets[3], timing_mode, args=args, kwargs=kwargs);

# exit()

# timing_mode = :none
# log_level = DecisionTree.DTOverview
# log_level = Logging.Warn

# n_instances = 1
# n_instances = 100
# n_instances = 500

# for dataset_name in ["IndianPines", "Pavia"]
# 	for loss in [DecisionTree.util.entropy]
# 		for min_samples_leaf in [1, 2, 4, 6]
# 			for min_purity_increase in [0.01, 0.05]
# 				for min_loss_at_leaf in [0.3, 0.5, 0.7]
# 					cur_args = merge(args, (loss=loss,
# 																	min_samples_leaf=min_samples_leaf,
# 																	min_purity_increase=min_purity_increase,
# 																	min_loss_at_leaf=min_loss_at_leaf,
# 																	))
# 					cur_kwargs = merge(kwargs, ())

# 					for i in 1:5
# 						rng_new = DecisionTree.mk_rng(abs(rand(rng_i, Int)))
# 						testDataset(("$(dataset_name), 1x1",                           traintestsplit(SampleLandCoverDataset(dataset_name,                 n_instances,        1,                   rng = rng_new),0.8)), timing_mode, log_level = log_level, args=cur_args, kwargs=cur_kwargs);
# 					end
# 				end
# 			end
# 		end
# 	end
# end

# exit()

# relations = [ModalLogic.RCC8Relations, subsets(ModalLogic.RCC8Relations,1), subsets(ModalLogic.RCC8Relations,2)...]
# i_relation = 1
# while i_relation <= length(relations)
# relation = relations[i_relation]
# # relation = ModalLogic._IA2DRel(RelationId , ModalLogic.IA_O)
# println(relation)
# cur_kwargs = merge(kwargs, (ontology=Ontology(ModalLogic.Interval2D,relation),))
# T = testDataset(datasets[databatch*3+3], timing_mode, log_level = Logging.Warn, args=args, kwargs=cur_kwargs);
# println(T)
# i_relation+=1
# end

loss = DecisionTree.util.entropy
min_samples_leaf = 4
min_purity_increase = 0.01
min_loss_at_leaf = 0.3

selected_args = merge(args, (loss = loss,
															min_samples_leaf = min_samples_leaf,
															min_purity_increase = min_purity_increase,
															min_loss_at_leaf = min_loss_at_leaf,
															))
log_level = DecisionTree.DTOverview
# log_level = Logging.Warn

# timing_mode = :btime
timing_mode = :none
round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt8


n_instances = 10
# n_instances = 100
# n_instances = 300
# n_instances = 500

# datasets = [
# 	("IndianPines, 1x1",  traintestsplit(SampleLandCoverDataset("IndianPines",  40,  1,                   rng = rng),0.8)),
# 	("Salinas, 1x1",      traintestsplit(SampleLandCoverDataset("Salinas",      40,  1,                   rng = rng),0.8)),
# 	("Salinas-A, 1x1",    traintestsplit(SampleLandCoverDataset("Salinas-A",    40,  1,                  rng = rng),0.8)),
# 	("PaviaCentre, 1x1",  traintestsplit(SampleLandCoverDataset("PaviaCentre",  70,  1,                   rng = rng),0.8)),
# 	("Pavia, 1x1",        traintestsplit(SampleLandCoverDataset("Pavia",        70,  1,                   rng = rng),0.8)),
# ]
# for dataset_name in ["IndianPines", "Salinas", "Salinas-A", "PaviaCentre", "Pavia"]


# for dataset_name in ["IndianPines", "Pavia"]
# 	for i in 1:10
# 		rng_new = DecisionTree.mk_rng(abs(rand(rng_i, Int)))
# 		testDataset((dataset_name * ", 1x1",  traintestsplit(SampleLandCoverDataset(dataset_name,  n_instances,  1,                   rng = rng_new),0.8)), timing_mode, log_level = log_level, args=selected_args, kwargs=kwargs, rng = rng_new);
# 	end
# end

# exit()

# pavia_instperclass = 70
# indian_pines_instperclass = 40

# www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
# datasets = Tuple{String,Tuple{Tuple{Array,Vector},Tuple{Array,Vector},Vector{String}}}[
# 	# ("simpleDataset",traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng()),0.8)),
# 	# ("simpleDataset2",traintestsplit(simpleDataset2(200,n_variables = 5,rng = my_rng()),0.8)),
# 	# ("Eduard-5",SplatEduardDataset(5)),
# 	# ("Eduard-10",SplatEduardDataset(10)),	
# 	#
# 	("Pavia, 1x1",                           traintestsplit(SampleLandCoverDataset("Pavia",                 pavia_instperclass,        1,                   rng = rng),0.8)),
# 	("Pavia, 3x3",                           traintestsplit(SampleLandCoverDataset("Pavia",                 pavia_instperclass,        3,                   rng = rng),0.8)),
# 	("Pavia, 5x5",                           traintestsplit(SampleLandCoverDataset("Pavia",                 pavia_instperclass,        5,                   rng = rng),0.8)),
# 	("IndianPines, 1x1",                     traintestsplit(SampleLandCoverDataset("IndianPines",           indian_pines_instperclass, 1,                   rng = rng),0.8)),
# 	("IndianPines, 3x3",                     traintestsplit(SampleLandCoverDataset("IndianPines",           indian_pines_instperclass, 3,                   rng = rng),0.8)),
# 	("IndianPines, 5x5",                     traintestsplit(SampleLandCoverDataset("IndianPines",           indian_pines_instperclass, 5,                   rng = rng),0.8)),
# 	("Pavia, 3x3 flattened",                 traintestsplit(SampleLandCoverDataset("Pavia",                 pavia_instperclass,        3, flattened = true, rng = rng),0.8)),
# 	("Pavia, 5x5 flattened",                 traintestsplit(SampleLandCoverDataset("Pavia",                 pavia_instperclass,        5, flattened = true, rng = rng),0.8)),
# 	("IndianPines, 3x3 flattened",           traintestsplit(SampleLandCoverDataset("IndianPines",           indian_pines_instperclass, 3, flattened = true, rng = rng),0.8)),
# 	("IndianPines, 5x5 flattened",           traintestsplit(SampleLandCoverDataset("IndianPines",           indian_pines_instperclass, 5, flattened = true, rng = rng),0.8)),
# ];

o_RCC8, o_RCC5 = getIntervalRCC8OntologyOfDim(Val(2)), getIntervalRCC5OntologyOfDim(Val(2))

for dataset_name in ["Salinas", "Salinas-A", "PaviaCentre", "Pavia", "IndianPines"]
              rng_i = DecisionTree.mk_rng(1)
	for i in 1:5
		rng_new = DecisionTree.mk_rng(abs(rand(rng_i, Int)))
		# for window_size in [3,1] #,5]
		for (window_size,flattened,ontology) in [(1,false,o_RCC8),(3,true,o_RCC8),(3,false,o_RCC8),(3,false,o_RCC5)] #,5]
			# for dataset_name in ["Salinas", "Salinas-A", "PaviaCentre"] # "IndianPines", "Pavia"]
			for useRelationAll in [false] # true]
				for initCondition in [DecisionTree.startAtCenter]
					for test_operators in [
							[TestOpGeq, TestOpLeq,
								TestOpGeq_60, TestOpLeq_60,
								TestOpGeq_70, TestOpLeq_70,
								TestOpGeq_80, TestOpLeq_80,
								TestOpGeq_90, TestOpLeq_90],
							# [TestOpLeq_90,TestOpLeq_80],
							# [TestOpGeq, TestOpLeq],
																	]

						cur_args = selected_args
						cur_kwargs = merge(kwargs, (
							ontology = ontology,
							useRelationAll = useRelationAll,
							initCondition = startAtCenter,
							test_operators = test_operators,
							))
						# testDataset(("Pavia, 3x3", traintestsplit(SampleLandCoverDataset("Pavia", 30,  3, n_variables = 10, rng = rng_new),0.8)), timing_mode, log_level = DecisionTree.DTOverview, args=cur_args, kwargs=cur_kwargs);
						# testDataset(datasets[databatch*5+3], timing_mode, log_level = DecisionTree.DTOverview, args=cur_args, kwargs=cur_kwargs);
						# exit()

						println("$(ontology)\t$(i)\t$(window_size)\t$(dataset_name)\t$(useRelationAll)\t$(initCondition)\t$(test_operators)")
						# rng_new = DecisionTree.mk_rng(abs(rand(rng_i, Int)))
						rng_new = copy(rng_new)
						dataset = SampleLandCoverDataset(dataset_name,                 n_instances,        window_size, flattened = flattened,                   rng = rng_new)
						testDataset("$(dataset_name), $(window_size)x$(window_size)" * (if flattened " flattened" else "" end),
														dataset,
														0.8,
														timing_mode,
														round_dataset_to_datatype = round_dataset_to_datatype,
														log_level = log_level,
														args = cur_args,
														kwargs = cur_kwargs,
														precompute_gammas = true,
														test_tree = true,
														test_forest = true,
														);
						
						# testDataset(datasets[databatch*5+1], timing_mode, log_level = log_level, args=cur_args, kwargs=cur_kwargs);
						# testDataset(datasets[databatch*5+2], timing_mode, log_level = log_level, args=cur_args, kwargs=cur_kwargs);
						# testDataset(datasets[databatch*5+3], timing_mode, log_level = log_level, args=cur_args, kwargs=cur_kwargs);
						# testDataset(datasets[databatch*5+4], timing_mode, log_level = log_level, args=cur_args, kwargs=cur_kwargs);
						# testDataset(datasets[databatch*5+5], timing_mode, log_level = log_level, args=cur_args, kwargs=cur_kwargs);
					end
				end
			end
		end
	end
end

exit()

T = testDataset(datasets[3], 0, args=args, kwargs=kwargs);
T = testDataset(datasets[6], 0, args=args, kwargs=kwargs);
T = testDataset("Pavia, 3x3",                           traintestsplit(SampleLandCoverDataset("Pavia", 30,  3, n_variables = 1, rng = my_rng),0.8), timing_mode, args=args, kwargs=kwargs);

exit()


# T = testDataset(("Pavia, sample", traintestsplit(SampleLandCoverDataset("Pavia", 5, 3, n_variables = 1, rng = my_rng),0.8)), timing_mode, args=args, kwargs=kwargs);
# T = testDataset(("Pavia, sample", traintestsplit(SampleLandCoverDataset("Pavia", 5, 1, n_variables = 1, rng = my_rng),0.8)), timing_mode, args=args, kwargs=kwargs);


# TODO test the same with window 3x5; also test with a specific initial world. One that allows During, for example, or one on the border
relations = [ModalLogic.RCC8Relations...,ModalLogic.IA2DRelations...,]
i_relation = 1
while i_relation <= length(relations)
	relation = relations[i_relation]
	# relation = ModalLogic._IA2DRel(RelationId , ModalLogic.IA_O)
	println(relation)
	kwargs = (
		initCondition=DecisionTree._startAtWorld(ModalLogic.Interval2D((1,3),(3,4))),
		# initCondition=DecisionTree._startAtWorld(ModalLogic.Interval2D((1,4),(1,6))),
		useRelationId = false,
		useRelationAll = false,
		test_operators=[TestOpGeq, TestOpLeq],
		ontology=Ontology(ModalLogic.Interval2D,[relation]),
		# ontology=Ontology(ModalLogic.Interval2D,relation_set),
	)
	T = testDataset(("Pavia, 3x3 mod",                           traintestsplit(SampleLandCoverDataset(30, ,5),  ("Pavia", rng = my_rng),0.8)), 0, args=args, kwargs=kwargs);
	println(T)
	i_relation+=1
end



# exit()

T = testDataset(("Pavia, sample", traintestsplit(SampleLandCoverDataset("Pavia", 5, 3, n_variables = 10, rng = my_rng),0.8)), timing_mode, args=args, kwargs=kwargs);
post_pruning_purity_thresholds = [0.7, 0.8, 0.9]
T = testDataset(datasets[1], timing_mode, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[2], timing_mode, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[3], timing_mode, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[4], timing_mode, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[5], timing_mode, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);
T = testDataset(datasets[6], timing_mode, post_pruning_purity_thresholds = post_pruning_purity_thresholds, args=args, kwargs=kwargs);


timing_mode = :time
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
					# test_operators=[TestOpLeq],
					test_operators=[TestOpGeq, TestOpLeq],
					# test_operators=[TestOpGeq, TestOpLeq, TestOpGeq075, TestOpLeq075],
				)

				T = testDataset(datasets[1], timing_mode, args=args, kwargs=kwargs);
				T = testDataset(datasets[3], timing_mode, args=args, kwargs=kwargs);
				T = testDataset(datasets[4], timing_mode, args=args, kwargs=kwargs);
				T = testDataset(datasets[6], timing_mode, args=args, kwargs=kwargs);

				T = testDataset(datasets[7], timing_mode, args=args, kwargs=kwargs);
				T = testDataset(datasets[8], timing_mode, args=args, kwargs=kwargs);
				T = testDataset(datasets[9], timing_mode, args=args, kwargs=kwargs);
				T = testDataset(datasets[10], timing_mode, args=args, kwargs=kwargs);
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

# X_train, Y_train, X_test, Y_test = traintestsplit(simpleDataset(200,n_variables = 50,rng = my_rng),0.8)
# model = fit!(DecisionTreeClassifier(pruning_purity_threshold=pruning_purity_threshold), X_train, Y_train)
# cm = confusion_matrix(Y_test, predict(model, X_test))
# @test cm.overall_accuracy > 0.99

# for relations in [ModalLogic.RCC8Relations, ModalLogic.IA2DRelations]
# 	for (X,Y) in Iterators.product(4:6,4:9)
# 		sum = 0
# 		for rel in relations
# 			sum += (ModalLogic.enumAccessibles(S, rel, X,Y) |> collect |> length)
# 			end
# 		# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
# 		@assert sum == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)
# 	end
# 	for (X,Y) in Iterators.product(4:6,4:9)
# 		sum = 0
# 		for rel in relations
# 			sum += (ModalLogic.enumAccessibles(S, rel, X,Y) |> distinct |> collect |> length)
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
	map(ModalLogic.print_world, ModalLogic.enumAccessibles(S, rel, X,Y) |> collect)
	global SUM
	SUM += (ModalLogic.enumAccessibles(S, rel, X,Y) |> collect |> length)
end
# println(X, " ", Y, " ", (X*(X+1))/2 * (Y*(Y+1))/2 - 1, " ", sum)
@assert SUM == ((X*(X+1))/2 * (Y*(Y+1))/2 - 1)

# Test that T = testDataset(datasets[1], timing_mode, args=args, kwargs=kwargs); with test_operators=[TestOpLeq] and without is equivalent
