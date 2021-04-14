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
# using ScikitLearnBase
using Statistics
using Test
# using Profile
# using PProf




using SHA
using Serialization

function get_dataset_hash_sha256(dataset)::String
	io = IOBuffer();
	serialize(io, dataset)
	result = bytes2hex(sha256(take!(io)))
	close(io)

	result
end


# TODO note that these splitting functions simply cut the dataset in two,
#  and they don't produce balanced cuts. To produce balanced cuts, one must manually stratify the dataset
traintestsplit(data::Tuple{MatricialDataset{D,3},AbstractVector{T},AbstractVector{String}},threshold; gammas = nothing, worldType = nothing) where {D,T} = begin
	(X,Y,class_labels) = data
	spl = floor(Int, length(Y)*threshold)
	X_train = X[:,1:spl,:]
	Y_train = Y[1:spl]
	gammas_train = 
		if isnothing(gammas) # || isnothing(worldType)
			gammas
		else
			DecisionTree.sliceGammasByInstances(worldType, gammas, 1:spl)
		end
	X_test  = X[:,spl+1:end,:]
	Y_test  = Y[spl+1:end]
	# if gammas == nothing
		# (X_train,Y_train),(X_test,Y_test),class_labels
	# else
	(X_train,Y_train),(X_test,Y_test),class_labels,gammas_train
	# end
end

traintestsplit(data::Tuple{MatricialDataset{D,4},AbstractVector{T},AbstractVector{String}},threshold; gammas = nothing, worldType = nothing) where {D,T} = begin
	(X,Y,class_labels) = data
	spl = floor(Int, length(Y)*threshold)
	X_train = X[:,:,1:spl,:]
	Y_train = Y[1:spl]
	gammas_train = 
		if isnothing(gammas) # || isnothing(worldType)
			gammas
		else
			DecisionTree.sliceGammasByInstances(worldType, gammas, 1:spl)
		end
	X_test  = X[:,:,spl+1:end,:]
	Y_test  = Y[spl+1:end]
	# if gammas == nothing
		# (X_train,Y_train),(X_test,Y_test),class_labels
	# else
	(X_train,Y_train),(X_test,Y_test),class_labels,gammas_train
	# end
end

function checkpoint_stdout(string::String)
	println("● ", string)
	flush(stdout)
end

include("example-datasets.jl")

gammas_saving_task = nothing

function testDataset(
		name                            ::String,
		dataset                         ::Tuple,
		split_threshold                 ::Union{Bool,AbstractFloat};
		debugging_level                 = DecisionTree.DTOverview,
		scale_dataset                   ::Union{Bool,Type} = false,
		post_pruning_purity_thresholds  = [],
		forest_args                     = [],
		tree_args                       = (),
		modal_args                      = (),
		precompute_gammas               = true,
		gammas_save_path                ::Union{String,NTuple{2,String},Nothing} = nothing,
		dataset_slice                   ::Union{AbstractVector,Nothing} = nothing,
		error_catching                  = false,
		rng                             = my_rng(),
		timeit                          ::Integer = 0,
	)
	println("Benchmarking dataset '$name'...")
	global_logger(ConsoleLogger(stderr, Logging.Warn));

	calculateGammas(modal_args, X_all_d) = begin
		if !precompute_gammas
			(modal_args, nothing, nothing)
		else
			haskey(modal_args, :ontology) || error("testDataset: precompute_gammas=1 requires `ontology` field in modal_args: $(modal_args)")

			X_all = OntologicalDataset{eltype(X_all_d),ndims(X_all_d)-2}(modal_args.ontology,X_all_d)

			worldType = modal_args.ontology.worldType

			old_logger = global_logger(ConsoleLogger(stderr, debugging_level))
			relationSet = nothing
			initCondition = modal_args.initCondition
			useRelationAll = modal_args.useRelationAll
			useRelationId = modal_args.useRelationId
			relationId_id = nothing
			relationAll_id = nothing
			availableModalRelation_ids = nothing
			allAvailableRelation_ids = nothing
			test_operators = deepcopy(modal_args.test_operators)
			(
				# X_all,
				test_operators, relationSet,
				useRelationId, useRelationAll, 
				relationId_id, relationAll_id,
				availableModalRelation_ids, allAvailableRelation_ids
				) = DecisionTree.treeclassifier.optimize_tree_parameters!(X_all, initCondition, useRelationAll, useRelationId, test_operators)

			# update values
			modal_args = merge(modal_args,
				(initCondition = initCondition, 
					useRelationAll = useRelationAll, 
					useRelationId = useRelationId, 
					test_operators = test_operators))

			# Generate path to gammas jld file

			if isa(gammas_save_path,String)
				gammas_save_path = (gammas_save_path,nothing)
			end

			gammas_save_path, dataset_name_str = gammas_save_path

			gammas_jld_path, gammas_hash_index_file, dataset_hash =
				if isnothing(gammas_save_path)
					(nothing, nothing, nothing)
				else
					dataset_hash = get_dataset_hash_sha256(X_all_d)
					(
						"$(gammas_save_path)/gammas_$(dataset_hash).jld",
						"$(gammas_save_path)/gammas_hash_index.csv",
						dataset_hash,
					)
				end

			gammas = 
				if !isnothing(gammas_jld_path) && isfile(gammas_jld_path)
					checkpoint_stdout("Loading gammas from file \"$(gammas_jld_path)\"...")

					Serialization.deserialize(gammas_jld_path)
				else
					gammas = DecisionTree.computeGammas(X_all,worldType,test_operators,relationSet,relationId_id,availableModalRelation_ids)
					if !isnothing(gammas_jld_path)
						checkpoint_stdout("Saving gammas to file \"$(gammas_jld_path)\" (size: $(Base.summarysize(gammas)))...")

						global gammas_saving_task
						if isa(gammas_saving_task, Task)
							# make sure there is no previous saving in progress
							wait(gammas_saving_task)
						end
						gammas_saving_task = @async Serialization.serialize(gammas_jld_path, gammas)
						# Add record line to the index file of the folder
						if !isnothing(dataset_name_str)
							# Generate path to gammas jld file)
							# TODO fix column_separator here
							append_in_file(gammas_hash_index_file, "$(dataset_hash);$(dataset_name_str)\n")
						end
					end
					gammas
				end

			println("(optimized) modal_args = ", modal_args)
			global_logger(old_logger);
			(modal_args, gammas, modal_args.ontology.worldType)
		end
	end

	println("forest_args = ", forest_args)
	# println("forest_args = ", length(forest_args), " × some forest_args structure")
	println("tree_args   = ", tree_args)
	println("modal_args  = ", modal_args)

	# Slice & split the dataset according to dataset_slice & split_threshold
	# The instances for which the gammas are computed are either all, or the ones specified for training.	
	# This depends on whether the dataset is already splitted or not.
	modal_args, (X_train, Y_train), (X_test, Y_test), class_labels, gammas_train = 
		if split_threshold != false

			# Unpack dataset
			length(dataset) == 3 || error("Wrong dataset length: $(length(dataset))")
			X, Y, class_labels = dataset

			# Apply scaling
			if scale_dataset != false
				X, Y, class_labels = scaleDataset((X, Y, class_labels), scale_dataset)
			end
			
			# Calculate gammas for the full set of instances
			modal_args, gammas, worldType = calculateGammas(modal_args, X)

			# Slice instances
			X, Y, gammas =
				if isnothing(dataset_slice)
					(X, Y, gammas)
				else
					(
						(@views ModalLogic.sliceDomainByInstances(X, dataset_slice)),
						(@views Y[dataset_slice]),
						if !isnothing(gammas)
							@views DecisionTree.sliceGammasByInstances(worldType, gammas, dataset_slice)
						else
							gammas
						end
					)
				end
			# dataset = (X, Y, class_labels)

			# Split in train/test
			((X_train, Y_train), (X_test, Y_test), class_labels, gammas_train) =
				traintestsplit((X, Y, class_labels), split_threshold, gammas = gammas, worldType = worldType)

			modal_args, (X_train, Y_train), (X_test, Y_test), class_labels, gammas_train
		else

			# Unpack dataset
			length(dataset) == 3 || error("Wrong dataset length: $(length(dataset))")
			(X_train, Y_train), (X_test, Y_test), class_labels = dataset

			# Apply scaling
			if scale_dataset != false
				(X_train, Y_train, class_labels) = scaleDataset((X_train, Y_train, class_labels), scale_dataset)
				(X_test,  Y_test,  class_labels) = scaleDataset((X_test,  Y_test,  class_labels), scale_dataset)
			end
			
			# Calculate gammas for the training instances
			modal_args, gammas, worldType = calculateGammas(modal_args, X_train)

			# Slice training instances
			X_train, Y_train, gammas_train =
				if isnothing(dataset_slice)
					(X_train, Y_train, gammas)
				else
					(
					@views ModalLogic.sliceDomainByInstances(X_train, dataset_slice),
					@views Y_train[dataset_slice],
					if !isnothing(gammas)
						@views DecisionTree.sliceGammasByInstances(worldType, gammas, dataset_slice)
					else
						gammas
					end
					)
				end
			# dataset = (X_train, Y_train), (X_test, Y_test), class_labels

			modal_args, (X_train, Y_train), (X_test, Y_test), class_labels, gammas_train
		end



	# println(" n_samples = $(size(X_train)[end-1])")
	println(" train size = $(size(X_train))")
	# global_logger(ConsoleLogger(stderr, Logging.Info))
	# global_logger(ConsoleLogger(stderr, debugging_level))
	# global_logger(ConsoleLogger(stderr, DecisionTree.DTDebug))

	function display_cm_as_row(cm::ConfusionMatrix)
		"|\t" *
		"$(round(cm.overall_accuracy*100, digits=2))%\t" *
		# "$(round(cm.mean_accuracy*100, digits=2))%\t" *
		"$(round(cm.kappa*100, digits=2))%\t" *
		# "$(round(DecisionTree.macro_F1(cm)*100, digits=2))%\t" *
		# "$(round.(cm.accuracies.*100, digits=2))%\t" *
		"$(round.(cm.F1s.*100, digits=2))%\t" *
		"$(round.(cm.sensitivities.*100, digits=2))%\t" *
		# "$(round.(cm.specificities.*100, digits=2))%\t" *
		"$(round.(cm.PPVs.*100, digits=2))%\t" *
		# "$(round.(cm.NPVs.*100, digits=2))%\t" *
		"||\t" *
		"$(round(DecisionTree.macro_weighted_F1(cm)*100, digits=2))%\t" *
		# "$(round(DecisionTree.macro_sensitivity(cm)*100, digits=2))%\t" *
		"$(round(DecisionTree.macro_weighted_sensitivity(cm)*100, digits=2))%\t" *
		# "$(round(DecisionTree.macro_specificity(cm)*100, digits=2))%\t" *
		"$(round(DecisionTree.macro_weighted_specificity(cm)*100, digits=2))%\t" *
		# "$(round(DecisionTree.mean_PPV(cm)*100, digits=2))%\t" *
		"$(round(DecisionTree.macro_weighted_PPV(cm)*100, digits=2))%\t" *
		# "$(round(DecisionTree.mean_NPV(cm)*100, digits=2))%\t" *
		"$(round(DecisionTree.macro_weighted_NPV(cm)*100, digits=2))%\t"
	end

	go_tree() = begin
		T = 
			if timeit == 0
				build_tree(Y_train, X_train; tree_args..., modal_args..., gammas = gammas_train, rng = rng);
			elseif timeit == 1
				@time build_tree(Y_train, X_train; tree_args..., modal_args..., gammas = gammas_train, rng = rng);
			elseif timeit == 2
				@btime build_tree($Y_train, $X_train; $tree_args..., $modal_args..., gammas = gammas_train, rng = $rng);
			end
		print_tree(T)
		
		println(" test size = $(size(X_test))")
		cm = nothing
		for pruning_purity_threshold in sort(unique([(Float64.(post_pruning_purity_thresholds))...,1.0]))
			println(" Purity threshold $pruning_purity_threshold")
			
			T_pruned = prune_tree(T, pruning_purity_threshold)
			preds = apply_tree(T_pruned, X_test);
			cm = confusion_matrix(Y_test, preds)
			# @test cm.overall_accuracy > 0.99

			println("RESULT:\t$(name)\t$(tree_args)\t$(modal_args)\t$(pruning_purity_threshold)\t$(display_cm_as_row(cm))")
			
			# println("  accuracy: ", round(cm.overall_accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
			for (i,row) in enumerate(eachrow(cm.matrix))
				for val in row
					print(lpad(val,3," "))
				end
				println("  " * "$(round(100*row[i]/sum(row), digits=2))%\t\t" * class_labels[i])
			end

			# println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
		end
		return (T, cm);
	end

	go_forest(f_args) = begin
		F = 
			if timeit == 0
				build_forest(Y_train, X_train; f_args..., modal_args..., gammas = gammas_train, rng = rng);
			elseif timeit == 1
				@time build_forest(Y_train, X_train; f_args..., modal_args..., gammas = gammas_train, rng = rng);
			elseif timeit == 2
				@btime build_forest($Y_train, $X_train; $f_args..., $modal_args..., gammas = gammas_train, rng = $rng);
			end
		print_forest(F)
		
		println(" test size = $(size(X_test))")

		preds = apply_forest(F, X_test);
		cm = confusion_matrix(Y_test, preds)
		# @test cm.overall_accuracy > 0.99

		println("RESULT:\t$(name)\t$(f_args)\t$(modal_args)\t$(display_cm_as_row(cm))")

		# println("  accuracy: ", round(cm.overall_accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
		for (i,row) in enumerate(eachrow(cm.matrix))
			for val in row
				print(lpad(val,3," "))
			end
			println("  " * "$(round(100*row[i]/sum(row), digits=2))%\t\t" * class_labels[i])
		end

		println("Forest OOB Error: $(round.(F.oob_error.*100, digits=2))%")

		return (F, cm);
	end

	go() = begin
		T = nothing
		F = []
		Tcm = nothing
		Fcm = []

		old_logger = global_logger(ConsoleLogger(stderr, debugging_level))
		
		checkpoint_stdout("Computing Tree...")

		T, Tcm = go_tree()

		for (i_forest, f_args) in enumerate(forest_args)
			checkpoint_stdout("Computing Random Forest $(i_forest) / $(length(f_args))...")
			this_F, this_Fcm = go_forest(f_args)
			push!(F, this_F)
			push!(Fcm, this_Fcm)
		end

		global_logger(old_logger);

		T, F, Tcm, Fcm
	end

	if error_catching 
		try
			go()
		catch e
			println("ERROR occurred!")
			println(e)
			return;
		end
	else
			go()
	end
end
