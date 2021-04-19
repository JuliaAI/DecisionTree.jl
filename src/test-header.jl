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

abstract type Support end

mutable struct ForestEvaluationSupport <: Support
	f::Union{Nothing,Support,AbstractVector{DecisionTree.Forest{S, Ta}}} where {S, Ta}
	f_args::NamedTuple{T, N} where {T, N}
	cm::Union{Nothing,AbstractVector{ConfusionMatrix}}
	enqueued::Bool
	ForestEvaluationSupport(f_args) = new(nothing, f_args, nothing, false)
end

function will_produce_same_forest_with_different_number_of_trees(f1::ForestEvaluationSupport, f2::ForestEvaluationSupport)
	# TODO: find a smart way to handle this (just not needed for now)
	@assert length(f1.f_args) == length(f2.f_args) "Can't compare two forests with different number of arguments."
	for (k, v1) in zip(keys(f1.f_args), values(f1.f_args))
		# do not compare n_trees
		if k == :n_trees
			continue
		end
		if v1 != f2.f_args[k]
			return false
		end
	end
	true
end


# TODO note that these splitting functions simply cut the dataset in two,
#  and they don't necessarily produce balanced cuts. To produce balanced cuts,
#  one must manually stratify the dataset beforehand
traintestsplit(data::Tuple{MatricialDataset{D,3},AbstractVector{T},AbstractVector{String}},threshold; gammas = nothing, worldType = nothing) where {D,T} = begin
	is_balanced = true
	(X,Y,class_labels) = data
	spl = ceil(Int, length(Y)*threshold)
	# make it even
	if length(class_labels) == 2 && is_balanced
		spl = isodd(spl) ? (spl-1) : spl
	end
	X_train = ModalLogic.sliceDomainByInstances(X, 1:spl)
	Y_train = Y[1:spl]
	gammas_train = 
		if isnothing(gammas) # || isnothing(worldType)
			gammas
		else
			DecisionTree.sliceGammasByInstances(worldType, gammas, 1:spl)
		end
	X_test  = ModalLogic.sliceDomainByInstances(X, spl+1:end)
	Y_test  = Y[spl+1:end]
	# if gammas == nothing
		# (X_train,Y_train),(X_test,Y_test),class_labels
	# else
	(X_train,Y_train),(X_test,Y_test),class_labels,gammas_train
	# end
end

import Dates

function checkpoint_stdout(string::String)
	println("● ", Dates.format(Dates.now(), "[ dd/mm/yyyy HH:MM:SS ] "), string)
	flush(stdout)
end

include("example-datasets.jl")

gammas_saving_task = nothing

function testDataset(
		name                            ::String,
		dataset                         ::Tuple,
		split_threshold                 ::Union{Bool,AbstractFloat};
		log_level                       = DecisionTree.DTOverview,
		scale_dataset                   ::Union{Bool,Type} = false,
		post_pruning_purity_thresholds  = [],
		forest_args                     = [],
		tree_args                       = (),
		modal_args                      = (),
		test_flattened                  = false,
		precompute_gammas               = true,
		optimize_forest_computation     = false,
		forest_runs						= 1,
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
			(modal_args, nothing, modal_args.ontology.worldType)
		else
			haskey(modal_args, :ontology) || error("testDataset: precompute_gammas=1 requires `ontology` field in modal_args: $(modal_args)")

			X_all = OntologicalDataset{eltype(X_all_d),ndims(X_all_d)-2}(modal_args.ontology,X_all_d)

			worldType = modal_args.ontology.worldType

			old_logger = global_logger(ConsoleLogger(stderr, log_level))
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
					checkpoint_stdout("Computing gammas...")
					gammas = DecisionTree.computeGammas(X_all,worldType,test_operators,relationSet,relationId_id,availableModalRelation_ids)
					if !isnothing(gammas_jld_path)
						checkpoint_stdout("Saving gammas to file \"$(gammas_jld_path)\" (size: $(sizeof(gammas)) bytes)...")

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
	# global_logger(ConsoleLogger(stderr, log_level))
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

	go_forest(f_args; prebuilt_model::Union{Nothing,AbstractVector{DecisionTree.Forest{S, T}}} = nothing) where {S,T} = begin
		Fs = 
			if isnothing(prebuilt_model)
				[
					if timeit == 0
						build_forest(Y_train, X_train; f_args..., modal_args..., gammas = gammas_train, rng = rng);
					elseif timeit == 1
						@time build_forest(Y_train, X_train; f_args..., modal_args..., gammas = gammas_train, rng = rng);
					elseif timeit == 2
						@btime build_forest($Y_train, $X_train; $f_args..., $modal_args..., gammas = gammas_train, rng = $rng);
					end
					for i in 1:forest_runs
				]
			else
				println("Using slice of a prebuilt forest.")
				# !!! HUGE PROBLEM HERE !!! #
				# BUG: can't compute oob_error of a forest built slicing another forest!!!
				forests::Vector{DecisionTree.Forest{S, T}} = []
				for f in prebuilt_model
					v_forest = @views f.trees[Random.randperm(rng, length(f.trees))[1:f_args.n_trees]]
					v_cms = @views f.cm[Random.randperm(rng, length(f.cm))[1:f_args.n_trees]]
					push!(forests, DecisionTree.Forest{S, T}(v_forest, v_cms, 0.0))
				end
				forests
			end

		for F in Fs
			print_forest(F)
		end
		
		cms = []
		for F in Fs
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

			push!(cms, cm)
		end

		return (Fs, cms);
	end

	go() = begin
		T = nothing
		Fs = []
		Tcm = nothing
		Fcms = []

		old_logger = global_logger(ConsoleLogger(stderr, log_level))
		
		checkpoint_stdout("Computing Tree...")

		T, Tcm = go_tree()

		# # TODO
		# if test_flattened == true
		# 	T, Tcm = go_flattened_tree()
		# 	# Flatten 
		# 	(X_train,Y_train), (X_test,Y_test), class_labels = dataset
		# 	X_train = ...
		# 	X_test = ...
		# 	dataset = (X_train,Y_train), (X_test,Y_test), class_labels
		# end

		if optimize_forest_computation
			# initialize support structures
			forest_supports_user_order = Vector{ForestEvaluationSupport}(undef, length(forest_args)) # ordered as user gave them
			for (i_forest, f_args) in enumerate(forest_args)
				forest_supports_user_order[i_forest] = ForestEvaluationSupport(f_args)
			end

			# biggest forest first
			forest_supports_build_order = Vector{ForestEvaluationSupport}() # ordered with biggest forest first
			append!(forest_supports_build_order, forest_supports_user_order)
			sort!(forest_supports_build_order, by = f -> f.f_args.n_trees, rev = true)

			for (i, f) in enumerate(forest_supports_build_order)
				if f.enqueued
					continue
				end

				f.enqueued = true

				for j in i:length(forest_supports_build_order)
					if forest_supports_build_order[j].enqueued
						continue
					end

					if will_produce_same_forest_with_different_number_of_trees(f, forest_supports_build_order[j])
						# reference the forest of the "equivalent" Support structor
						# equivalent means has the same parameters except for the n_trees
						forest_supports_build_order[j].f = forest_supports_build_order[i]
						forest_supports_build_order[j].enqueued = true
					end
				end
			end

			for (i, f) in enumerate(forest_supports_build_order)
				checkpoint_stdout("Computing Random Forest $(i) / $(length(forest_supports_build_order))...")
				model = f

				while isa(model, ForestEvaluationSupport)
					model = model.f
				end

				forest_supports_build_order[i].f, forest_supports_build_order[i].cm = go_forest(f.f_args, prebuilt_model = model)
			end

			# put resulting forests in vector in the order the user gave them
			for (i, f) in enumerate(forest_supports_user_order)
				@assert f.f isa AbstractVector{DecisionTree.Forest{S, T}} where {S,T} "This is not a Vector of Forests! eltype = $(eltype(f.f))"
				@assert f.cm isa AbstractVector{ConfusionMatrix} "This is not a Vector of ConfusionMatrix!"
				@assert length(f.f) == forest_runs "There is a support struct with less than $(forest_runs) forests: $(length(f.f))"
				@assert length(f.cm) == forest_runs "There is a support struct with less than $(forest_runs) confusion matrices: $(length(f.cm))"
				@assert f.f_args == forest_args[i] "f_args mismatch! $(f.f_args) == $(f_args[i])"

				push!(Fs, f.f)
				push!(Fcms, f.cm)
			end
		else
			for (i_forest, f_args) in enumerate(forest_args)
				checkpoint_stdout("Computing Random Forest $(i_forest) / $(length(forest_args))...")
				this_F, this_Fcm = go_forest(f_args)
				push!(Fs, this_F)
				push!(Fcms, this_Fcm)
			end
		end

		global_logger(old_logger);

		T, Fs, Tcm, Fcms
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
