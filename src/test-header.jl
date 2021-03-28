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

function testDataset(name::String, dataset::Tuple, split_threshold::Union{Bool,AbstractFloat}, timeit::Integer = 2; debugging_level = DecisionTree.DTOverview, scale_dataset::Union{Bool,Type} = false, post_pruning_purity_thresholds = [], args = (), kwargs = (), error_catching = false, rng = my_rng())
	println("Benchmarking dataset '$name'...")
	global_logger(ConsoleLogger(stderr, Logging.Warn));
	if split_threshold != false
		length(dataset) == 3 || error("Wrong dataset length: $(length(dataset))")
		if scale_dataset != false
			dataset = scaleDataset(dataset, scale_dataset)
		end
		(X_train, Y_train), (X_test, Y_test),class_labels = traintestsplit(dataset, split_threshold)
	else
		length(dataset) == 3 || error("Wrong dataset length: $(length(dataset))")
		(X_train, Y_train), (X_test, Y_test),class_labels = dataset 
		if scale_dataset != false
			(X_train, Y_train, class_labels) = scaleDataset((X_train, Y_train, class_labels))
			(X_test, Y_test, class_labels) = scaleDataset((X_test, Y_test, class_labels))
		end
	end
	println("args = ", args)
	println("kwargs = ", kwargs)

	# println(" n_samples = $(size(X_train)[end-1])")
	println(" train size = $(size(X_train))")
	# global_logger(ConsoleLogger(stderr, Logging.Info))
	global_logger(ConsoleLogger(stderr, debugging_level))
	# global_logger(ConsoleLogger(stderr, DecisionTree.DTDebug))
	
	go() = begin
		if timeit == 0
			T = build_tree(Y_train, X_train; args..., kwargs..., rng = rng);
		elseif timeit == 1
			T = @time build_tree(Y_train, X_train; args..., kwargs..., rng = rng);
		elseif timeit == 2
			T = @btime build_tree($Y_train, $X_train; $args..., $kwargs..., rng = $rng);
		end
		print(T)
		
		println(" test size = $(size(X_test))")
		for pruning_purity_threshold in sort(unique([(Float64.(post_pruning_purity_thresholds))...,1.0]))
			println(" Purity threshold $pruning_purity_threshold")
			
			global_logger(ConsoleLogger(stderr, Logging.Warn));

			T_pruned = prune_tree(T, pruning_purity_threshold)
			preds = apply_tree(T_pruned, X_test);
			cm = confusion_matrix(Y_test, preds)
			# @test cm.overall_accuracy > 0.99
			
			function display_cm_as_row(cm::ConfusionMatrix)
				"|\t" *
				"$(round(cm.overall_accuracy*100, digits=2))%\t" *
				"$(round(cm.mean_accuracy*100, digits=2))%\t" *
				"$(round(cm.kappa*100, digits=2))%\t" *
				# "$(round(DecisionTree.macro_F1(cm)*100, digits=2))%\t" *
				"$(round.(cm.accuracies.*100, digits=2))%\t" *
				"$(round.(cm.F1s.*100, digits=2))%\t" *
				"$(round.(cm.sensitivities.*100, digits=2))%\t" *
				"$(round.(cm.specificities.*100, digits=2))%\t" *
				"$(round.(cm.PPVs.*100, digits=2))%\t" *
				"$(round.(cm.NPVs.*100, digits=2))%\t" *
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

			println("RESULT:\t$(name)\t$(args)\t$(kwargs)\t$(pruning_purity_threshold)\t$(display_cm_as_row(cm))")
			
			# println("  accuracy: ", round(cm.overall_accuracy*100, digits=2), "% kappa: ", round(cm.kappa*100, digits=2), "% ")
			for (i,row) in enumerate(eachrow(cm.matrix))
				for val in row
					print(lpad(val,3," "))
				end
				println("  " * "$(round(100*row[i]/sum(row), digits=2))%\t\t" * class_labels[i])
			end

			global_logger(ConsoleLogger(stderr, Logging.Info));

			# println("nodes: ($(num_nodes(T_pruned)), height: $(height(T_pruned)))")
		end
		return T;
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
