include("test-header.jl")
include("table-printer.jl")
include("progressive-iterator-manager.jl")

main_rng = DecisionTree.mk_rng(1)

dataset_rng = spawn_rng(main_rng)

train_seed = abs(rand(main_rng,Int))

################################################################################
################################################################################
################################################################################

results_dir = "./results-siemens"

iteration_progress_json_file_path = results_dir * "/progress.json"
concise_output_file_path = results_dir * "/grouped_in_models.csv"
full_output_file_path = results_dir * "/full_columns.csv"
gammas_save_path = results_dir * "/gammas"

column_separator = ";"

################################################################################
################################################################################
################################################################################

modal_args = (
	initCondition = DecisionTree.startWithRelationAll,
	useRelationId = true,
	useRelationAll = false,
	ontology = getIntervalOntologyOfDim(Val(1)),
	test_operators = [ModalLogic.TestOpGeq_70, ModalLogic.TestOpLeq_70],
)

################################################################################
################################################################################
################################################################################

# Optimization arguments for single-tree
# tree_args = (
# 	loss = DecisionTree.util.entropy,
# 	min_samples_leaf = 1,
# 	min_purity_increase = 0.0,
# 	min_loss_at_leaf = 0.0,
# )
tree_args = (
	loss = DecisionTree.util.entropy,
	min_samples_leaf = 2,
	min_purity_increase = 0.01,
	min_loss_at_leaf = 0.4,
)

# Optimization arguments for trees in a forest (no pruning is performed)
forest_tree_args = (
	loss = DecisionTree.util.entropy,
	min_samples_leaf = 1,
	min_purity_increase = 0.0,
	min_loss_at_leaf = 0.0,
)


forest_args = []

# for n_trees in [50,100]
# 	for n_subfeatures in [sqrt_f] # [id_f, sqrt_f]
# 		for n_subrelations in [sqrt_f] # [id_f, half_f, sqrt_f]
# 			push!(forest_args, (
# 				n_subfeatures       = n_subfeatures,
# 				n_trees             = n_trees,
# 				partial_sampling    = 1.0,
# 				n_subrelations      = n_subrelations,
# 				forest_tree_args...
# 			))
# 		end
# 	end
# end

# TODO
# dataset_kwargs = (,
# )

################################################################################
################################################################################
################################################################################

# TODO gather into kwargs

split_threshold = 0.8
# split_threshold = false

error_catching = false

precompute_gammas = true
# precompute_gammas = false

# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug

test_flattened = false

timeit = 0
# timeit = 2

# scale_dataset = false
# scale_dataset = UInt16
scale_dataset = Float32

post_pruning_purity_thresholds = []

################################################################################
################################################################################
################################################################################


exec_runs = 1:10
exec_nmeans = [5, 10, 15]
exec_hour = 1:2
exec_distance = -2:-1:-24

exec_ranges = [exec_nmeans, exec_hour, exec_distance]
exec_ranges_names = ["nmeans", "hour", "distance"]
exec_dicts = load_or_create_execution_progress_dictionary(
	iteration_progress_json_file_path, exec_ranges, exec_ranges_names
)

# if the output files does not exist initialize it
if ! isfile(concise_output_file_path)
	concise_output_file = open(concise_output_file_path, "a+")
	print_head(concise_output_file, tree_args, forest_args, tree_columns = [""], forest_columns = [""], separator = column_separator)
	close(concise_output_file)
end
if ! isfile(full_output_file_path)
	full_output_file = open(full_output_file_path, "a+")
	print_head(full_output_file, tree_args, forest_args, separator = column_separator)
	close(full_output_file)
end

################################################################################
################################################################################
################################################################################

# RUN
for i in exec_runs
	
	dataset_seed = abs(rand(dataset_rng,Int))

	println("DATA SEED: $(dataset_seed)")
	
	for params_combination in IterTools.product(exec_ranges...)

		# Unpack params combination
		nmeans, hour, distance = params_combination

		dataset_rng = Random.MersenneTwister(dataset_seed)
		train_rng   = Random.MersenneTwister(train_seed)

		############################################################################
		# CHECK WHETHER THIS ITERATION WAS ALREADY COMPUTED OR NOT
		done = false
		for dict in exec_dicts
			if getindex.([dict], exec_ranges_names) == collect(params_combination) &&
					i in dict["runs"]
				done = true
				break
			end
		end
		if done
			println("Iteration $(params_combination) already done, skipping...")
			continue
		end
		############################################################################

		# LOAD DATASET
		# (X_train,Y_train), (X_test,Y_test), class_labels  = SiemensDataset_not_stratified(nmeans, hour, distance, , subdir = "Siemens-2")
		dataset, n_pos, n_neg = SiemensDataset_not_stratified(nmeans, hour, distance, subdir = "Siemens-2") # params_combination...
		# ... dataset_kwargs, rng = dataset_rng)
		n_per_class = min(n_pos, n_neg)
		(X_train,Y_train), (X_test,Y_test), class_labels = dataset

		dataset_slice = Array{Int,2}(undef, 2, n_per_class)
		dataset_slice[1,:] .=          Random.randperm(dataset_rng, n_pos)[1:n_per_class]
		dataset_slice[2,:] .= n_pos .+ Random.randperm(dataset_rng, n_neg)[1:n_per_class]
		dataset_slice = vec(dataset_slice)
		# println(dataset_slice)
		
		# println(size(X_train))
		# println(size(X_test))
		# println(size(Y_train))
		# println(size(Y_test))
		
		# readline()
		
		############################################################################
		dataset_name_str = string(
			string(i), column_separator,
			string(nmeans), column_separator,
			string(hour), column_separator,
			string(distance), column_separator, # TODO use params_combination
			# string(dataset_rng.seed)
		)

		# ACTUAL COMPUTATION
		T, F, Tcm, Fcm = testDataset(
				"$(params_combination)",
				dataset,
				split_threshold,
				log_level                       = log_level,
				scale_dataset                   = scale_dataset,
				post_pruning_purity_thresholds  = post_pruning_purity_thresholds,
				forest_args                     = forest_args,
				tree_args                       = tree_args,
				modal_args                      = modal_args,
				test_flattened                  = test_flattened,
				precompute_gammas               = precompute_gammas,
				gammas_save_path                = (gammas_save_path, dataset_name_str),
				dataset_slice                   = dataset_slice,
				error_catching                  = error_catching,
				rng                             = train_rng,
				timeit                          = timeit,
			);
		############################################################################

		# PRINT RESULT IN FILES
		row_ref = string(
			string(i), ",",
			string(nmeans), ",",
			string(hour), ",",
			string(distance),
		)

		# PRINT CONCISE
		concise_output_string = string(row_ref, column_separator)
		concise_output_string *= string(data_to_string(T, Tcm; separator=", "), column_separator)
		for j in 1:length(forest_args)
			concise_output_string *= string(data_to_string(F[j], Fcm[j]; separator=", "))
			concise_output_string *= string(column_separator)
		end
		concise_output_string *= string("\n")
		append_in_file(concise_output_file_path, concise_output_string)

		# PRINT FULL
		full_output_string = string(row_ref, column_separator)
		full_output_string *= string(data_to_string(T, Tcm; start_s = "", end_s = ""), column_separator)
		for j in 1:length(forest_args)
			full_output_string *= string(data_to_string(F[j], Fcm[j]; start_s = "", end_s = ""))
			full_output_string *= string(column_separator)
		end
		full_output_string *= string("\n")
		append_in_file(full_output_file_path, full_output_string)

		############################################################################
		# ADD THIS STEP TO THE "HISTORY" OF ALREADY COMPUTED ITERATION
		for dict in exec_dicts
			if getindex.([dict], exec_ranges_names) == collect(params_combination)
				#println("Iteration completed.")
				push!(dict["runs"], i)
			end
		end
		export_execution_progress_dictionary(iteration_progress_json_file_path, exec_dicts)
		############################################################################
	end
end
