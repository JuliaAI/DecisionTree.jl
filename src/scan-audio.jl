include("test-header.jl")
include("table-printer.jl")
include("progressive-iterator-manager.jl")

main_rng = DecisionTree.mk_rng(1)

# Optimization arguments for single-tree
tree_args = (
	loss = DecisionTree.util.entropy,
	min_samples_leaf = 1,
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

audio_kwargs = (
	wintime = 0.025, # in ms          # 0.020-0.040
	steptime = 0.010, # in ms         # 0.010-0.015
	fbtype = :mel,                    # [:mel, :htkmel, :fcmel]
	window_f = DSP.hamming, # [DSP.hamming, (nwin)->DSP.tukey(nwin, 0.25)]
	pre_emphasis = 0.97,              # any, 0 (no pre_emphasis)
	nbands = 40,                      # any, (also try 20)
	sumpower = false,                 # [false, true]
	dither = false,                   # [false, true]
	# bwidth = 1.0,                   # 
	# minfreq = 0.0,
	# maxfreq = (sr)->(sr/2),
	# usecmp = false,
)

modal_args = (
	initCondition = DecisionTree.startWithRelationAll,
	useRelationId = true,
	useRelationAll = false,
	ontology = getIntervalOntologyOfDim(Val(1)),
	test_operators = [ModalLogic.TestOpGeq_70, ModalLogic.TestOpLeq_70],
)


# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug

timeit = 0
# timeit = 2

#scale_dataset = Float32
# scale_dataset = UInt16
scale_dataset = false

# TODO
# TEST:
# - decision tree
# - RF with:
forest_args = []

for n_trees in [1,50,100]
	for n_subfeatures in [id_f, sqrt_f]
		for n_subrelations in [id_f, sqrt_f]
			push!(forest_args, (
				n_subfeatures       = n_subfeatures,
				n_trees             = n_trees,
				partial_sampling    = 1.0,
				n_subrelations      = n_subrelations,
				forest_tree_args...
			))
		end
	end
end
# nfreqs

precompute_gammas = true

test_flattened = false

exec_runs = 1:5
exec_n_tasks = 1:1
exec_n_versions = 1:2
exec_nbands = [20,40,60]
exec_dataset_kwargs =   [(
							max_points = 30,
							ma_size = 75,
							ma_step = 50,
						),(
							max_points = 30,
							ma_size = 45,
							ma_step = 30,
						),(
							max_points = 30,
							ma_size = 25,
							ma_step = 15,
						)
						]

forest_runs = 5
optimize_forest_computation = true

results_dir = "./results-audio-scan"

iteration_progress_json_file_path = results_dir * "/progress.json"
concise_output_file_path = results_dir * "/grouped_in_models.csv"
full_output_file_path = results_dir * "/full_columns.csv"
gammas_save_path = results_dir * "/gammas"

column_separator = ";"

import JLD2
save_datasets = true
just_produce_datasets_jld = false
saved_datasets_path = results_dir * "/datasets"
mkpath(saved_datasets_path)

exec_dicts = load_or_create_execution_progress_dictionary(
	iteration_progress_json_file_path, exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs
)


just_test_filters = false
iteration_whitelist = [
	# FOR TESTING
#	(
#		n_version = 1,
#		nbands = 20,
#		dataset_kwargs = (max_points = 2, ma_size = 75, ma_step = 50),
#	),
#	(
#		n_version = 1,
#		nbands = 20,
#		dataset_kwargs = (max_points = 5, ma_size = 75, ma_step = 50),
#	),
#	(
#		n_version = 1,
#		nbands = 20,
#		dataset_kwargs = (max_points = 10, ma_size = 75, ma_step = 50),
#	),
	# TASK 1
	(
		n_version = 1,
		nbands = 40,
		dataset_kwargs = (max_points = 30, ma_size = 25, ma_step = 15)
	),
	(
		n_version = 1,
		nbands = 60,
		dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50)
	),
	(
		n_version = 1,
		nbands = 60,
		dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30)
	),
	# TASK 2
	(
		n_version = 2,
		nbands = 20,
		dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50)
	),
	(
		n_version = 2,
		nbands = 20,
		dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30)
	),
	(
		n_version = 2,
		nbands = 40,
		dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50)
	),
	(
		n_version = 2,
		nbands = 40,
		dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30)
	),
]

iteration_blacklist = []

if "-f" in ARGS
	if isfile(iteration_progress_json_file_path)
		println("Removing old $(iteration_progress_json_file_path)...")
		rm(iteration_progress_json_file_path)
	end
	if isfile(concise_output_file_path)
		println("Removing old $(concise_output_file_path)...")
		rm(concise_output_file_path)
	end
	if isfile(full_output_file_path)
		println("Removing old $(full_output_file_path)...")
		rm(full_output_file_path)
	end
end

# if the output files does not exists initilize them
if ! isfile(concise_output_file_path)
	concise_output_file = open(concise_output_file_path, "a+")
	print_head(concise_output_file, tree_args, forest_args, tree_columns = [""], forest_columns = ["", "σ²"], separator = column_separator)
	close(concise_output_file)
end
if ! isfile(full_output_file_path)
	full_output_file = open(full_output_file_path, "a+")
	print_head(full_output_file, tree_args, forest_args, separator = column_separator,
		forest_columns = ["K", "sensitivity", "specificity", "precision", "accuracy", "oob_error", "σ² K", "σ² sensitivity", "σ² specificity", "σ² precision", "σ² accuracy", "σ² oob_error"],
	)
	close(full_output_file)
end

# a = KDDDataset_not_stratified((1,1), merge(audio_kwargs, (nbands=40,)); exec_dataset_kwargs[1]...)

# RUN
for i in exec_runs
	rng = spawn_rng(main_rng)
	println("SEED: " * string(Int64.(rng.seed)))
	dataset_seed = abs(rand(rng, Int))
	# TASK
	for n_task in exec_n_tasks
		for n_version in exec_n_versions
			# DATASET
			for nbands in exec_nbands
				for dataset_kwargs in exec_dataset_kwargs
					# FILTER ITERATIONS
					test_parameters = (
						n_task = n_task,
						n_version = n_version,
						nbands = nbands,
						dataset_kwargs = dataset_kwargs,
					)
					if !is_whitelisted_test(test_parameters, iteration_whitelist) || is_blacklisted_test(test_parameters, iteration_blacklist)
						continue
					end
					#####################################################

					# CHECK WHETHER THIS ITERATION WAS ALREADY COMPUTED OR NOT
					done = false
					for dict in exec_dicts
						if (
							dict["n_task"] == n_task &&
							dict["n_version"] == n_version &&
							dict["nbands"] == nbands &&
							is_same_kwargs(dict["dataset_kwargs"], dataset_kwargs) &&
							i in dict["runs"]
							)
							done = true
						end
					end

					dataset_rng = Random.MersenneTwister(dataset_seed)
					train_rng = spawn_rng(rng)

					row_ref = string(
						string(string(dataset_seed), ",",
						string(n_task), ",",
						string(n_version), ",",
						string(nbands), ",",
						string(values(dataset_kwargs)))
					)

					# Placed here so we can keep track of which iteration is being skipped
					checkpoint_stdout("Computing iteration $(row_ref)...")

					if just_test_filters
						continue
					end

					if done
						println("Iteration already done, skipping...")
						continue
					end
					#####################################################

					# LOAD DATASET
					dataset_file_safe_name = string(
						string(string(dataset_seed), ".",
						string(n_task), ".",
						string(n_version), ".",
						string(nbands), ".",
						replace(string(values(dataset_kwargs)), ", " => ".")), ".jld"
					)
					dataset_file_name = saved_datasets_path * "/" * dataset_file_safe_name

					dataset = nothing
					n_pos = nothing
					n_neg = nothing
					cur_audio_kwargs = merge(audio_kwargs, (nbands=nbands,))
					if save_datasets && isfile(dataset_file_name)
						checkpoint_stdout("Loading dataset $(dataset_file_name)...")
						JLD2.@load dataset_file_name dataset n_pos n_neg
						if just_produce_datasets_jld
							continue
						end
					else
						dataset, n_pos, n_neg = KDDDataset_not_stratified((n_task,n_version), cur_audio_kwargs; dataset_kwargs...) # , rng = dataset_rng)
						if save_datasets
							checkpoint_stdout("Saving dataset $(dataset_file_name)...")
							JLD2.@save dataset_file_name dataset n_pos n_neg
							if just_produce_datasets_jld
								continue
							end
						end
					end
					n_per_class = min(n_pos, n_neg)
					# using Random
					# n_pos = 10
					# n_neg = 15 
					# dataset_rng = Random.MersenneTwister(2)

					dataset_slice = Array{Int,2}(undef, 2, n_per_class)
					dataset_slice[1,:] .=          Random.randperm(dataset_rng, n_pos)[1:n_per_class]
					dataset_slice[2,:] .= n_pos .+ Random.randperm(dataset_rng, n_neg)[1:n_per_class]
					dataset_slice = dataset_slice[:]
					# println(dataset_slice)

					#####################################################
					dataset_name_str = string(
						string(n_task), column_separator,
						string(n_version), column_separator,
						string(cur_audio_kwargs), column_separator,
						string(dataset_kwargs), column_separator,
						string(dataset_rng.seed)
					)

					# ACTUAL COMPUTATION
					T, Fs, Tcm, Fcms = testDataset(
								"($(n_task),$(n_version))",
								dataset,
								0.8,
								log_level                   =   log_level,
								scale_dataset               =   scale_dataset,
								dataset_slice               =   dataset_slice,
								forest_args                 =   forest_args,
								tree_args                   =   tree_args,
								modal_args                  =   modal_args,
								test_flattened              =   test_flattened,
								precompute_gammas           =   precompute_gammas,
								optimize_forest_computation =   optimize_forest_computation,
								forest_runs                 =   forest_runs,
								gammas_save_path            =   (gammas_save_path, dataset_name_str),
								rng                         =   train_rng
								);
					#####################################################
					# PRINT RESULT IN FILES
					#####################################################

					# PRINT CONCISE
					concise_output_string = string(row_ref, column_separator)
					concise_output_string *= string(data_to_string(T, Tcm; separator=", "), column_separator)
					for j in 1:length(forest_args)
						concise_output_string *= string(data_to_string(Fs[j], Fcms[j]; alt_separator=", ", separator = column_separator))
						concise_output_string *= string(j == length(forest_args) ? "\n" : column_separator)
					end
					append_in_file(concise_output_file_path, concise_output_string)

					# PRINT FULL
					full_output_string = string(row_ref, column_separator)
					full_output_string *= string(data_to_string(T, Tcm; start_s = "", end_s = ""), column_separator)
					for j in 1:length(forest_args)
						full_output_string *= string(data_to_string(Fs[j], Fcms[j]; start_s = "", end_s = "", alt_separator = column_separator))
						full_output_string *= string(j == length(forest_args) ? "\n" : column_separator)
					end
					append_in_file(full_output_file_path, full_output_string)
					#####################################################

					# ADD THIS STEP TO THE "HISTORY" OF ALREADY COMPUTED ITERATION
					for dict in exec_dicts
						if (
							dict["n_task"] == n_task &&
							dict["n_version"] == n_version &&
							dict["nbands"] == nbands &&
							is_same_kwargs(dict["dataset_kwargs"], dataset_kwargs)
							)
							#println("Iteration completed.")
							push!(dict["runs"], i)
						end
					end
					export_execution_progress_dictionary(iteration_progress_json_file_path, exec_dicts)
					#####################################################
				end
			end
		end
	end
end

# selected_args = merge(args, (loss = loss,
# 															min_samples_leaf = min_samples_leaf,
# 															min_purity_increase = min_purity_increase,
# 															min_loss_at_leaf = min_loss_at_leaf,
# 															))

# dataset_kwargs = (
# 	max_points = -1,
# 	ma_size = 1,
# 	ma_step = 1,
# )
# 
# dataset = KDDDataset_not_stratified((1,1), audio_kwargs; dataset_kwargs..., rng = main_rng); # 141/298
# dataset[1] |> size # (1413, 282)
# dataset = KDDDataset_not_stratified((1,2), audio_kwargs; dataset_kwargs..., rng = main_rng); # 141/298
# dataset[1] |> size # (2997, 282)
# dataset = KDDDataset_not_stratified((2,1), audio_kwargs; dataset_kwargs..., rng = main_rng); # 54/32
# dataset[1] |> size # (1413, 64)
# dataset = KDDDataset_not_stratified((2,2), audio_kwargs; dataset_kwargs..., rng = main_rng); # 54/32
# dataset[1] |> size # (2997, 64)
# dataset = KDDDataset_not_stratified((3,1), audio_kwargs; dataset_kwargs..., rng = main_rng); # 54/20
# dataset[1] |> size # (1413, 40)
# dataset = KDDDataset_not_stratified((3,2), audio_kwargs; dataset_kwargs..., rng = main_rng); # 54/20
# dataset[1] |> size # (2673, 40)

# testDataset("Test", dataset, 0.8, 0, log_level=log_level,
# 			forest_args=forest_args, args=args, kwargs=modal_args,
# 			test_tree = true, test_forest = true);

