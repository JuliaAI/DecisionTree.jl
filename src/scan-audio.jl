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

scale_dataset = false
# scale_dataset = UInt16

# TODO
# TEST:
# - decision tree
# - RF with:
forest_args = []

for n_trees in [1,50,100]
	for n_subfeatures in [id_f, sqrt_f]
		for n_subrelations in [id_f, half_f, sqrt_f]
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
						)]

forest_runs = 5
optimize_forest_computation = true

results_dir = "./results-audio-scan"

iteration_progress_json_file_path = results_dir * "/progress.json"
concise_output_file_path = results_dir * "/grouped_in_models.csv"
full_output_file_path = results_dir * "/full_columns.csv"
gammas_save_path = results_dir * "/gammas"

column_separator = ";"

exec_dicts = load_or_create_execution_progress_dictionary(
	iteration_progress_json_file_path, exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs
)

# if the output files does not exists initilize them
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

					if done
						println("Iteration already done, skipping...")
						continue
					end
					#####################################################

					# LOAD DATASET
					cur_audio_kwargs = merge(audio_kwargs, (nbands=nbands,))
					dataset, n_pos, n_neg = KDDDataset_not_stratified((n_task,n_version), cur_audio_kwargs; dataset_kwargs...) # , rng = dataset_rng)
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
								debugging_level             =   log_level,
								scale_dataset               =   scale_dataset,
								dataset_slice               =   dataset_slice,
								forest_args                 =   forest_args,
								tree_args                   =   tree_args,
								modal_args                  =   modal_args,
								optimize_forest_computation =   optimize_forest_computation,
								forest_runs                 =   forest_runs,
								gammas_save_path            =   (gammas_save_path, dataset_name_str),
								rng                         =   train_rng
								);
					#####################################################

					# PRINT RESULT IN FILES
					function percent(num::Real; digits=2)
						return round.(num.*100, digits=digits)
					end

					function data_to_string(
							M::Union{DecisionTree.DTree{S, T},DecisionTree.DTNode{S, T}},
							cm::ConfusionMatrix;
							start_s = "(",
							end_s = ")",
							separator = ";"
						) where {S, T}

						result = start_s
						result *= string(percent(cm.kappa), separator)
						result *= string(percent(cm.sensitivities[1]), separator)
						result *= string(percent(cm.specificities[1]), separator)
						result *= string(percent(cm.PPVs[1]), separator)
						result *= string(percent(cm.overall_accuracy))

						if isa(M, DecisionTree.Forest{S, T})
							result *= separator
							result *= string(percent(M.oob_error))
						end

						result *= end_s

						result
					end

					function data_to_string(
							Ms::AbstractVector{DecisionTree.Forest{S, T}},
							cms::AbstractVector{ConfusionMatrix};
							start_s = "(",
							end_s = ")",
							separator = ";"
						) where {S, T}

						result = start_s
						result *= start_s
						result *= string(percent(mean(map(cm->cm.kappa, cms))), separator)
						result *= string(var(map(cm->cm.kappa, cms)))
						result *= end_s * separator
						result *= start_s
						result *= string(percent(mean(map(cm->cm.sensitivities[1], cms))), separator)
						result *= string(var(map(cm->cm.sensitivities[1], cms)))
						result *= end_s * separator
						result *= start_s
						result *= string(percent(mean(map(cm->cm.specificities[1], cms))), separator)
						result *= string(var(map(cm->cm.specificities[1], cms)))
						result *= end_s * separator
						result *= start_s
						result *= string(percent(mean(map(cm->cm.PPVs[1], cms))), separator)
						result *= string(var(map(cm->cm.PPVs[1], cms)))
						result *= end_s * separator
						result *= start_s
						result *= string(percent(mean(map(cm->cm.overall_accuracy, cms))), separator)
						result *= string(var(map(cm->cm.overall_accuracy, cms)))
						result *= end_s * separator
						result *= start_s
						result *= string(percent(mean(map(M->M.oob_error, Ms))), separator)
						result *= string(var(map(M->M.oob_error, Ms)))
						result *= end_s

						result *= end_s

						result
					end

					# PRINT CONCISE
					concise_output_string = string(row_ref, column_separator)
					concise_output_string *= string(data_to_string(T, Tcm; separator=", "), column_separator)
					for j in 1:length(forest_args)
						concise_output_string *= string(data_to_string(Fs[j], Fcms[j]; separator=", "))
						concise_output_string *= string(j == length(forest_args) ? "\n" : column_separator)
					end
					append_in_file(concise_output_file_path, concise_output_string)

					# PRINT FULL
					full_output_string = string(row_ref, column_separator)
					full_output_string *= string(data_to_string(T, Tcm; start_s = "", end_s = ""), column_separator)
					for j in 1:length(forest_args)
						full_output_string *= string(data_to_string(Fs[j], Fcms[j]; start_s = "", end_s = ""))
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

# testDataset("Test", dataset, 0.8, 0, debugging_level=log_level,
# 			forest_args=forest_args, args=args, kwargs=modal_args,
# 			test_tree = true, test_forest = true);

