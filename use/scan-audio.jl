include("scanner.jl")
include("table-printer.jl")
include("progressive-iterator-manager.jl")

main_rng = DecisionTree.mk_rng(1)

# Optimization arguments for single-tree
tree_args = [
#	(
#		loss = DecisionTree.util.entropy,
#		min_samples_leaf = 1,
#		min_purity_increase = 0.01,
#		min_loss_at_leaf = 0.6,
#	)
]

for loss in [DecisionTree.util.entropy]
	for min_samples_leaf in [1] # [1,2]
		for min_purity_increase in [0.01] # [0.01, 0.001]
			for min_loss_at_leaf in [0.6] # [0.4, 0.6]
				push!(tree_args, 
					(
						loss = loss,
						min_samples_leaf = min_samples_leaf,
						min_purity_increase = min_purity_increase,
						min_loss_at_leaf = min_loss_at_leaf,
					)
				)
			end
		end
	end
end

# Optimization arguments for trees in a forest (no pruning is performed)
forest_tree_args = (
	loss = DecisionTree.util.entropy,
	min_samples_leaf = 1,
	min_purity_increase = 0.0,
	min_loss_at_leaf = 0.0,
)

audio_kwargs_partial_mfcc = (
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

audio_kwargs_full_mfcc = (
	wintime=0.025,
	steptime=0.01,
	numcep=13,
	lifterexp=-22,
	sumpower=false,
	preemph=0.97,
	dither=false,
	minfreq=0.0,
	# maxfreq=sr/2,
	nbands=20,
	bwidth=1.0,
	dcttype=3,
	fbtype=:htkmel,
	usecmp=false,
	modelorder=0
)


modal_args = (
	initCondition = DecisionTree.startWithRelationAll,
	useRelationId = true,
	useRelationAll = false,
	ontology = getIntervalOntologyOfDim(Val(1)),
#	test_operators = [ModalLogic.TestOpGeq_70, ModalLogic.TestOpLeq_70],
	test_operators = [ModalLogic.TestOpGeq_80, ModalLogic.TestOpLeq_80],
	# test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
)


preprocess_wavs = [ noise_gate!, normalize! ]

# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug

timeit = 0
# timeit = 1
# timeit = 2

#round_dataset_to_datatype = Float32
# round_dataset_to_datatype = UInt16
round_dataset_to_datatype = false

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

split_threshold = 0.8

precompute_gammas = true

test_flattened = false

# exec_runs = 1 # 1:5
# exec_n_tasks = 1 # 1:1
# exec_n_versions = 1 # 1:2
# exec_nbands = 20 # [20,40,60]
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
#						),(
#							max_points = 30,
#							ma_size = 25,
#							ma_step = 15,
						)
						]
exec_use_full_mfcc = [false, true]

exec_ranges = [exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs, exec_use_full_mfcc]
exec_ranges_names = ["n_task", "n_version", "nbands", "dataset_kwargs", "use_full_mfcc"]

forest_runs = 5
optimize_forest_computation = true

results_dir = "./results-audio-scan"

iteration_progress_json_file_path = results_dir * "/progress.json"
concise_output_file_path = results_dir * "/grouped_in_models.csv"
full_output_file_path = results_dir * "/full_columns.csv"
gammas_save_path = results_dir * "/gammas"
save_tree_path = results_dir * "/trees"

column_separator = ";"

save_datasets = false
just_produce_datasets_jld = false
saved_datasets_path = results_dir * "/datasets"
mkpath(saved_datasets_path)

if "-f" in ARGS
	if isfile(iteration_progress_json_file_path)
		println("Backing up existing $(iteration_progress_json_file_path)...")
		backup_file_using_creation_date(iteration_progress_json_file_path)
	end
	if isfile(concise_output_file_path)
		println("Backing up existing $(concise_output_file_path)...")
		backup_file_using_creation_date(concise_output_file_path)
	end
	if isfile(full_output_file_path)
		println("Backing up existing $(full_output_file_path)...")
		backup_file_using_creation_date(full_output_file_path)
	end
end

exec_dicts = load_or_create_execution_progress_dictionary(
	iteration_progress_json_file_path, exec_ranges, exec_ranges_names
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
#	TASK 1
	(
		n_version = 1,
		nbands = 40,
		dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50)
	),
	(
		n_version = 1,
		nbands = 60,
		dataset_kwargs = (max_points = 30, ma_size = 75, ma_step = 50)
	),
	# TASK 2
	(
		n_version = 2,
		nbands = 20,
		dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30)
	),
	(
		n_version = 2,
		nbands = 40,
		dataset_kwargs = (max_points = 30, ma_size = 45, ma_step = 30)
	)
]

iteration_blacklist = []

# if the output files does not exists initilize them
print_head(concise_output_file_path, tree_args, forest_args, tree_columns = [""], forest_columns = ["", "σ²"], separator = column_separator)
print_head(full_output_file_path, tree_args, forest_args, separator = column_separator,
	forest_columns = ["K", "sensitivity", "specificity", "precision", "accuracy", "oob_error", "σ² K", "σ² sensitivity", "σ² specificity", "σ² precision", "σ² accuracy", "σ² oob_error"],
)

# a = KDDDataset_not_stratified((1,1), merge(audio_kwargs, (nbands=40,)); exec_dataset_kwargs[1]...)

# Old seeds (TODO: remove)
#run_to_seed = Dict{Number, Number}(
#	1 => 7933197233428195239,
#	2 => 1735640862149943821,
#	3 => 3245434972983169324,
#	4 => 1708661255722239460,
#	5 => 1107158645723204584
#)



# RUN
for i in exec_runs
	dataset_seed = i
	train_seed = i

	println("DATA SEED: $(dataset_seed)")

	for params_combination in IterTools.product(exec_ranges...)
		# Unpack params combination
		params_namedtuple = (zip(map(Symbol, exec_ranges_names), params_combination) |> Dict |> namedtuple)

		# FILTER ITERATIONS
		if (length(iteration_whitelist) > 0 && !is_whitelisted_test(params_namedtuple, iteration_whitelist)) || is_blacklisted_test(params_namedtuple, iteration_blacklist)
			continue
		end
		#####################################################

		# CHECK WHETHER THIS ITERATION WAS ALREADY COMPUTED OR NOT
		done = is_combination_already_computed(exec_dicts, exec_ranges_names, params_combination, i)

		dataset_rng = Random.MersenneTwister(dataset_seed)

		row_ref = string(
			string(dataset_seed),
			["," * replace(string(values(value)), ", " => ",") for value in values(params_namedtuple)]...,
		)

		# Placed here so we can keep track of which iteration is being skipped
		checkpoint_stdout("Computing iteration $(row_ref)...")

		if just_test_filters
			continue
		end
		println(params_combination)

		if done && !just_produce_datasets_jld
			println("Iteration $(params_combination) already done, skipping...")
			continue
		end
		#####################################################
		n_task, n_version, nbands, dataset_kwargs, use_full_mfcc = params_combination

		# LOAD DATASET
		dataset_file_name = saved_datasets_path * "/" * row_ref

		dataset = nothing
		n_pos = nothing
		n_neg = nothing
		
		cur_audio_kwargs = merge(
			if use_full_mfcc
				audio_kwargs_full_mfcc
			else
				audio_kwargs_partial_mfcc
			end
			, (nbands=nbands,))

		dataset = 
			if save_datasets && isfile(dataset_file_name * ".jld")
				if just_produce_datasets_jld
					continue
				end
				checkpoint_stdout("Loading dataset $(dataset_file_name * ".jld")...")
				JLD2.@load (dataset_file_name * ".jld") dataset n_pos n_neg
				(X,Y) = dataset
				if isfile(dataset_file_name * "-balanced.jld")
					JLD2.@load (dataset_file_name * "-balanced.jld") balanced_dataset dataset_slice
				else
					n_per_class = min(n_pos, n_neg)
					dataset_slice = Array{Int,2}(undef, 2, n_per_class)
					dataset_slice[1,:] .=          Random.randperm(dataset_rng, n_pos)[1:n_per_class]
					dataset_slice[2,:] .= n_pos .+ Random.randperm(dataset_rng, n_neg)[1:n_per_class]
					dataset_slice = dataset_slice[:]

					(X_train, Y_train), (X_test, Y_test), _ = traintestsplit(dataset, split_threshold)
					balanced_dataset = (X[dataset_slice], Y[dataset_slice])
					JLD2.@save (dataset_file_name * "-balanced.jld") balanced_dataset dataset_slice
					balanced_train = (X_train, Y_train)
					JLD2.@save (dataset_file_name * "-balanced-train.jld") balanced_train
					balanced_test = (X_test,  Y_test)
					JLD2.@save (dataset_file_name * "-balanced-test.jld")  balanced_test
				end

				dataset
				# Y = 1 .- Y
				# Y = [ (y == 0 ? "yes" : "no") for y in Y]
				dataset = (X,Y)
			else
				checkpoint_stdout("Creating dataset...")
				# TODO wrap dataset creation into a function accepting the rng and other parameters...
				dataset, n_pos, n_neg = KDDDataset_not_stratified((n_task,n_version), cur_audio_kwargs; dataset_kwargs..., preprocess_wavs = preprocess_wavs, use_full_mfcc = use_full_mfcc) # , rng = dataset_rng)
				n_per_class = min(n_pos, n_neg)
				# using Random
				# n_pos = 10
				# n_neg = 15 
				# dataset_rng = Random.MersenneTwister(2)

				dataset_slice = Array{Int,2}(undef, 2, n_per_class)
				dataset_slice[1,:] .=          Random.randperm(dataset_rng, n_pos)[1:n_per_class]
				dataset_slice[2,:] .= n_pos .+ Random.randperm(dataset_rng, n_neg)[1:n_per_class]
				dataset_slice = dataset_slice[:]

				if save_datasets
					checkpoint_stdout("Saving dataset $(dataset_file_name)...")
					(X, Y) = dataset
					JLD2.@save (dataset_file_name * ".jld")                dataset n_pos n_neg
					(X_train, Y_train), (X_test, Y_test), _ = traintestsplit(dataset, split_threshold)
					balanced_dataset = (X[dataset_slice], Y[dataset_slice])
					JLD2.@save (dataset_file_name * "-balanced.jld") balanced_dataset dataset_slice
					balanced_train = (X_train, Y_train)
					JLD2.@save (dataset_file_name * "-balanced-train.jld") balanced_train
					balanced_test = (X_test,  Y_test)
					JLD2.@save (dataset_file_name * "-balanced-test.jld")  balanced_test
					if just_produce_datasets_jld
						continue
					end
				end
				dataset
			end
		# println(dataset_slice)

		#####################################################
		dataset_name_str = string(
			[string(value) * column_separator for value in values(params_namedtuple)]...,
			string(cur_audio_kwargs), column_separator,
			string(dataset_rng.seed)
		)

		# ACTUAL COMPUTATION
		Ts, Fs, Tcms, Fcms = testDataset(
					"($(n_task),$(n_version))", # TODO more verbose name?
					dataset,
					split_threshold,
					log_level                   =   log_level,
					round_dataset_to_datatype   =   round_dataset_to_datatype,
					dataset_slice               =   dataset_slice,
					forest_args                 =   forest_args,
					tree_args                   =   tree_args,
					modal_args                  =   modal_args,
					test_flattened              =   test_flattened,
					precompute_gammas           =   precompute_gammas,
					optimize_forest_computation =   optimize_forest_computation,
					forest_runs                 =   forest_runs,
					gammas_save_path            =   (gammas_save_path, dataset_name_str),
					save_tree_path              =   save_tree_path,
					train_seed                  =   train_seed,
					timeit                      =   timeit
				);
		#####################################################
		# PRINT RESULT IN FILES
		#####################################################

		# PRINT CONCISE
		concise_output_string = string(row_ref, column_separator)
		for j in 1:length(tree_args)
			concise_output_string *= string(data_to_string(Ts[j], Tcms[j]; alt_separator=", ", separator = column_separator))
			concise_output_string *= string(column_separator)
		end
		for j in 1:length(forest_args)
			concise_output_string *= string(data_to_string(Fs[j], Fcms[j]; alt_separator=", ", separator = column_separator))
			concise_output_string *= string(column_separator)
		end
		concise_output_string *= string("\n")
		append_in_file(concise_output_file_path, concise_output_string)

		# PRINT FULL
		full_output_string = string(row_ref, column_separator)
		for j in 1:length(tree_args)
			full_output_string *= string(data_to_string(Ts[j], Tcms[j]; start_s = "", end_s = "", alt_separator = column_separator))
			full_output_string *= string(column_separator)
		end
		for j in 1:length(forest_args)
			full_output_string *= string(data_to_string(Fs[j], Fcms[j]; start_s = "", end_s = "", alt_separator = column_separator))
			full_output_string *= string(column_separator)
		end
		full_output_string *= string("\n")
		append_in_file(full_output_file_path, full_output_string)
		#####################################################

		# ADD THIS STEP TO THE "HISTORY" OF ALREADY COMPUTED ITERATION
		push_seed_to_history!(exec_dicts, exec_ranges_names, params_combination, i)
		export_execution_progress_dictionary(iteration_progress_json_file_path, exec_dicts)
		#####################################################
	end
end

checkpoint_stdout("Finished!")

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

