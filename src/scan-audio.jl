include("test-header.jl")

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

# RUN
for i in 1:10
	rng = spawn_rng(main_rng)
	# TASK
	for n_task in 1:1
		for n_version in 1:2
			# DATASET
			for nbands in [20,40,60]
				for dataset_kwargs in [(
		max_points = 30,
		ma_size = 75,
		ma_step = 50,
	),(
		max_points = 30,
		ma_size = 25,
		ma_step = 15,
	),(
		max_points = 30,
		ma_size = 45,
		ma_step = 30,
	)]
					cur_audio_kwargs = merge(audio_kwargs, (nbands=nbands,))
					dataset = KDDDataset((n_task,n_version), cur_audio_kwargs; dataset_kwargs..., rng = rng)

					# TODO
					# TEST:
					# - decision tree
					# - RF with:
							# for n_trees in [1,50,100]
								# for n_subfeatures in [0,nbands]
									# for n_subrelations in [x->x, x->ceil(x/2), x->ceil(sqrt(x))]

					# cur_modal_args = merge(modal_args, (
						# n_trees=n_trees,
						# n_subfeatures=n_subfeatures,))
					
					# forest_args = (
					# 	n_subfeatures = n_subfeatures,
					# 	n_trees = n_trees,
					# 	partial_sampling = 1.0,
					# 	n_subrelations=n_subrelations,
					# ...forest_tree_args
					# )

					# nfreqs

					# testDataset("($(n_task),$(n_version))", dataset, 0.8, 0,
					# 			debugging_level=log_level,
					# 			scale_dataset=scale_dataset,
					# 			forest_args=forest_args,
					# 			args=args,
					# 			kwargs=cur_modal_args,
					# 			test_tree = true,
					# 			test_forest = true,
					# 			);

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


# dataset = KDDDataset((1,1), audio_kwargs; dataset_kwargs..., rng = rng) # 110/137 -> 110/110
# dataset = KDDDataset((1,2), audio_kwargs; dataset_kwargs..., rng = rng) # 110/137 -> 110/110
# dataset = KDDDataset((2,1), audio_kwargs; dataset_kwargs..., rng = rng) # 26/8 -> 8/8
# dataset = KDDDataset((2,2), audio_kwargs; dataset_kwargs..., rng = rng) # 46/8 -> 8/8
# dataset = KDDDataset((3,1), audio_kwargs; dataset_kwargs..., rng = rng) # 46/13 -> 13/13
# dataset = KDDDataset((3,2), audio_kwargs; dataset_kwargs..., rng = rng) # 46/13 -> 13/13

# testDataset("Test", dataset, 0.8, 0, debugging_level=log_level,
# 			forest_args=forest_args, args=args, kwargs=modal_args,
# 			test_tree = true, test_forest = true);

