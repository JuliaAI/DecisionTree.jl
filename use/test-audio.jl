# julia -i -t4 test-audio.jl
# julia -i test-audio.jl
include("test-header.jl")
include("table-printer.jl")
include("progressive-iterator-manager.jl")

rng = my_rng()

tree_args = [(
	loss = DecisionTree.util.entropy,
	min_samples_leaf = 1,
	min_purity_increase = 0.01,
	min_loss_at_leaf = 0.4,
)]

# TODO add parameter: allow relationAll at all levels? Maybe it must be part of the relations... I don't know
modal_args = (
	# n_subrelations = x -> ceil(sqrt(x)),
	# n_subrelations = x -> ceil(x/2),
	initCondition = DecisionTree.startWithRelationAll,
	useRelationId = true,
	useRelationAll = false,
	ontology = getIntervalOntologyOfDim(Val(1)),
	# test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
	test_operators = [ModalLogic.TestOpGeq_70, ModalLogic.TestOpLeq_70],
	# test_operators = [ModalLogic.TestOpGeq_85, ModalLogic.TestOpLeq_85],
	# test_operators = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq, ModalLogic.TestOpGeq_85, ModalLogic.TestOpLeq_85],
	# rng = my_rng,
)

forest_args = [(
	n_subfeatures = sqrt_f,    # with 40 features: [40/10, 40/5, 40/3]
	n_subrelations = sqrt_f,
	n_trees = 5,               # [5,10,20,40,80]
	partial_sampling = 1.0,
	loss = DecisionTree.util.entropy,
	min_samples_leaf = 1,
	min_purity_increase = 0.0,
	min_loss_at_leaf = 0.0,
)]

# log_level = Logging.Warn
log_level = DecisionTree.DTOverview
# log_level = DecisionTree.DTDebug

timeit = 0
# timeit = 2

# rng_i = spawn_rng(rng)

dataset_kwargs = (
	max_points = 2,
	ma_size = 40,   # [100, 50, 20, 10]
	ma_step = 30,   # [ma_size, ma_size*.75, ma_size*.5]
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

n_task = 1
n_version = 2
round_dataset_to_datatype = false
# round_dataset_to_datatype = UInt16
dataset, n_pos, n_neg = KDDDataset_not_stratified((n_task,n_version), audio_kwargs; dataset_kwargs...)
println(dataset[1] |> size)

testDataset("Test",
		dataset,
		0.8;
		log_level          =  log_level,
		round_dataset_to_datatype      =  round_dataset_to_datatype,
		forest_args        =  forest_args,
		tree_args          =  tree_args,
		modal_args         =  modal_args,
		precompute_gammas  =  true
	);

# round_dataset_to_datatype = false

# for i in 1:10
# 	for n_task in 1:1
# 		for n_version in 1:2
# 			dataset = KDDDataset_not_stratified((n_task,n_version), audio_kwargs; dataset_kwargs...)
# 			# (1,1) -> 994

# 			testDataset("($(n_task),$(n_version))", dataset, 0.8, 0,
# 						log_level=log_level,
# 						round_dataset_to_datatype=round_dataset_to_datatype,
# 						forest_args=forest_args,
# 						args=args,
# 						modal_args=modal_args,
# 						test_tree = true,
# 						test_forest = true,
# 						);

# 		end
# 	end
# end


# selected_args = merge(args, (loss = loss,
# 															min_samples_leaf = min_samples_leaf,
# 															min_purity_increase = min_purity_increase,
# 															min_loss_at_leaf = min_loss_at_leaf,
# 															))


# dataset = KDDDataset_not_stratified((1,1), audio_kwargs; dataset_kwargs..., rng = rng) # 110/137 -> 110/110
# dataset = KDDDataset_not_stratified((1,2), audio_kwargs; dataset_kwargs..., rng = rng) # 110/137 -> 110/110
# dataset = KDDDataset_not_stratified((2,1), audio_kwargs; dataset_kwargs..., rng = rng) # 26/8 -> 8/8
# dataset = KDDDataset_not_stratified((2,2), audio_kwargs; dataset_kwargs..., rng = rng) # 46/8 -> 8/8
# dataset = KDDDataset_not_stratified((3,1), audio_kwargs; dataset_kwargs..., rng = rng) # 46/13 -> 13/13
# dataset = KDDDataset_not_stratified((3,2), audio_kwargs; dataset_kwargs..., rng = rng) # 46/13 -> 13/13

# testDataset("Test", dataset, 0.8, 0, log_level=log_level,
# 			forest_args=forest_args, args=args, modal_args=modal_args,
# 			test_tree = true, test_forest = true);

