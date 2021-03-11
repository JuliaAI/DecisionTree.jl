# Utilities

# include("../util.jl")
using .util: Label
using .ModalLogic

include("tree.jl")

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
		node   :: treeclassifier.NodeMeta{S},
		list   :: AbstractVector{T},
		labels :: AbstractVector{T}) where {S<:Real, T<:Label}

	if node.is_leaf
		return DTLeaf{T}(list[node.label], labels[node.region])
	else
		left = _convert(node.l, list, labels)
		right = _convert(node.r, list, labels)
		return DTInternal{S, T}(node.modality, node.feature, node.test_operator, node.threshold, left, right)
	end
end

################################################################################

# TODO include adimensional case? Probably not in the dataset itself
# buildOntologicalDataset(features :: MatricialDataset{T,3}) where {T} =
	# OntologicalDataset{T,1}(ModalLogic.getIntervalOntologyOfDim(Val(1)),features)
# buildOntologicalDataset(features :: MatricialDataset{T,4}) where {T} =
	# OntologicalDataset{T,2}(ModalLogic.getIntervalOntologyOfDim(Val(2)),features)

# Build a stump (tree with depth 1)
function build_stump(
		labels      :: AbstractVector{Label},
		features    :: MatricialDataset{T,D},
		weights     :: Union{Nothing, AbstractVector{U}} = nothing;
		ontology    :: Ontology            = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
		test_operators      :: AbstractVector{<:ModalLogic.TestOperator}     = [ModalLogic.TestOpGeq, ModalLogic.TestOpLes],
		rng         :: Random.AbstractRNG  = Random.GLOBAL_RNG) where {T, U, D}
	
	X = OntologicalDataset{T,D-2}(ontology,features)
	
	t = treeclassifier.fit(
		X                   = X,
		Y                   = labels,
		W                   = weights,
		loss                = treeclassifier.util.zero_one,
		# max_features        = n_variables(X),
		max_depth           = 1,
		min_samples_leaf    = 1,
		min_samples_split   = 2,
		min_purity_increase = 0.0,
		max_purity_split    = 1.0,
		initCondition       = startWithRelationAll,
		useRelationAll      = true,
		useRelationId       = true,
		test_operators      = test_operators,
		rng                 = rng)

	root = _convert(t.root, t.list, labels[t.labels])
	# TODO remove This is in order to mantain compatibility
	if initCondition == startWithRelationAll && ontology == ModalLogic.getIntervalOntologyOfDim(Val(D-2))
		root
	else
		DTree{T, Label}(root, ontology.worldType, initCondition)
	end
end

function build_tree(
		labels              :: AbstractVector{Label},
		features            :: MatricialDataset{T,D},
		# n_subfeatures       :: Int                = 0,
		max_depth           :: Int                = -1,
		min_samples_leaf    :: Int                = 1,
		min_samples_split   :: Int                = 2,
		min_purity_increase :: AbstractFloat      = 0.0,
		max_purity_split    :: AbstractFloat      = 1.0;
		initCondition       :: _initCondition     = startWithRelationAll,
		useRelationAll      :: Bool               = true,
		useRelationId       :: Bool               = true,
		ontology            :: Ontology           = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
		test_operators      :: AbstractVector{<:ModalLogic.TestOperator}     = [ModalLogic.TestOpGeq, ModalLogic.TestOpLes],
		loss                :: Function           = util.entropy,
		rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {T, D}

	# TODO disaccoppia dataset e ontologia di riferimento.
	X = OntologicalDataset{T,D-2}(ontology,features)

	if max_depth == -1
		max_depth = typemax(Int)
	end
	# if n_subfeatures == 0
		# n_subfeatures = n_variables(X)
	# end

	rng = mk_rng(rng)::Random.AbstractRNG
	t = treeclassifier.fit(
		X                   = X,
		Y                   = labels,
		W                   = nothing,
		loss                = loss,
		# max_features        = n_subfeatures,
		max_depth           = max_depth,
		min_samples_leaf    = min_samples_leaf,
		min_samples_split   = min_samples_split,
		min_purity_increase = min_purity_increase,
		max_purity_split    = max_purity_split,
		initCondition       = initCondition,
		useRelationAll      = useRelationAll,
		useRelationId       = useRelationId,
		test_operators      = test_operators,
		rng                 = rng)

	root = _convert(t.root, t.list, labels[t.labels])
	# TODO remove This is in order to mantain compatibility
	if initCondition == startWithRelationAll && ontology == ModalLogic.getIntervalOntologyOfDim(Val(D-2))
		root
	else
		DTree{T, Label}(root, ontology.worldType, initCondition)
	end
end

function prune_tree(tree::DTNode{S, T}, max_purity_threshold::AbstractFloat = 1.0) where {S, T}
	if max_purity_threshold >= 1.0
		return tree
	end
	# Prune the tree once TODO make more efficient (avoid copying so many nodes.)
	function _prune_run(tree::DTNode{S, T}) where {S, T}
		N = length(tree)
		if N == 1        ## a DTLeaf
			return tree
		elseif N == 2    ## a stump
			all_labels = [tree.left.values; tree.right.values]
			majority = majority_vote(all_labels)
			matches = findall(all_labels .== majority)
			purity = length(matches) / length(all_labels)
			if purity >= max_purity_threshold
				return DTLeaf{T}(majority, all_labels)
			else
				return tree
			end
		else
			# TODO also associate an Internal node with values and majority (all_labels, majority)
			return DTInternal{S, T}(tree.modality, tree.featid, tree.test_operator, tree.featval,
						_prune_run(tree.left),
						_prune_run(tree.right))
		end
	end

	# Keep pruning until "convergence"
	pruned = _prune_run(tree)
	while true
		length(pruned) < length(tree) || break
		pruned = _prune_run(tree)
		tree = pruned
	end
	return pruned
end

function prune_tree(tree::DTree{S, T}, max_purity_threshold::AbstractFloat = 1.0) where {S, T}
	DTree{S,T}(prune_tree(tree.root), tree.worldType, tree.initCondition)
end


apply_tree(leaf::DTLeaf{T}, Xi::MatricialInstance{U,MN}, S::WorldSet{WorldType}) where {U, T, MN, WorldType<:AbstractWorld} = leaf.majority

function apply_tree(tree::DTInternal{U, T}, Xi::MatricialInstance{U,MN}, S::WorldSet{WorldType}) where {U, T, MN, WorldType<:AbstractWorld}
	return (
		if tree.featid == 0
			@error " found featid == 0, TODO figure out where does this come from" tree
			# apply_tree(tree.left, X, S)
		else
			@logmsg DTDetail "applying branch..."
			satisfied = true
			channel = ModalLogic.getInstanceFeature(Xi, tree.featid)
			@logmsg DTDetail " S" S
			(satisfied,S) = ModalLogic.modalStep(S, tree.modality, channel, tree.test_operator, tree.featval)
			@logmsg DTDetail " ->(satisfied,S')" satisfied S
			apply_tree((satisfied ? tree.left : tree.right), Xi, S)
		end
	)
end

# Apply tree with initialConditions to a dimensional dataset in matricial form
function apply_tree(tree::DTree{S, T}, d::MatricialDataset{S,D}) where {S, T, D}
	@logmsg DTDetail "apply_tree..."
	n_instances = n_samples(d)
	predictions = Array{T,1}(undef, n_instances)
	for i in 1:n_instances
		@logmsg DTDetail " instance $i/$n_instances"
		# TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code
		w0params =
			if tree.initCondition == startWithRelationAll
				[ModalLogic.emptyWorld]
			elseif tree.initCondition == startAtCenter
				[ModalLogic.centeredWorld, channel_size(d)...]
			elseif typeof(tree.initCondition) <: _startAtWorld
				[tree.initCondition.w]
		end
		predictions[i] = apply_tree(tree.root, ModalLogic.getInstance(d, i), WorldSet{tree.worldType}([tree.worldType(w0params...)]))
	end
	return (if T <: Float64
			Float64.(predictions)
		else
			predictions
		end)
end

# Apply tree to a dimensional dataset in matricial form
function apply_tree(tree::DTNode{S, T}, d::MatricialDataset{S,D}) where {S, T, D}
	apply_tree(DTree{S, T}(tree, ModalLogic.getIntervalOntologyOfDim(Val(D-2)).worldType, startWithRelationAll), d)
end


#=
TODO

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels::AbstractVector{Label}) = Dict(v => k for (k, v) in enumerate(labels))

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::AbstractVector{Label}, votes::AbstractVector{Label}, weights=1.0)
	label2ind = label_index(labels)
	counts = zeros(Float64, length(label2ind))
	for (i, label) in enumerate(votes)
		if isa(weights, Real)
			counts[label2ind[label]] += weights
		else
			counts[label2ind[label]] += weights[i]
		end
	end
	return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::AbstractVector` to each row in X
# and returns a matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::AbstractMatrix)
	N = size(X, 1)
	N_cols = length(row_fun(X[1, :])) # gets the number of columns
	out = Array{Float64}(undef, N, N_cols)
	for i in 1:N
		out[i, :] = row_fun(X[i, :])
	end
	return out
end

"""    apply_tree_proba(::Node, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::DTLeaf{T}, features::AbstractVector{S}, labels) where {S, T} =
	compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::DTInternal{S, T}, features::AbstractVector{S}, labels) where {S, T}
	if tree.featval === nothing
		return apply_tree_proba(tree.left, features, labels)
	elseif eval(Expr(:call, tree.test_operator, features[tree.featid], tree.featval))
		return apply_tree_proba(tree.left, features, labels)
	else
		return apply_tree_proba(tree.right, features, labels)
	end
end

apply_tree_proba(tree::DTNode{S, T}, features::AbstractMatrix{S}, labels) where {S, T} =
	stack_function_results(row->apply_tree_proba(tree, row, labels), features)

=#
