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
################################################################################
################################################################################

# Build models on (multi-dimensional) arrays
function build_stump(
	labels    :: AbstractVector{Label},
	features  :: MatricialDataset{T,D},
	weights   :: Union{Nothing,AbstractVector{U}} = nothing;
	ontology  :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
	kwargs...) where {T, D, U}
	build_stump(OntologicalDataset{T,D-2}(ontology,features), labels, weights; kwargs...)
end

function build_tree(
	labels    :: AbstractVector{Label},
	features  :: MatricialDataset{T,D},
	weights   :: Union{Nothing,AbstractVector{U}} = nothing;
	ontology  :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
	kwargs...) where {T, D, U}
	build_tree(OntologicalDataset{T,D-2}(ontology,features), labels, weights; kwargs...)
end

function build_forest(
	labels    :: AbstractVector{Label},
	features  :: MatricialDataset{T,D};
	# weights   :: Union{Nothing,AbstractVector{U}} = nothing TODO
	ontology  :: Ontology = ModalLogic.getIntervalOntologyOfDim(Val(D-2)),
	kwargs...) where {T, D, U}
	# build_forest(OntologicalDataset{T,D-2}(ontology,features), labels, weights; kwargs...)
	build_forest(OntologicalDataset{T,D-2}(ontology,features), labels; kwargs...)
end

################################################################################
################################################################################
################################################################################

# Build a stump (tree with depth 1)
function build_stump(
		X	  		       :: OntologicalDataset{T, N},
		Y     		     :: AbstractVector{Label},
		W     		     :: Union{Nothing,AbstractVector{U}} = nothing;
		kwargs...) where {T, N, U}
	@assert !haskey(kwargs, :max_depth) || kwargs.max_depth == 1 "build_stump doesn't allow max_depth != 1"
	build_tree(X, Y, W; max_depth = 1, kwargs...)
end

# Build a tree on an OntologicalDataset
function build_tree(
	X                   :: OntologicalDataset{T, N},
	Y                   :: AbstractVector{S},
	W                   :: Union{Nothing,AbstractVector{U}} = nothing;
	gammas              :: Union{GammasType{NTO, Ta},Nothing} = nothing,
	loss                :: Function           = util.entropy,
	n_subfeatures       :: Function           = x -> x,
	max_depth           :: Int                = -1,
	min_samples_leaf    :: Int                = 1,
	min_purity_increase :: AbstractFloat      = 0.0,
	min_loss_at_leaf    :: AbstractFloat      = -Inf,
	n_subrelations      :: Function           = x -> x,
	initCondition       :: _initCondition     = startWithRelationAll,
	useRelationAll      :: Bool               = true,
	useRelationId       :: Bool               = true,
	test_operators      :: AbstractVector{<:ModalLogic.TestOperator}     = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
	rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {T, N, S, U, NTO, Ta}

	if max_depth == -1
		max_depth = typemax(Int)
	end

	rng = mk_rng(rng)
	t = treeclassifier.fit(
		X                   = X,
		Y                   = Y,
		W                   = W,
		gammas              = gammas,
		loss                = loss,
		max_features        = n_subfeatures(n_variables(X)),
		max_depth           = max_depth,
		min_samples_leaf    = min_samples_leaf,
		min_purity_increase = min_purity_increase,
		min_loss_at_leaf    = min_loss_at_leaf,
		max_relations		    = n_subrelations,
		initCondition       = initCondition,
		useRelationAll      = useRelationAll,
		useRelationId       = useRelationId,
		test_operators      = test_operators,
		rng                 = rng)

	root = _convert(t.root, t.list, Y[t.labels])
	# TODO remove This is in order to restore full compatibility with the rest of the package
	if initCondition == startWithRelationAll && X.ontology == ModalLogic.getIntervalOntologyOfDim(Val(N))
		root
	else
		DTree{T, Label}(root, X.ontology.worldType, initCondition)
	end
end

# TODO fix this using specified purity
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

function _empty_tree_leaves(leaf::DTLeaf{T}) where {T}
		DTLeaf{T}(leaf.majority, [])
end

function _empty_tree_leaves(node::DTInternal{S, T}) where {S, T}
	return DTInternal{S, T}(
		node.modality,
		node.featid,
		node.test_operator,
		node.featval,
		_empty_tree_leaves(node.left),
		_empty_tree_leaves(node.right)
	)
end

function _empty_tree_leaves(tree::DTree{S, T}) where {S, T}
	return DTree{S, T}(
		_empty_tree_leaves(tree.root),
		tree.worldType,
		tree.initCondition
	)
end

function update_tree_leaves(leaf::DTLeaf{T}, Xi::MatricialInstance{U,MN}, S::WorldSet{WorldType}, class::T) where {T, U, MN, WorldType<:AbstractWorld}
	return DTLeaf{T}(leaf.majority, val)
end

function update_tree_leaves(node::DTInternal{U, T}, Xi::MatricialInstance{U,MN}, S::WorldSet{WorldType}, class::T) where {U, T, MN, WorldType<:AbstractWorld}
	satisfied = true
	channel = ModalLogic.getInstanceFeature(Xi, node.featid)
	(satisfied,S) = ModalLogic.modalStep(S, node.modality, channel, node.test_operator, node.featval)

	return DTInternal{U, T}(
		node.modality,
		node.featid,
		node.test_operator,
		node.featval,
		satisfied ? update_tree_leaves(node.left, Xi, S, class) : node.left,
		(!satisfied) ? update_tree_leaves(node.right, Xi, S, class) : node.right,
	)
end

function update_tree_leaves(tree::DTree{S, T}, d::Tuple{MatricialDataset{S,D},Vector{CT},Vector{CN}}; reset_leaves = true) where {S, T, D, CT, CN}
	new_tree = tree
	if reset_leaves
		new_tree = _empty_tree_leaves(tree)
	end
	for i in 1:n_samples(d[1])
		w0params =
			if tree.initCondition == startWithRelationAll
				[ModalLogic.emptyWorld]
			elseif tree.initCondition == startAtCenter
				[ModalLogic.centeredWorld, channel_size(d)...]
			elseif typeof(tree.initCondition) <: _startAtWorld
				[tree.initCondition.w]
			end
		new_tree = DTree{S, T}(
			update_tree_leaves(new_tree.root, ModalLogic.getInstance(d[1], i), WorldSet{tree.worldType}([tree.worldType(w0params...)]), d[2][i]),
			tree.worldType,
			tree.initCondition
		)
	end
	return new_tree
end

function update_tree_leaves(tree::DTNode{S, T}, d::Tuple{MatricialDataset{S,D},Vector{CT},Vector{CN}}; reset_leaves = true) where {S, T, D, CT, CN}
	return update_tree_leaves(DTree{S, T}(tree, ModalLogic.getIntervalOntologyOfDim(Val(D-2)).worldType, startWithRelationAll), d, reset_leaves = reset_leaves)
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

function build_forest(
	X                   :: OntologicalDataset{T, N},
	Y                   :: AbstractVector{S}
	;
	# , W                   :: Union{Nothing,AbstractVector{U}} = nothing; TODO these must also be used for the calculation of the oob_error
	# Forest parameters
	n_trees             = 100,
	partial_sampling    = 0.7,      # portion of instances sampled (without replacement) by each tree
	# Tree parameters
	gammas              :: Union{GammasType{NTO, Ta},Nothing} = nothing,
	loss                :: Function           = util.entropy,
	n_subfeatures       :: Function           = x -> ceil(Int, sqrt(x)),
	max_depth           :: Int                = -1,
	min_samples_leaf    :: Int                = 1,
	min_purity_increase :: AbstractFloat      = 0.0,
	min_loss_at_leaf    :: AbstractFloat      = -Inf,
	n_subrelations      :: Function           = x -> x,
	initCondition       :: _initCondition     = startWithRelationAll,
	useRelationAll      :: Bool               = true,
	useRelationId       :: Bool               = true,
	test_operators      :: AbstractVector{<:ModalLogic.TestOperator}     = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
	rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {T, N, S, U, NTO, Ta}

	rng = mk_rng(rng)
	
	if n_trees < 1
		throw("the number of trees must be >= 1")
	end
	
	if !(0.0 < partial_sampling <= 1.0)
		throw("partial_sampling must be in the range (0,1]")
	end
	
	# precompute-gammas, since they are shared by all trees
	if isnothing(gammas)
		(
			# X,
			test_operators, relationSet,
			useRelationId, useRelationAll, 
			relationId_id, relationAll_id,
			availableModalRelation_ids, allAvailableRelation_ids
		) = treeclassifier.optimize_tree_parameters!(X, initCondition, useRelationAll, useRelationId, test_operators)
		gammas = computeGammas(X,X.ontology.worldType,test_operators,relationSet,relationId_id,availableModalRelation_ids)
	end

	t_samples = n_samples(X.domain)
	num_samples = floor(Int, partial_sampling * t_samples)

	trees = Vector{Union{DTree{T,S},DTNode{T,S}}}(undef, n_trees)
	cms = Vector{ConfusionMatrix}(undef, n_trees)
	oob_samples = Vector{Vector{Integer}}(undef, n_trees)

	rngs = [spawn_rng(rng) for i in 1:n_trees]
	Threads.@threads for i in 1:n_trees
		inds = rand(rngs[i], 1:t_samples, num_samples)

		# v_weights = @views W[inds]
		v_labels = @views Y[inds]
		v_features = ModalLogic.sliceDomainByInstances(X.domain, inds; return_view = true)
		v_gammas = sliceGammasByInstances(X.ontology.worldType, gammas, inds; return_view = true)

		trees[i] = build_tree(
			v_labels,
			v_features
			# , v_weights
			;
			ontology             = X.ontology,
			gammas               = v_gammas,
			loss                 = loss,
			n_subfeatures        = n_subfeatures,
			max_depth            = max_depth,
			min_samples_leaf     = min_samples_leaf,
			min_purity_increase  = min_purity_increase,
			min_loss_at_leaf     = min_loss_at_leaf,
			n_subrelations       = n_subrelations,
			initCondition        = initCondition,
			useRelationAll       = useRelationAll,
			useRelationId        = useRelationId,
			test_operators       = test_operators,
			rng                  = rngs[i])

		# grab out-of-bag indices
		oob_samples[i] = setdiff(1:t_samples, inds)

		tree_preds = apply_tree(trees[i], ModalLogic.sliceDomainByInstances(X.domain, oob_samples[i]; return_view = true))
		cms[i] = confusion_matrix(Y[oob_samples[i]], tree_preds)
	end

	oob_classified = Vector{Bool}()
	# For each observation z_i, construct its random forest
	# predictor by averaging (or majority voting) only those 
	# trees corresponding to boot-strap samples in which z_i did not appear.
	Threads.@threads for i in 1:t_samples
		selected_trees = fill(false, n_trees)

		# pick every tree trained without i-th sample
		for j in 1:n_trees
			if i in oob_samples[j] # if i is present in the j-th tree, selecte thi tree
				selected_trees[j] = true
			end
		end
		
		index_of_trees_to_test_with = findall(selected_trees)

		if length(index_of_trees_to_test_with) == 0
			continue
		end

		v_features = ModalLogic.sliceDomainByInstances(X.domain, [i]; return_view = true)
		v_labels = @views Y[[i]]

		# TODO: optimization - no need to pass through ConfusionMatrix
		pred = apply_forest(trees[index_of_trees_to_test_with], v_features)
		cm = confusion_matrix(v_labels, pred)

		push!(oob_classified, cm.overall_accuracy > 0.5)
	end

	oob_error = 1.0 - (length(findall(oob_classified)) / length(oob_classified))

	return Forest{T, S}(trees, cms, oob_error)
end

# use an array of trees to test features
function apply_forest(trees::AbstractVector{Union{DTree{S,T},DTNode{S,T}}}, features::MatricialDataset{S, D}; tree_weights::Union{AbstractVector{N},Nothing} = nothing) where {S, T, D, N<:Real}
	@logmsg DTDetail "apply_forest..."
	n_trees = length(trees)
	n_instances = n_samples(features)

	votes = Matrix{T}(undef, n_trees, n_instances)
	for i in 1:n_trees
		votes[i,:] = apply_tree(trees[i], features)
	end

	predictions = Array{T}(undef, n_instances)
	for i in 1:n_instances
		if T <: Float64
			if isnothing(tree_weights)
				predictions[i] = mean(votes[:,i])
			else
				weighted_votes = Vector{N}()
				for j in 1:length(votes[:,i])
					weighted_votes = votes[j,i] * tree_weights[j]
				end
				predictions[i] = mean(weighted_votes)
			end
		else
			predictions[i] = best_score(votes[:,i], tree_weights)
		end
	end

	return predictions
end

# use a proper forest to test features
function apply_forest(forest::Forest{S,T}, features::MatricialDataset{S,D}; use_weighted_trees::Bool = false) where {S, T, D}
	if use_weighted_trees
		# TODO: choose HOW to weight a tree... overall_accuracy is just an example (maybe can be parameterized)
		apply_forest(forest.trees, features, tree_weights = map(cm -> cm.overall_accuracy, forest.cm))
	else
		apply_forest(forest.trees, features)
	end
end
