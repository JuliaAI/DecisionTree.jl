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
		return DTInternal{S, T}(node.feature, node.threshold, node.testsign, node.modality, left, right)
	end
end

################################################################################

# TODO include adimensional case? Probably not in the dataset itself
buildOntologicalDataset(features :: AbstractArray{S,3}) where {S} =
	OntologicalDataset{S,1}(IntervalOntology,features)

# Build a stump (tree with depth 1)
function build_stump(
		labels      :: AbstractVector{Label},
		features    :: AbstractArray{S,N},
		weights     :: Union{Nothing, AbstractVector{U}} = nothing;
		ontology    :: Ontology            = ModalLogic.getParallelOntologyOfDim(Val(N-2)),
		rng         :: Random.AbstractRNG  = Random.GLOBAL_RNG) where {S, U, N}
	
	X = OntologicalDataset{S,N-2}(ontology,features)
	
	t = treeclassifier.fit(
		X                   = X,
		Y                   = labels,
		W                   = weights,
		loss                = treeclassifier.util.zero_one,
		max_features        = n_variables(X),
		max_depth           = 1,
		min_samples_leaf    = 1,
		min_samples_split   = 2,
		min_purity_increase = 0.0;
		rng                 = rng)

	return _convert(t.root, t.list, labels[t.labels])
end

function build_tree(
		labels              :: AbstractVector{Label},
		features            :: AbstractArray{S,N},
		n_subfeatures       :: Int                = 0,
		max_depth           :: Int                = -1,
		min_samples_leaf    :: Int                = 1,
		min_samples_split   :: Int                = 2,
		min_purity_increase :: AbstractFloat      = 0.0;
		ontology            :: Ontology           = ModalLogic.getParallelOntologyOfDim(Val(N-2)),
		loss                :: Function           = util.entropy,
		rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {S, N}
	
	# TODO disaccoppia dataset e ontologia di riferimento.
	X = OntologicalDataset{S,N-2}(ontology,features)

	if max_depth == -1
		max_depth = typemax(Int)
	end
	if n_subfeatures == 0
		n_subfeatures = n_variables(X)
	end

	rng = mk_rng(rng)::Random.AbstractRNG
	t = treeclassifier.fit(
		X                   = X,
		Y                   = labels,
		W                   = nothing,
		loss                = loss,
		max_features        = n_subfeatures,
		max_depth           = max_depth,
		min_samples_leaf    = min_samples_leaf,
		min_samples_split   = min_samples_split,
		min_purity_increase = min_purity_increase,
		rng                 = rng)

	return _convert(t.root, t.list, labels[t.labels])
end

function prune_tree(tree::DTNode{S, T}, purity_thresh::AbstractFloat = 1.0) where {S, T}
	if purity_thresh >= 1.0
		return tree
	end
	# Prune the tree once
	# TODO check how the modal pruning should be performed
	function _prune_run(tree::DTNode{S, T}) where {S, T}
		N = length(tree)
		if N == 1        ## a DTLeaf
			return tree
		elseif N == 2    ## a stump
			all_labels = [tree.left.values; tree.right.values]
			majority = majority_vote(all_labels)
			matches = findall(all_labels .== majority)
			purity = length(matches) / length(all_labels)
			if purity >= purity_thresh
				return DTLeaf{T}(majority, all_labels)
			else
				return tree
			end
		else
			return DTInternal{S, T}(tree.featid, tree.featval, tree.testsign, tree.modality,
						_prune_run(tree.left, purity_thresh),
						_prune_run(tree.right, purity_thresh))
		end
	end

	# Keep pruning until "convergence"
	while true
		pruned = _prune_run(tree, purity_thresh)
		length(pruned) < length(tree) || break
		tree = pruned
	end
	return pruned
end


apply_tree(leaf::DTLeaf{T}, Xi::AbstractArray{U,N}, S::AbstractSet{<:AbstractWorld}) where {U, T, N} = leaf.majority

function apply_tree(tree::DTInternal{U, T}, Xi::AbstractArray{U,N}, S::AbstractSet{<:AbstractWorld}) where {U, T, N}
	return (
		if tree.featid == 0
			@error " found featid == 0, TODO figure out where does this come from" tree
			# apply_tree(tree.left, X, S)
		# elseif tree.modality == ModalLogic.RelationEq # TODO actually, no need for this edge case, because enum would return S anyway
		else
			@info "applying branch..."
			satisfied = true
			Xfi = ModalLogic.getslice(Xi, tree.featid)
			@info " S" S
			(satisfied,S) = ModalLogic.modalStep(S, Xfi, tree.modality, tree.featval, Val(false))
			@info " ->S'" S
			if satisfied
				apply_tree(tree.left, Xi, S)
			else
				apply_tree(tree.right, Xi, S)
			end
		end
	)
end

# Apply tree to a dimensional dataset in matricial form
function apply_tree(tree::DTNode{S, T}, features::AbstractArray{S,N}) where {S, T, N}
	@info "apply_tree..."
	# TODO don't create an ontological dataset, there is no need. Instead, attach the ontology to the tree as metadata.
	ontology = IntervalOntology
	X = OntologicalDataset{S,N-2}(ontology,features)
	n_samp = n_samples(X)
	# n_variables(X)
	predictions = Array{T,1}(undef, n_samp)
	for i in 1:n_samp
		@info " instance {$i}/{$n_samp}"
		# TODO figure out: is it better to interpret the whole dataset at once, or instance-by-instance? The first one enables reusing training code
		# attach to the tree the worldType, and use that
		predictions[i] = apply_tree(tree, ModalLogic.getslice(X.domain, i), Set([X.ontology.worldType(ModalLogic.InitialWorld)]))
	end
	return (if T <: Float64
			Float64.(predictions)
		else
			predictions
		end)
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
	elseif eval(Expr(:call, tree.testsign, features[tree.featid], tree.featval))
		return apply_tree_proba(tree.left, features, labels)
	else
		return apply_tree_proba(tree.right, features, labels)
	end
end

apply_tree_proba(tree::DTNode{S, T}, features::AbstractMatrix{S}, labels) where {S, T} =
	stack_function_results(row->apply_tree_proba(tree, row, labels), features)

=#
