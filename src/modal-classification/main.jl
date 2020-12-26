# Utilities

include("tree.jl")

# Conversion: NodeMeta (node + training info) -> DTNode (bare decision tree model)
function _convert(
		node   :: treeclassifier.NodeMeta{S},
		list   :: AbstractVector{Label},
		labels :: AbstractVector{Label}) where {S}

	if node.is_leaf
		return DTLeaf{T}(list[node.label], labels[node.region])
	else
		left = _convert(node.l, list, labels)
		right = _convert(node.r, list, labels)
		return DTInternal{S, T}(node.feature, node.threshold, node.testsign, node.modality, left, right)
	end
end

################################################################################

# Build a stump (tree with depth 1)
function build_stump(
		labels      :: AbstractVector{Label},
		features    :: AbstractMatrix{S},
		weights     :: Union{Nothing, AbstractVector{U}} = nothing;
		rng         :: Random.AbstractRNG                = Random.GLOBAL_RNG) where {S}

	t = treeclassifier.fit(
		X                   = features,
		Y                   = labels,
		W                   = weights,
		loss                = treeclassifier.util.zero_one,
		max_features        = size(features, 2),
		max_depth           = 1,
		min_samples_leaf    = 1,
		min_samples_split   = 2,
		min_purity_increase = 0.0;
		rng                 = rng)

	return _convert(t.root, t.list, labels[t.labels])
end

function build_tree(
		labels              :: AbstractVector{Label},
		features            :: AbstractMatrix{S},
		n_subfeatures       :: Int                = 0,
		max_depth           :: Int                = -1,
		min_samples_leaf    :: Int                = 1,
		min_samples_split   :: Int                = 2,
		min_purity_increase :: AbstractFloat      = 0.0;
		loss                :: Function           = util.entropy,
		rng                 :: Random.AbstractRNG = Random.GLOBAL_RNG) where {S}

	if max_depth == -1
		max_depth = typemax(Int)
	end
	if n_subfeatures == 0
		n_subfeatures = size(features, 2)
	end

	rng = mk_rng(rng)::Random.AbstractRNG
	t = treeclassifier.fit(
		X                   = features,
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
			return DTInternal{S, T}(tree.featid, tree.featval, tree.testsign, tree.modality
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


apply_tree(leaf::DTLeaf{T}, feature::AbstractVector{S}) where {S, T} = leaf.majority

function apply_tree(tree::DTInternal{S, T}, features::AbstractVector{S}) where {S, T}
	# TODO: use tree.modality
	return (if tree.featid == 0
		apply_tree(tree.left, features)
	elseif eval(Expr(:call, tree.testsign, features[tree.featid], tree.featval))
		apply_tree(tree.left, features)
	else
		apply_tree(tree.right, features)
	end)
end

function apply_tree(tree::DTNode{S, T}, features::AbstractMatrix{S}) where {S, T}
	N = size(features,1)
	predictions = Array{T}(undef, N)
	for i in 1:N
		predictions[i] = apply_tree(tree, features[i, :])
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
