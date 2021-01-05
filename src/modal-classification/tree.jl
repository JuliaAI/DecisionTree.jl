# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>
using ComputedFieldTypes

module treeclassifier
	include("../util.jl")
	import Random

	export fit

	
	mutable struct NodeMeta{S}
		l           :: NodeMeta{S}      # left child
		r           :: NodeMeta{S}      # right child
		label       :: Label            # most likely label
		feature     :: Int              # feature used for splitting
		threshold   :: S                # threshold value
		# TODO: testsign
		# TODO: modality
		# TODO: S
		is_leaf     :: Bool
		depth       :: Int
		region      :: UnitRange{Int}   # a slice of the samples used to decide the split of the node
		features    :: Vector{Int}      # a list of features not known to be constant
		split_at    :: Int              # index of samples
		function NodeMeta{S}(
				features :: Vector{Int},
				region   :: UnitRange{Int},
				depth    :: Int) where S
			node = new{S}()
			node.depth = depth
			node.region = region
			node.features = features
			node.is_leaf = false
			node
		end
	end

	struct Tree{S, T}
		root   :: NodeMeta{S}
		list   :: Vector{T}
		labels :: Vector{Label}
	end

	@inline setfeature!(i::Integer, Xf::AbstractArray{S where S, 1}, X::OntologicalDataset{T where T,0}, idxs::AbstractVector{Integer}, feature::Integer) ::T = begin
		Xf[i] = getfeature(X, idxs, feature)
	end
	@inline setfeature!(i::Integer, Xf::AbstractArray{S where S, 2}, X::OntologicalDataset{T where T,1}, idxs::AbstractVector{Integer}, feature::Integer) ::AbstractArray{T,2} = begin
		Xf[i,:] = getfeature(X, idxs, feature)
	end
	@inline setfeature!(i::Integer, Xf::AbstractArray{S where S, 3}, X::OntologicalDataset{T where T,2}, idxs::AbstractVector{Integer}, feature::Integer) ::AbstractArray{T,3} = begin
		Xf[i,:,:] = getfeature(X, idxs, feature)
	end

	# find an optimal split satisfying the given constraints
	# (max_depth, min_samples_split, min_purity_increase)
	# TODO dispatch _split! on the learning parameters?
	@computed function _split!(
			X                   :: OntologicalDataset{S, N}, # the ontological dataset
			Y                   :: AbstractVector{Label},    # the label array
			W                   :: AbstractVector{U},        # the weight vector
			
			purity_function     :: Function,
			node                :: NodeMeta{S},              # the node to split
			max_features        :: Int,                      # number of features to consider
			max_depth           :: Int,                      # the maximum depth of the resultant tree
			min_samples_leaf    :: Int,                      # the minimum number of samples each leaf needs to have
			min_samples_split   :: Int,                      # the minimum number of samples in needed for a split
			min_purity_increase :: AbstractFloat,            # minimum purity needed for a split
			
			indX                :: AbstractVector{Int},      # an array of sample indices (we split using samples in indX[node.region])
			
			# The six arrays below are given for optimization purposes
			
			nc                  :: AbstractVector{U},   # nc maintains a dictionary of all labels in the samples
			ncl                 :: AbstractVector{U},   # ncl maintains the counts of labels on the left
			ncr                 :: AbstractVector{U},   # ncr maintains the counts of labels on the right
			
			Xf                  :: AbstractArray{S, N-1},
			Yf                  :: AbstractVector{Label},
			Wf                  :: AbstractVector{U},
			Sf                  :: AbstractVector{Set{X.ontology.worldType}},
			rng                 :: Random.AbstractRNG) where {S, U, N}

		# Region of indx to use to perform the split
		region = node.region
		n_samples = length(region)
		r_start = region.start - 1

		n_classes = length(nc)

		# Class counts
		nc[:] .= zero(U)
		@simd for i in region
			@inbounds nc[Y[indX[i]]] += W[indX[i]]
		end
		nt = sum(nc)
		node.label = argmax(nc) # Assign the most likely label before the split

		# Check leaf conditions
		if (min_samples_leaf * 2 >  n_samples
		 || min_samples_split    >  n_samples
		 || max_depth            <= node.depth
		 || nc[node.label]       == nt)
			node.is_leaf = true
			return
		end

		# Number of non-constants features. TODO only makes sense in the adimensional case?
		features = node.features
		n_features = length(features)

		# Binary relations (= unary modal operators)
		# Note: the equality operator is the first, and is the one representing
		#  the propositional case.
		relations = [ModalLogic.Relation_Eq, subtypes(X.ontology.relation)...]

		# Optimization tracking variables
		best_purity = typemin(U)
		best_feature = -1
		threshold_lo = X[1]
		threshold_hi = X[1]

		indf = 1
		# the number of new constants found during this split
		n_const = 0
		# true if every feature is constant
		unsplittable = true
		# the number of non constant features we will see if
		# only sample n_features used features
		# is a hypergeometric random variable
		total_features = n_features(X)
		# this is the total number of features that we expect to not
		# be one of the known constant features. since we know exactly
		# what the non constant features are, we can sample at 'non_consts_used'
		# non constant features instead of going through every feature randomly.
		non_consts_used = util.hypergeometric(n_features, total_features-n_features, max_features, rng)

		# Find best split (TODO for now, we only handle numerical features)
		
		# For each feature/channel
		@inbounds while (unsplittable || indf <= non_consts_used) && indf <= n_features
			feature = let
				indr = rand(rng, indf:n_features)
				features[indf], features[indr] = features[indr], features[indf]
				features[indf]
			end

			# In the begining, every node is on right of the threshold
			ncl[:] .= zero(U)
			ncr[:] = nc

			# Gather all values for the current feature
			@simd for i in 1:n_samples
				# TODO make this a view? featureview?
				setfeature!(i, Xf, X, indX[i + r_start], feature)
			end

			# Sort [Xf, Yf, Wf, Sf and indX] by Xf
			util.q_bi_sort!(Xf, indX, 1, n_samples, r_start)
			@simd for i in 1:n_samples
				Yf[i] = Y[indX[i + r_start]]
				Wf[i] = W[indX[i + r_start]]
				Sf[i] = Sf[indX[i + r_start]]
			end

			for Rel in relations:
				@info "Testing relation " Rel "..."

				# TODO test this bit...
				# Rel = IA_L
				# S = 

				# Find, for each instance, the highest value for any world,
				#                       and the lowest value for any world
				maxPeaks = fill(typemin(S), n_samples)
				minPeaks = fill(typemax(S), n_samples)
				for i in 1:n_samples
					@info "instance " inst "/" n_samples "..."
					# TODO this findmin/findmax can be made more efficient for intervals.
					for w in enumAcc(Sf[i], Rel, N TODO)
						maxPeaks[i] = max(maxPeaks[i], readMax(w, Xf))
						minPeaks[i] = min(minPeaks[i], readMin(w, Xf))
					end
					@info "maxPeak " maxPeaks[i] "."
					@info "minPeak " minPeaks[i] "."
				end

				thresholdDomain = union(Set(maxPeaks),Set(minPeaks))
				@info "thresholdDomain " thresholdDomain "."

				for t in thresholdDomain
					TODO
				end

				hi = 0
				nl, nr = zero(U), nt
				is_constant = true
				last_f = Xf[1]
				while hi < n_samples
					lo = hi + 1
					curr_f = Xf[lo]
					hi = (lo < n_samples && curr_f == Xf[lo+1]
						? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
						: lo)

					(lo != 1) && (is_constant = false)
					# honor min_samples_leaf
					# if nl >= min_samples_leaf && nr >= min_samples_leaf
					# @assert nl == lo-1,
					# @assert nr == n_samples - (lo-1) == n_samples - lo + 1
					if lo-1 >= min_samples_leaf && n_samples - (lo-1) >= min_samples_leaf
						unsplittable = false
						purity = -(nl * purity_function(ncl, nl)
								 + nr * purity_function(ncr, nr))
						if purity > best_purity && !isapprox(purity, best_purity)
							# will take average at the end
							threshold_lo = last_f
							threshold_hi = curr_f
							@show threshold_lo
							@show threshold_hi
							best_purity  = purity
							best_feature = feature
						end
					end

					# fill ncl and ncr in the direction
					# that would require the smaller number of iterations
					# i.e., hi - lo < n_samples - hi
					if (hi << 1) < n_samples + lo
						@simd for i in lo:hi
							ncr[Yf[i]] -= Wf[i]
						end
					else
						ncr[:] .= zero(U)
						@simd for i in (hi+1):n_samples
							ncr[Yf[i]] += Wf[i]
						end
					end

					nr = zero(U)
					@simd for lab in 1:n_classes
						nr += ncr[lab]
						ncl[lab] = nc[lab] - ncr[lab]
					end

					nl = nt - nr
					last_f = curr_f
				end

				# keep track of constant features to be used later.
				if is_constant
					n_const += 1
					features[indf], features[n_const] = features[n_const], features[indf]
				end
			end

			indf += 1
		end

		# Partition and split according to best_purity and best_feature
		@inbounds if (unsplittable # no splits honor min_samples_leaf
			|| (best_purity / nt + purity_function(nc, nt) < min_purity_increase))
			node.is_leaf = true
			return
		else
			@simd for i in 1:n_samples
				Xf[i] = X[indX[i + r_start], best_feature]
			end

			try
				node.threshold = (threshold_lo + threshold_hi) / 2.0
			catch
				node.threshold = threshold_hi
			end
			# split the samples into two parts: ones that are greater than
			# the threshold and ones that are less than or equal to the threshold
			#                                 ---------------------
			# (so we partition at threshold_lo instead of node.threshold)
			node.split_at = util.partition!(indX, Xf, threshold_lo, region)
			node.feature = best_feature
			node.features = features[(n_const+1):n_features]
		end

		return _split!

	end
	# Split node at a previously-set node.split_at value.
	@inline function fork!(node::NodeMeta{S}) where S
		ind = node.split_at
		region = node.region
		features = node.features
		# no need to copy because we will copy at the end
		node.l = NodeMeta{S}(features, region[    1:ind], node.depth+1)
		node.r = NodeMeta{S}(features, region[ind+1:end], node.depth+1)
	end

	function check_input(
			X                   :: OntologicalDataset{S, N},
			Y                   :: AbstractVector{Label},
			W                   :: AbstractVector{U},
			max_features        :: Int,
			max_depth           :: Int,
			min_samples_leaf    :: Int,
			min_samples_split   :: Int,
			min_purity_increase :: AbstractFloat) where {S, U, N}
		n_samples, n_features = n_samples(X), n_features(X)
		if length(Y) != n_samples
			throw("dimension mismatch between X.domain and Y ($(size(X.domain)) vs $(size(Y))")
		elseif length(W) != n_samples
			throw("dimension mismatch between X.domain and W ($(size(X.domain)) vs $(size(W))")
		elseif max_depth < -1
			throw("unexpected value for max_depth: $(max_depth) (expected:"
				* " max_depth >= 0, or max_depth = -1 for infinite depth)")
		elseif n_features < max_features
			throw("number of features $(n_features) is less than the number "
				* "of max features $(max_features)")
		elseif max_features < 0
			throw("number of features $(max_features) must be >= zero ")
		elseif min_samples_leaf < 1
			throw("min_samples_leaf must be a positive integer "
				* "(given $(min_samples_leaf))")
		elseif min_samples_split < 2
			throw("min_samples_split must be at least 2 "
				* "(given $(min_samples_split))")
		end
	end

	# Xf slices X by across the features dimension. As such, it has one dimension less than X.
	@computed init_Xf(X::OntologicalDataset{S, N}) where S = Array{S, N-1}(undef, n_samples, size(X)[3:end]...)
	# init_Xf(X::OntologicalDataset{S, 2}) where S = Array{S, 1}(undef, n_samples, size(X)[3:end]...)
	# init_Xf(X::OntologicalDataset{S, 3}) where S = Array{S, 2}(undef, n_samples, size(X)[3:end]...)
	# init_Xf(X::OntologicalDataset{S, 4}) where S = Array{S, 3}(undef, n_samples, size(X)[3:end]...)

	function _fit(
			X                       :: OntologicalDataset{S, N},
			Y                       :: AbstractVector{Label},
			W                       :: AbstractVector{U},
			loss                    :: Function,
			n_classes               :: Int,
			max_features            :: Int,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_samples_split       :: Int,
			min_purity_increase     :: AbstractFloat,
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG) where {S, U, N}

		# Dataset sizes
		n_samples, n_features = n_samples(X), n_features(X)

		# Array memory for class counts
		nc  = Array{U}(undef, n_classes)
		ncl = Array{U}(undef, n_classes)
		ncr = Array{U}(undef, n_classes)

		# Array memory for dataset
		Xf = init_Xf(X)
		Yf = Array{Label}(undef, n_samples)
		Wf = Array{U}(undef, n_samples)

		# TODO Perhaps operating with Sets is better
		Sf = [Set([Interval(-1, 0)]) for i in 1:n_sampless]::Array{Set{X.ontology.world},1}

		# Sample indices (array of indices that will be sorted and partitioned across the leaves)
		indX = collect(1:n_samples)
		# Create root node
		root = NodeMeta{S}(collect(1:n_features), 1:n_samples, 0)
		# Stack of nodes to process
		stack = NodeMeta{S}[root]
		@inbounds while length(stack) > 0
			# Pop node and process it
			node = pop!(stack)
			_split!(
				X, Y, W,
				loss, node,
				max_features,
				max_depth,
				min_samples_leaf,
				min_samples_split,
				min_purity_increase,
				indX,
				nc, ncl, ncr, Xf, Yf, Wf, Sf, rng)
			# After processing, if needed, perform the split and push the two children for a later processing step
			# TODO: this step could be parallelized
			if !node.is_leaf
				fork!(node)
				push!(stack, node.r)
				push!(stack, node.l)
			end
		end

		return (root, indX)
	end

	function fit(;
			# In the modal case, dataset instances are Kripke models.
			# In this implementation, we don't accept a generic Kripke model in the explicit form of
			#  a graph; instead, an instance is a dimensional domain (e.g. a matrix or a 3D matrix) onto which
			#  worlds and relations are determined by a given Ontology.
			X                       :: OntologicalDataset{S, N},
			Y                       :: AbstractVector{T},
			W                       :: Union{Nothing, AbstractVector{U}},
			loss = util.entropy     :: Function,
			max_features            :: Int,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_samples_split       :: Int,
			min_purity_increase     :: AbstractFloat,
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG) where {S, T, U, N}

		# Obtain the dataset's "outer size": number of samples and number of features
		n_samples, n_features = n_samples(X), n_features(X)

		# Translate labels to categorical form
		labels, Y_ = util.assign(Y)

		# Use unary weights if no weight is supplied
		if W == nothing
			W = fill(1, n_samples)
		end

		# Check validity of the input
		check_input(
			X, Y, W,
			max_features,
			max_depth,
			min_samples_leaf,
			min_samples_split,
			min_purity_increase)


		# Call core learning function
		root, indX = _fit(
			X, Y_, W,
			loss,
			length(labels),
			max_features,
			max_depth,
			min_samples_leaf,
			min_samples_split,
			min_purity_increase,
			rng)

		return Tree{S, T}(root, labels, indX)
	end
end
