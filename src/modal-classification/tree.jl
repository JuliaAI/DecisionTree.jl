# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>
using ComputedFieldTypes

module treeclassifier
	include("../util.jl")
	import Random

	export fit

	
	mutable struct NodeMeta{S} # {S,U}
		features    :: Vector{Int}      # a list of features
		region      :: UnitRange{Int}   # a slice of the samples used to decide the split of the node
		# worlds      :: AbstractVector{AbstractSet{AbstractWorld}} # current set of worlds for each training instance
		depth       :: Int
		modal_depth :: Int
		is_leaf     :: Bool             # whether this is a leaf node, or a split one
		label       :: Label            # most likely label
		# split properties
		split_at    :: Int              # index of samples
		l           :: NodeMeta{S}      # left child
		r           :: NodeMeta{S}      # right child
		# purity      :: U                # purity grade attained if this is a split
		modality    :: AbstractRelation # modal operator (e.g. RelationEQ for the propositional case)
		feature     :: Int              # feature used for splitting
		threshold   :: S                # threshold value
		# TODO: testsign
		function NodeMeta{S}(
				features    :: Vector{Int},
				region      :: UnitRange{Int},
				depth       :: Int,
				modal_depth :: Int
				) where S
			node = new{S}()
			node.features = features
			node.region = region
			node.depth = depth
			node.modal_depth = modal_depth
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
			X                   :: OntologicalDataset{T, N}, # the ontological dataset
			Y                   :: AbstractVector{Label},    # the label array
			W                   :: AbstractVector{U},        # the weight vector
			S                   :: AbstractVector{Set{X.ontology.worldType}}, # the vector of current worlds
			
			purity_function     :: Function,
			node                :: NodeMeta{T},              # the node to split
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
			
			Xf                  :: AbstractArray{T, N-1},
			Yf                  :: AbstractVector{Label},
			Wf                  :: AbstractVector{U},
			# Sf                  :: AbstractVector{Set{X.ontology.worldType}},
			rng                 :: Random.AbstractRNG) where {T, U, N}

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

		# Gather all values needed for the current set of instances
		@simd for i in 1:n_samples
			Yf[i] = Y[indX[i + r_start]]
			Wf[i] = W[indX[i + r_start]]
			# Sf[i] = S[indX[i + r_start]]
		end

		# Feature ids and number of features
		features = node.features
		n_features = length(features)

		# Binary relations (= unary modal operators)
		# Note: the equality operator is the first, and is the one representing
		#  the propositional case.
		relations = [ModalLogic.RelationEq, subtypes(X.ontology.relationType)...]

		# Optimization tracking variables
		best_purity = typemin(U)
		best_relation = ModalLogic.RelationNone
		best_feature = -1
		best_threshold = T(-1)
		# threshold_lo = X[1]
		# threshold_hi = X[1]

		# true if every feature is constant
		unsplittable = true
		# the number of non constant features we will see if
		# only sample n_features used features
		# is a hypergeometric random variable
		# this is the total number of features that we expect to not
		# be one of the known constant features. since we know exactly
		# what the non constant features are, we can sample at 'non_consts_used'
		# non constant features instead of going through every feature randomly.
		non_consts_used = util.hypergeometric(n_features, n_variables(X)-n_features, max_features, rng)

		# Find best split (TODO for now, we only handle numerical features)
		
		# For each feature/variable/channel
		indf = 1
		@inbounds while (unsplittable || indf <= non_consts_used) && indf <= n_features
			feature = let
				indr = rand(rng, indf:n_features)
				features[indf], features[indr] = features[indr], features[indf]
				f = features[indf]
				indf += 1
				f
			end

			# Gather all values needed for the current feature
			@simd for i in 1:n_samples
				# TODO make this a view? featureview?
				setfeature!(i, Xf, X, indX[i + r_start], feature)
			end

			## Test all conditions
			# For each relational operator
			for relation in relations:
				@info "Testing relation " relation "..."

				# Find, for each instance, the highest value for any world
				#                       and the lowest value for any world
				maxPeaks = fill(typemin(T), n_samples)
				minPeaks = fill(typemax(T), n_samples)
				for i in 1:n_samples
					@info " instance " i "/" n_samples "..."
					# TODO this findmin/findmax can be made more efficient for intervals.
					for w in enumAcc(S[indX[i + r_start]], relation, Xf[i]) # Sf[i]
						maxPeaks[i] = max(maxPeaks[i], ModalLogic.WMax(w, Xf[i]))
						minPeaks[i] = min(minPeaks[i], ModalLogic.WMin(w, Xf[i]))
					end
					@info "  maxPeak " maxPeaks[i] "."
					@info "  minPeak " minPeaks[i] "."
				end

				# Obtain the list of reasonable thresholds
				thresholdDomain = union(Set(maxPeaks),Set(minPeaks))
				@info "thresholdDomain " thresholdDomain "."

				# Look for thresholds 'a' for the proposition "feature <= a"
				# TODO do the same with "feature > a" (but avoid for relation = Eq because then the case is redundant)
				for threshold in thresholdDomain
					@info " threshold " threshold "..."

					# Re-initialize right class counts
					nr = zero(U)
					ncr[:] .= zero(U)
					for i in 1:n_samples
						@info "  instance " i "/" n_samples
						@info "   peaks (" minPeaks[i] "/" maxPeaks[i] ")"
						satisfied = true
						if maxPeaks[i] <= threshold
							# This is definitely a nl (Sf[i] makes a modal step)
							@info "   " "YES!!!"
						elseif minPeaks[i] > threshold
							# This is definitely a nr (Sf[i] stays the same)
							@info "   " "NO!!!"
							satisfied = false
						else
							@info "   must manually check worlds."
							worlds = enumAcc(S[indX[i + r_start]], relation, Xf[i])
							# The check makes sure that the existence of a world prevents a from being vacuously true
							if length(collect(Iterators.take(worlds, 1))) > 1
								satisfied = false
								for w in worlds # Sf[i]
									if(ModalLogic.WLeq(w, Xf[i], threshold)) # WLeq is <=
										satisfied = true
										break
									end
								end
							end
						end
						@info "   " (satisfied ? "YES" : "NO")
						if !satisfied
							nr += Wf[i]
							ncr[Yf[i]] += Wf[i]
						end
					end

					# Calculate left class counts
					@simd for lab in 1:n_classes # TODO something like @simd ncl .= nc - ncr instead
						ncl[lab] = nc[lab] - ncr[lab]
					end
					nl = nt - nr
					@info " (nl,nr) = (" nl "," nr ")\n"

					# Honor min_samples_leaf
					if nl >= min_samples_leaf && n_samples - nl >= min_samples_leaf
						unsplittable = false
						purity = -(nl * purity_function(ncl, nl) +
							      	 nr * purity_function(ncr, nr))
						@info " purity = " purity
						if purity > best_purity && !isapprox(purity, best_purity)
							best_purity    = purity
							best_relation  = relation
							best_feature   = feature
							best_threshold = threshold
							# TODO: At the end, we should take the average between current and last.
							#  This requires thresholds to be sorted
							# threshold_lo, threshold_hi  = last_f, curr_f
							@info " new optimum:"
							@info " best_purity = " best_purity
							@info " " best_relation ", " best_feature ", " best_threshold
							# @info threshold_lo, threshold_hi
						end
					end
				end # for threshold
			end # for relation
		end # while feature

		# If the split is good, partition and split according to the optimum
		@inbounds if (unsplittable # no splits honor min_samples_leaf
			|| (best_purity / nt + purity_function(nc, nt) < min_purity_increase))
			node.is_leaf = true
			return
		else
			# try
			# 	node.threshold = (threshold_lo + threshold_hi) / 2.0
			# catch
			# 	node.threshold = threshold_hi
			# end

			# split the samples into two parts:
			# - ones that are > threshold
			# - ones that are <= threshold

			# node.purity    = best_purity
			node.modality  = best_relation
			node.feature   = best_feature
			node.threshold = best_threshold

			# Compute new world sets (= make a modal step)
			@simd for i in 1:n_samples
				setfeature!(i, Xf, X, indX[i + r_start], best_feature)
			end
			# TODO instead of using memory, here, just use two opposite indices and perform substitutions. indj = n_samples
			satisfied_flags = Array{Int,1}(1, n_samples)
			for i in 1:n_samples
				worlds = enumAcc(S[indX[i + r_start]], best_relation, Xf[i])
				if length(collect(Iterators.take(worlds, 1))) > 1
					# TODO maybe it's better to use an array and then create a set with = Set(worlds)
					new_worlds = Set{X.ontology.worldType,1}(undef, 0)
					for w in worlds # Sf[i]
						if(ModalLogic.WLeq(w, Xf[i], best_threshold)) # WLeq is <=
							push!(new_worlds, w)
						end
					end
					S[indX[i + r_start]] = new_worlds
				else
					satisfied_flags[i] = 0
				end
			end

			node.split_at = partition!(indX, satisfied_flags, 0, region)
			# For debug:
			# indX = rand(1:10, 10)
			# satisfied_flags = rand([1,0], 10)
			# partition!(indX, satisfied_flags, 0, 1:10)
			
			# Sort [Xf, Yf, Wf, Sf and indX] by Xf
			# util.q_bi_sort!(satisfied_flags, indX, 1, n_samples, r_start)
			# node.split_at = searchsortedfirst(satisfied_flags, true)
			# TODO... node.split_at = util.partition!(indX, Xf, node.threshold, region)
			# TODO Sort indX[region], similarly to util.q_bi_sort!(Xf, indX, 1, n_samples, r_start)
			
			# TODO no need for a full array for each node, in the dimensional case? 
			# if so, then the non-const thing doesn't make sense!
			node.features = features[1:n_features]
		end
	end
	# Split node at a previously-set node.split_at value.
	# The children inherits some of the data
	@inline function fork!(node::NodeMeta{S}) where S
		ind = node.split_at
		region = node.region
		features = node.features
		depth = node.depth+1
		mdepth = (node.modality == ModalLogic.RelationNone ? node.modal_depth : node.modal_depth+1)
		# no need to copy because we will copy at the end
		node.l = NodeMeta{S}(features, region[    1:ind], depth, mdepth)
		node.r = NodeMeta{S}(features, region[ind+1:end], depth, mdepth)
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
		n_samples, n_features = n_samples(X), n_variables(X)
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
		n_samples, n_features = n_samples(X), n_variables(X)

		# Array memory for class counts
		# TODO transform all of these Array{somthing,1} into Vector's (aesthetic changeX)
		nc  = Array{U,1}(undef, n_classes)
		ncl = Array{U,1}(undef, n_classes)
		ncr = Array{U,1}(undef, n_classes)

		# TODO We need to write on S, thus it cannot be a static array like X Y and W;
		# Should belong inside each meta-node and then be copied? That's a waste of space(for each instance),
		# We only need the worlds for the currentinstance set.
		# What if it's not fixed size? Maybe it should be like the subset of indX[region], so that indX[region.start] is parallel to node.S[1]
		# TODO make the initial entity and initial modality a training parameter. Probably, the first modality (modal_depth=0) should be Exist... (= All worlds). Create the allWorlds enumerator
		S = [Set([X.ontology.worldType(-1, 0)]) for i in 1:n_samples]

		# Array memory for dataset
		Xf = init_Xf(X)
		Yf = Array{Label,1}(undef, n_samples)
		Wf = Array{U,1}(undef, n_samples)
		# TODO Perhaps operating with Sets is better
		# Sf = Array{Set{X.ontology.worldType},1}(undef, n_samples)

		# Sample indices (array of indices that will be sorted and partitioned across the leaves)
		indX = collect(1:n_samples)
		# Create root node
		root = NodeMeta{S}(collect(1:n_features), 1:n_samples, 0, 0)
		# Stack of nodes to process
		stack = NodeMeta{S}[root]
		@inbounds while length(stack) > 0
			# Pop node and process it
			node = pop!(stack)
			_split!(
				X, Y, W, S
				loss, node,
				max_features,
				max_depth,
				min_samples_leaf,
				min_samples_split,
				min_purity_increase,
				indX,
				nc, ncl, ncr, Xf, Yf, Wf, # Sf, 
				rng)
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
		n_samples, n_features = n_samples(X), n_variables(X)

		# Translate labels to categorical form
		labels, Y_ = util.assign(Y)

		# Use unary weights if no weight is supplied
		if W == nothing
			# TODO optimize w in the case of all-ones: write a subtype of AbstractVector:
			#  AllOnesVector, so that getindex(W, i) = 1 and sum(W) = size(W).
			#  This allows the compiler to optimize constants at compile-time
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
