# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>

module treeclassifier
	
	export fit

	using ..ModalLogic
	using ..DecisionTree
	using DecisionTree.util
	using BenchmarkTools
	using Logging: @logmsg
	import Random

	mutable struct NodeMeta{S<:Real} # {S,U}
		features         :: Vector{Int}      # a list of features
		region           :: UnitRange{Int}   # a slice of the samples used to decide the split of the node
		# worlds      :: AbstractVector{WorldSet{W}} # current set of worlds for each training instance
		depth            :: Int
		modal_depth      :: Int
		is_leaf          :: Bool             # whether this is a leaf node, or a split one
		label            :: Label            # most likely label
		# split properties
		split_at         :: Int              # index of samples
		l                :: NodeMeta{S}      # left child
		r                :: NodeMeta{S}      # right child
		purity           :: AbstractFloat        # purity grade attained if this is a split; TODO generalize to weight type `U`
		modality         :: R where R<:AbstractRelation # modal operator (e.g. RelationId for the propositional case)
		feature          :: Int                      # feature used for splitting
		test_operator    :: ModalLogic.TestOperator  # test_operator (e.g. <=)
		threshold        :: S                        # threshold value
		function NodeMeta{S}(
				region      :: UnitRange{Int},
				depth       :: Int,
				modal_depth :: Int
				) where S<:Real
			node = new{S}()
			node.region = region
			node.depth = depth
			node.modal_depth = modal_depth
			node.is_leaf = false
			node
		end
	end

	struct Tree{S, T}
		root           :: NodeMeta{S}
		list           :: Vector{T}
		labels         :: Vector{Label}
		initCondition  :: DecisionTree._initCondition
	end

	@inline setfeature!(i::Integer, ud::MatricialUniDataset{T,1}, d::MatricialDataset{T,2}, idx::Integer, feature::Integer) where T = begin
		@views ud[i] = ModalLogic.getFeature(d, idx, feature) # ::T
	end
	@inline setfeature!(i::Integer, ud::MatricialUniDataset{T,2}, d::MatricialDataset{T,3}, idx::Integer, feature::Integer) where T = begin
		@views ud[:,i] = ModalLogic.getFeature(d, idx, feature) # ::AbstractArray{T,2}
	end
	@inline setfeature!(i::Integer, ud::MatricialUniDataset{T,3}, d::MatricialDataset{T,4}, idx::Integer, feature::Integer) where T = begin
		@views ud[:,:,i] = ModalLogic.getFeature(d, idx, feature) # ::AbstractArray{T,3}
	end
	# TODO:
	# @inline setfeature!(i::Integer, Xf::AbstractArray{T, M}, X::OntologicalDataset{T,N}, idx::Integer, feature::Integer) where {T,N,M} = begin
		# Xf[i,[(:) for i in 1:N]...] = ModalLogic.getFeature(X, idx, feature)
	# end

	# find an optimal split satisfying the given constraints
	# (e.g. max_depth, min_samples_leaf, min_purity_increase)
	# TODO not using max_features, rng (which is still useful e.g. rand(rng, 1:10)) anymore
	# TODO move this function inside the caller function, and get rid of all parameters
	function _split!(
							X                   :: OntologicalDataset{T, N}, # the ontological dataset
							Y                   :: AbstractVector{Label},    # the label array
							W                   :: AbstractVector{U},        # the weight vector
							S                   :: AbstractVector{WorldSet{WorldType}}, # the vector of current worlds (TODO AbstractVector{<:AbstractSet{X.ontology.worldType}})
							
							loss_function       :: Function,
							node                :: NodeMeta{T},              # the node to split
							max_depth           :: Int,                      # the maximum depth of the resultant tree
							min_samples_leaf    :: Int,                      # the minimum number of samples each leaf needs to have
							min_loss_at_leaf    :: AbstractFloat,            # maximum purity allowed on a leaf
							min_purity_increase :: AbstractFloat,            # minimum purity increase needed for a split
							test_operators      :: AbstractVector{<:ModalLogic.TestOperator},
							
							indX                :: AbstractVector{Int},      # an array of sample indices (we split using samples in indX[node.region])
							
							# The six arrays below are given for optimization purposes
							
							nc                  :: AbstractVector{U},   # nc maintains a dictionary of all labels in the samples
							ncl                 :: AbstractVector{U},   # ncl maintains the counts of labels on the left
							ncr                 :: AbstractVector{U},   # ncr maintains the counts of labels on the right
							
							Xf                  :: MatricialUniDataset{T, M},
							Yf                  :: AbstractVector{Label},
							Wf                  :: AbstractVector{U},
							Sf                  :: AbstractVector{WorldSet{WorldType}},
							# Gammas            :: AbstractVector{<:AbstractDict{<:ModalLogic.AbstractRelation,<:AbstractVector{<:AbstractDict{WorldType,NTuple{NTO,T}}}}},
							# Gammas            :: TODO Union with AbstractArray{<:AbstractDict{WorldType,NTuple{NTO,T}},3},
							Gammas            :: AbstractArray{NTuple{NTO,T},L},
							# TODO Ef                  :: AbstractArray{T},
							
							rng                 :: Random.AbstractRNG,
							relationSet         :: Vector{<:ModalLogic.AbstractRelation},
							relation_ids        :: AbstractVector{Int},
							) where {WorldType<:AbstractWorld, T, U, N, M, NTO, L}  # WT<:X.ontology.worldType

		# Region of indx to use to perform the split
		region = node.region
		n_instances = length(region)
		r_start = region.start - 1

		# Class counts
		nc[:] .= zero(U)
		@simd for i in region
			@inbounds nc[Y[indX[i]]] += W[indX[i]]
		end
		nt = sum(nc)
		node.label = argmax(nc) # Assign the most likely label before the split

		# @logmsg DTOverview "node purity min_loss_at_leaf " loss_function(nc, nt) min_loss_at_leaf

		@logmsg DTDebug "_split!(...) " n_instances region nt

		# Check leaf conditions
		if ((min_samples_leaf * 2 >  n_instances)
		 || (nc[node.label]       == nt)
		 || (loss_function(nc, nt)  <= min_loss_at_leaf)
		 || (max_depth            <= node.depth))
			node.is_leaf = true
			@logmsg DTDetail "leaf created: " (min_samples_leaf * 2 >  n_instances) (nc[node.label] == nt) (loss_function(nc, nt)  <= min_loss_at_leaf) (max_depth <= node.depth)
			return
		end

		# Gather all values needed for the current set of instances
		@simd for i in 1:n_instances
			Yf[i] = Y[indX[i + r_start]]
			Wf[i] = W[indX[i + r_start]]
			Sf[i] = S[indX[i + r_start]]
		end

		# Optimization tracking variables
		best_purity__nt = typemin(U)
		best_relation = ModalLogic.RelationNone
		best_feature = -1
		best_test_operator = ModalLogic.TestOpNone
		best_threshold = T(-1)
		best_nl = -1 # TODO this is just for checking
		best_unsatisfied = [] # TODO this is just for checking
		# threshold_lo = ...
		# threshold_hi = ...

		# A node can be found to be unsplittable if no split honors the constraints (e.g. min_samples_leaf)
		#  or, say, if every feature is constant (but frankly, that's quite unlikely...)
		unsplittable = true

		n_vars = n_variables(X)
		
		#####################
		## Find best split ##
		#####################
		# For each variable
		@inbounds for feature in 1:n_vars
			@logmsg DTDebug "Testing feature $(feature)/$(n_vars)..."
			# DEBUG: Uncomment this to stop at the first valid split encountered for any feature
			# if !unsplittable
			# 	break
			# end
			
			# Gather all values needed for the current feature
			# TODO note that Xf is almost not required animore...
			# @simd for i in 1:n_instances
			# 	# @logmsg DTDetail "Instance $(i)/$(n_instances)"
			# 	# TODO make this a view? featureview?
			# 	setfeature!(i, Xf, X.domain, indX[i + r_start], feature)
			# end

			## Test all conditions
			# For each relational operator
			for relation_id in relation_ids
				relation = relationSet[relation_id]
				@logmsg DTDebug "Testing relation $(relation) (id: $(relation_id))..." # "/$(length(relation_ids))"
				########################################################################
				########################################################################
				########################################################################
				# Find, for each instance, the highest value for any world
				#                       and the lowest value for any world
				# @info "Computing peaks..." # channel
				# opGeqMaxThresh_old     = fill(typemin(T), n_instances)
				# opLesMinThresh_old     = fill(typemax(T), n_instances)
				# for i in 1:n_instances
				# 	# if relation == ModalLogic.Topo_TPP println("relation ", relation, " ", relation_id) end
				# 	# if relation == ModalLogic.Topo_TPP println("instance ", i) end
				# 	# if relation == ModalLogic.Topo_TPP println("Sf[i] ", Sf[i]) end
				# 	channel = ModalLogic.getChannel(Xf but remember this is not computed anymore, i)
				# 	# if relation == ModalLogic.Topo_TPP println("channel ", channel) end
				# 	# @info " instance $i/$n_instances" # channel
				# 	# TODO this findmin/findmax can be made more efficient, and even more efficient for intervals.
				# 	for w in ModalLogic.enumAcc(Sf[i], relation, channel)
				# 		# if relation == ModalLogic.Topo_TPP println("world ", w) end
				#     TODO expand this code to multiple test_operators
				# 		(_wmin,_wmax) = ModalLogic.WExtrema(test_operators, w, channel)
				# 		# if relation == ModalLogic.Topo_TPP println("wmin, wmax ", _wmin, " ", _wmax) end
				# 		opGeqMaxThresh_old[i] = max(opGeqMaxThresh_old[i], _wmin)
				# 		opLesMinThresh_old[i] = min(opLesMinThresh_old[i], _wmax)
				# 	end
				# 	# if relation == ModalLogic.Topo_TPP println("opGeqMaxThresh_old ", opGeqMaxThresh_old[i]) end
				# 	# if relation == ModalLogic.Topo_TPP println("opLesMinThresh_old ", opLesMinThresh_old[i]) end
				# end

				########################################################################
				########################################################################
				########################################################################

				thresholds = Array{T,2}(undef, length(test_operators), n_instances)
				for (i_test_operator,test_operator) in enumerate(test_operators)
					@views cur_thr = thresholds[i_test_operator,:]
					fill!(cur_thr, ModalLogic.bottom(test_operator, T))
				end

				# TODO optimize this!!
				firstWorld = X.ontology.worldType(ModalLogic.firstWorld)
				for i in 1:n_instances
					# TODO slice Gammas in Gammasf?
					@logmsg DTDetail " Instance $(i)/$(n_instances)" indX[i + r_start]
					worlds = if (relation != ModalLogic.RelationAll)
							Sf[i]
						else
							[firstWorld]
						end
					for w in worlds
						# TODO maybe read the specific value of Gammas referred to the test_operator?
						cur_gammas = readGammas(Gammas, w, indX[i + r_start], relation_id, feature)
						@logmsg DTDetail " cur_gammas" w cur_gammas
						for (i_test_operator,test_operator) in enumerate(test_operators) # TODO use correct indexing for test_operators
							# if relation == ModalLogic.Topo_TPP println("world ", w) end
							# if relation == ModalLogic.Topo_TPP println("w_opGeqMaxThresh, w_opLesMinThresh ", w_opGeqMaxThresh, " ", w_opLesMinThresh) end
							# (w_opGeqMaxThresh,w_opLesMinThresh) = readGammas(Gammas, w, indX[i + r_start], relation_id, feature)
							# @logmsg DTDetail "w_opGeqMaxThresh,w_opLesMinThresh " w w_opGeqMaxThresh w_opLesMinThresh
							# opGeqMaxThresh[i] = max(opGeqMaxThresh[i], w_opGeqMaxThresh)
							# opLesMinThresh[i] = min(opLesMinThresh[i], w_opLesMinThresh)

							# opExtremeThreshArr,optimizer = ModalLogic.polarity(test_operator) ? (opGeqMaxThresh,max) : (opLesMinThresh,min)
							# opExtremeThreshArr[i] = optimizer(opExtremeThreshArr[i], cur_gammas[i_test_operator])

							thresholds[i_test_operator,i] = ModalLogic.opt(test_operator)(thresholds[i_test_operator,i], cur_gammas[i_test_operator])
						end
						# if relation == ModalLogic.Topo_TPP println("opGeqMaxThresh ", opGeqMaxThresh[i]) end
						# if relation == ModalLogic.Topo_TPP println("opLesMinThresh ", opLesMinThresh[i]) end
					end
				end

				# TODO sort this and optimize?
				# TODO no need to do union!! Just use opGeqMaxThresh for one and opLesMinThresh for the other...
				# Obtain the list of reasonable thresholds
				
				# thresholdDomain = setdiff(union(Set(opGeqMaxThresh),Set(opLesMinThresh)),Set([typemin(T), typemax(T)]))

				# @logmsg DTDebug "Thresholds computed: " thresholds
				# readline()


				# Look for the correct test operator
				for (i_test_operator,test_operator) in enumerate(test_operators)
					thresholdArr = thresholds[i_test_operator,:]
					thresholdDomain = setdiff(Set(thresholdArr),Set([typemin(T), typemax(T)]))
					# Look for thresholdArr 'a' for the propositions like "feature >= a"
					for threshold in thresholdDomain
						@logmsg DTDebug " Testing condition: $(ModalLogic.display_modal_test(relation, test_operator, feature, threshold))"
						# Re-initialize right class counts
						nr = zero(U)
						ncr[:] .= zero(U)
						unsatisfied = fill(1, n_instances)
						for i in 1:n_instances
							# @logmsg DTDetail " instance $i/$n_instances ExtremeThresh ($(opGeqMaxThresh[i])/$(opLesMinThresh[i]))"
							satisfied = ModalLogic.evaluateThreshCondition(test_operator, threshold, thresholdArr[i])
							
							if !satisfied
								@logmsg DTDetail "NO"
								nr += Wf[i]
								ncr[Yf[i]] += Wf[i]
							else
								unsatisfied[i] = 0
								@logmsg DTDetail "YES"
							end
						end

						# Calculate left class counts
						@simd for lab in 1:length(nc) # TODO something like @simd ncl .= nc - ncr instead
							ncl[lab] = nc[lab] - ncr[lab]
						end
						nl = nt - nr
						@logmsg DTDebug "  (n_left,n_right) = ($nl,$nr)"

						# Honor min_samples_leaf
						if nl >= min_samples_leaf && (n_instances - nl) >= min_samples_leaf
							unsplittable = false
							# TODO figure out exactly what this purity is?
							purity__nt = -(nl * loss_function(ncl, nl) +
								      	 nr * loss_function(ncr, nr))
							if purity__nt > best_purity__nt && !isapprox(purity__nt, best_purity__nt)
								best_purity__nt     = purity__nt
								best_relation       = relation
								best_feature        = feature
								best_test_operator  = test_operator
								best_threshold      = threshold
								best_nl             = nl # TODO for checking consistency purposes only
								best_unsatisfied    = unsatisfied # TODO for checking consistency purposes only
								# TODO: At the end, we should take the average between current and last.
								#  This requires thresholds to be sorted
								# threshold_lo, threshold_hi  = last_f, curr_f
								@logmsg DTDetail "  Found new optimum: " (best_purity__nt/nt) best_relation best_feature best_test_operator best_threshold
							end
						end
					end # for threshold
				end # for test_operator
			end # for relation
		end # for feature

		# @logmsg DTOverview "purity increase" best_purity__nt/nt loss_function(nc, nt) (best_purity__nt/nt + loss_function(nc, nt)) (best_purity__nt/nt - loss_function(nc, nt))
		# If the split is good, partition and split according to the optimum
		@inbounds if (unsplittable
			|| (best_purity__nt/nt + loss_function(nc, nt) <= min_purity_increase)
			)
			@logmsg DTDebug " Leaf" unsplittable min_purity_increase (best_purity__nt/nt) loss_function(nc, nt) ((best_purity__nt/nt) + loss_function(nc, nt))
			node.is_leaf = true
			return
		else
			best_purity = best_purity__nt/nt
			# try
			# 	node.threshold = (threshold_lo + threshold_hi) / 2.0
			# catch
			# 	node.threshold = threshold_hi
			# end

			# split the samples into two parts:
			# - ones that are > threshold
			# - ones that are <= threshold

			node.purity         = best_purity
			node.modality       = best_relation
			node.feature        = best_feature
			node.test_operator  = best_test_operator
			node.threshold      = best_threshold
			
			# Compute new world sets (= make a modal step)
			# TODO remove the use of Xf
			@simd for i in 1:n_instances
				setfeature!(i, Xf, X.domain, indX[i + r_start], best_feature)
			end

			# TODO instead of using memory, here, just use two opposite indices and perform substitutions. indj = n_instances
			unsatisfied_flags = fill(1, n_instances)
			for i in 1:n_instances
				channel = ModalLogic.getChannel(Xf, i)
				@logmsg DTDetail " Instance $(i)/$(n_instances)" channel Sf[i]
				(satisfied,S[indX[i + r_start]]) = ModalLogic.modalStep(Sf[i], best_relation, channel, best_test_operator, best_threshold)
				unsatisfied_flags[i] = !satisfied # I'm using unsatisfied because then sorting puts YES instances first but TODO use the inverse sorting and use satisfied flag instead
			end

			@logmsg DTOverview " Branch ($(sum(unsatisfied_flags))+$(n_instances-sum(unsatisfied_flags))=$(n_instances) samples) on condition: $(ModalLogic.display_modal_test(best_relation, best_test_operator, best_feature, best_threshold)), purity $(best_purity)"

			@logmsg DTDetail " unsatisfied_flags" unsatisfied_flags

			if best_unsatisfied != unsatisfied_flags || best_nl != n_instances-sum(unsatisfied_flags) || length(unique(unsatisfied_flags)) == 1
				errStr = "Something's wrong with the optimization steps.\n"
				errStr *= "Branch ($(sum(unsatisfied_flags))+$(n_instances-sum(unsatisfied_flags))=$(n_instances) samples) on condition: $(ModalLogic.display_modal_test(best_relation, best_test_operator, best_feature, best_threshold)), purity $(best_purity)"
				if length(unique(unsatisfied_flags)) == 1
					errStr *= "Uninformative split.\n$(unsatisfied_flags)\n"
				end
				if best_unsatisfied != unsatisfied_flags || best_nl != n_instances-sum(unsatisfied_flags)
					errStr *= "Different unsatisfied and best_unsatisfied:\ncomputed: $(best_unsatisfied)\n$(best_nl)\nactual: $(unsatisfied_flags)\n$(n_instances-sum(unsatisfied_flags))\n"
				end
				for i in 1:n_instances
					errStr *= "$(ModalLogic.getFeature(X.domain, indX[i + r_start], best_feature))\t$(Sf[i])\t$(!(unsatisfied_flags[i]==1))\t$(S[indX[i + r_start]])\n";
				end
				throw(Base.ErrorException(errStr))
			end

			@logmsg DTDetail "pre-partition" region indX[region] unsatisfied_flags
			node.split_at = util.partition!(indX, unsatisfied_flags, 0, region)
			@logmsg DTDetail "post-partition" indX[region] node.split_at

			# For debug:
			# indX = rand(1:10, 10)
			# unsatisfied_flags = rand([1,0], 10)
			# partition!(indX, unsatisfied_flags, 0, 1:10)
			
			# Sort [Xf, Yf, Wf, Sf and indX] by Xf
			# util.q_bi_sort!(unsatisfied_flags, indX, 1, n_instances, r_start)
			# node.split_at = searchsortedfirst(unsatisfied_flags, true)
		end
	end
	# Split node at a previously-set node.split_at value.
	# The children inherits some of the data
	@inline function fork!(node::NodeMeta{S}) where S
		ind = node.split_at
		region = node.region
		depth = node.depth+1
		mdepth = (node.modality == ModalLogic.RelationNone ? node.modal_depth : node.modal_depth+1)
		@logmsg DTDetail "fork!(...): " node ind region mdepth
		# no need to copy because we will copy at the end
		node.l = NodeMeta{S}(region[    1:ind], depth, mdepth)
		node.r = NodeMeta{S}(region[ind+1:end], depth, mdepth)
	end

	include("compute-thresholds.jl")

	function check_input(
			X                   :: OntologicalDataset{T, N},
			Y                   :: AbstractVector{Label},
			W                   :: AbstractVector{U},
			loss_function       :: Function,
			max_depth           :: Int,
			min_samples_leaf    :: Int,
			min_loss_at_leaf    :: AbstractFloat,
			min_purity_increase :: AbstractFloat) where {T, U, N}
			n_instances, n_vars = n_samples(X), n_variables(X)
		if length(Y) != n_instances
			throw("dimension mismatch between X and Y ($(size(X.domain)) vs $(size(Y))")
		elseif length(W) != n_instances
			throw("dimension mismatch between X and W ($(size(X.domain)) vs $(size(W))")
		elseif max_depth < -1
			throw("unexpected value for max_depth: $(max_depth) (expected:"
				* " max_depth >= 0, or max_depth = -1 for infinite depth)")
		# elseif n_vars < max_features
			# throw("number of features $(n_vars) is less than the number "
				# * "of max features $(max_features)")
		# elseif max_features < 0
			# throw("number of features $(max_features) must be >= zero ")
		elseif min_samples_leaf < 1
			throw("min_samples_leaf must be a positive integer "
				* "(given $(min_samples_leaf))")
		elseif loss_function in [util.gini, util.zero_one] && (min_loss_at_leaf > 1.0 || min_loss_at_leaf <= 0.0)
			throw("min_loss_at_leaf for loss $(loss_function) must be in (0,1]"
				* "(given $(min_loss_at_leaf))")
		elseif loss_function in [util.entropy]
			min_loss_at_leaf_thresh = 0.75 # min_purity_increase 0.01
			min_purity_increase_thresh = 0.5
			if (min_loss_at_leaf >= min_loss_at_leaf_thresh)
				println("Warning! It is advised to use min_loss_at_leaf<$(min_loss_at_leaf_thresh) with loss $(loss_function)"
					* "(given $(min_loss_at_leaf))")
			elseif (min_purity_increase >= min_purity_increase_thresh)
				println("Warning! It is advised to use min_loss_at_leaf<$(min_purity_increase_thresh) with loss $(loss_function)"
					* "(given $(min_purity_increase))")
			end
		end
		# TODO check that X doesn't have nans, typemin(T), typemax(T), missings, nothing etc. ...
	end

	function _fit(
			X                       :: OntologicalDataset{T, N},
			Y                       :: AbstractVector{Label},
			W                       :: AbstractVector{U},
			loss                    :: Function,
			n_classes               :: Int,
			max_depth               :: Int,
			min_samples_leaf        :: Int, # TODO generalize to min_samples_leaf_relative and min_weight_leaf
			min_purity_increase     :: AbstractFloat,
			min_loss_at_leaf        :: AbstractFloat,
			initCondition           :: DecisionTree._initCondition,
			useRelationAll          :: Bool,
			useRelationId           :: Bool,
			test_operators          :: AbstractVector{<:ModalLogic.TestOperator},
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG) where {T, U, N}

		# Dataset sizes
		n_instances = n_samples(X)

		# Note: in the propositional case, some pairs of operators (e.g. <= and >)
		#  are complementary, and thus it is redundant to check both at the same node.
		#  We avoid this by only keeping one of the two operators.
		# TODO optimize this: use opposite_test_operator() to check pairs.
		# TODO But first, check that TestOpGeq95 and TestOpLeq05 are actually complementary
		if prod(channel_size(X)) == 1
			if test_operators ⊆ ModalLogic.all_ordered_test_operators
				test_operators = [ModalLogic.TestOpGeq]
				# test_operators = filter(e->e ≠ ModalLogic.TestOpLeq,test_operators)
			end
		end

		# Array memory for class counts
		nc  = Vector{U}(undef, n_classes)
		ncl = Vector{U}(undef, n_classes)
		ncr = Vector{U}(undef, n_classes)

		# TODO We need to write on S, thus it cannot be a static array like X Y and W;
		# Should belong inside each meta-node and then be copied? That's a waste of space(for each instance),
		# We only need the worlds for the currentinstance set.
		# What if it's not fixed size? Maybe it should be like the subset of indX[region], so that indX[region.start] is parallel to node.S[1]
		# TODO make the initial entity and initial modality a training parameter?

		w0params =
			if initCondition == startWithRelationAll
				[ModalLogic.emptyWorld]
			elseif initCondition == startAtCenter
				[ModalLogic.centeredWorld, channel_size(X)...]
			elseif typeof(initCondition) <: DecisionTree._startAtWorld
				[initCondition.w]
		end
		S = WorldSet{X.ontology.worldType}[[X.ontology.worldType(w0params...)] for i in 1:n_instances]

		# Array memory for dataset
		Xf = Array{T, N+1}(undef, channel_size(X)..., n_instances)
		Yf = Vector{Label}(undef, n_instances)
		Wf = Vector{U}(undef, n_instances)
		Sf = Vector{WorldSet{X.ontology.worldType}}(undef, n_instances)

		# Binary relations (= unary modal operators)
		# Note: the equality operator is the first, and is the one representing
		#  the propositional case.
		# TODO check what happens if ModalLogic.RelationAll o ModalLogic.RelationId is in X.ontology.relationSet (I mean... optimize that case)
		relationSet = [ModalLogic.RelationId, ModalLogic.RelationAll, (X.ontology.relationSet)...]
		relationId_id = 1
		relationAll_id = 2
		relation_ids = map((x)->x+2, 1:length(X.ontology.relationSet))
		# TODO figure out if one should use this

		availableModalRelation_ids = if useRelationAll
			[relationAll_id, relation_ids...]
		else
			relation_ids
		end

		allAvailableRelation_ids = if useRelationId
			[relationId_id, availableModalRelation_ids...]
		else
			availableModalRelation_ids
		end
		
		# Fix test_operators order
		test_operators = unique(test_operators)
		ModalLogic.sort_test_operators!(test_operators)
		
		# if length(test_operators) > 2
		# 	println("test_operators")
		# 	println(test_operators)
		# 	readline()
		# end

		# TODO check that length(channel_size(X)) == complexity(worldType)


		# TODO use Ef = Dict(X.ontology.worldType,NTuple{NTO,T})
		# Fill with ModalLogic.enumAcc(Sf[i], ModalLogic.RelationAll, channel)... 
		# TODO Ef = Array{T,1+worldTypeSize(X.ontology.worldType)}(undef, )

		# Calculate Gammas
		# TODO expand for generic test operators
		# TODO test with array-only Gammas = Array{T, 4}(undef, 2, n_worlds(X.ontology.worldType, channel_size(X)), n_instances, n_variables(X))
		# TODO try something like Gammas = fill(No: Dict{X.ontology.worldType,NTuple{NTO,T}}(), n_instances, n_variables(X))
		
		# TODO improve code leveraging world/dimensional dataset structure

		# Gammas = Vector{Dict{ModalLogic.AbstractRelation,Vector{Dict{X.ontology.worldType,NTuple{NTO,T}}}}}(undef, n_variables(X))
		
		# TODO maybe use offset-arrays? https://docs.julialang.org/en/v1/devdocs/offset-arrays/
		Gammas = computeGammas(X,X.ontology.worldType,test_operators,relationSet,relationId_id,availableModalRelation_ids)
		# Gammas = @btime computeGammas($X,$X.ontology.worldType,$test_operators,$relationSet,$relationId_id,$availableModalRelation_ids)

		# Sample indices (array of indices that will be sorted and partitioned across the leaves)
		indX = collect(1:n_instances)
		# Create root node
		root = NodeMeta{T}(1:n_instances, 0, 0)
		# Stack of nodes to process
		stack = Tuple{NodeMeta{T},Bool}[(root,(initCondition == startWithRelationAll))]
		# The first iteration is treated sightly differently
		@inbounds while length(stack) > 0
			# Pop node and process it
			(node,onlyUseRelationAll) = pop!(stack)
			_split!(
				X, Y, W, S,
				loss, node,
				max_depth,
				min_samples_leaf,
				min_loss_at_leaf,
				min_purity_increase,
				test_operators,
				indX,
				nc, ncl, ncr, Xf, Yf, Wf, Sf, Gammas,
				rng,
				relationSet,
				(onlyUseRelationAll ? [relationAll_id] : allAvailableRelation_ids)
				)
			# After processing, if needed, perform the split and push the two children for a later processing step
			if !node.is_leaf
				fork!(node)
				# Note: the left (positive) child is not limited to RelationAll, whereas the right child is only if the current node is as well.
				push!(stack, (node.l, false))
				push!(stack, (node.r, onlyUseRelationAll))
			end
		end

		return (root, indX)
	end

	function fit(;
			# In the modal case, dataset instances are Kripke models.
			# In this implementation, we don't accept a generic Kripke model in the explicit form of
			#  a graph; instead, an instance is a dimensional domain (e.g. a matrix or a 3D matrix) onto which
			#  worlds and relations are determined by a given Ontology.
			X                       :: OntologicalDataset{T, N},
			Y                       :: AbstractVector{S},
			W                       :: Union{Nothing, AbstractVector{U}},
			loss = util.entropy     :: Function,
			max_depth               :: Int,
			min_samples_leaf        :: Int,
			min_purity_increase     :: AbstractFloat,
			min_loss_at_leaf        :: AbstractFloat, # TODO add this to scikit's interface.
			initCondition           :: DecisionTree._initCondition,
			useRelationAll          :: Bool,
			useRelationId           :: Bool,
			test_operators          :: AbstractVector{<:ModalLogic.TestOperator} = [ModalLogic.TestOpGeq, ModalLogic.TestOpLeq],
			rng = Random.GLOBAL_RNG :: Random.AbstractRNG) where {T, S, U, N}

		# Obtain the dataset's "outer size": number of samples and number of features
		n_instances = n_samples(X)

		# Translate labels to categorical form
		labels, Y_ = util.assign(Y)

		# Use unary weights if no weight is supplied
		if W == nothing
			# TODO optimize w in the case of all-ones: write a subtype of AbstractVector:
			#  AllOnesVector, so that getindex(W, i) = 1 and sum(W) = size(W).
			#  This allows the compiler to optimize constants at compile-time
			W = fill(1, n_instances)
		end

		# Check validity of the input
		check_input(
			X, Y, W,
			loss,
			max_depth,
			min_samples_leaf,
			min_loss_at_leaf,
			min_purity_increase,
			)

		# Call core learning function
		root, indX = _fit(
			X, Y_, W,
			loss,
			length(labels),
			max_depth,
			min_samples_leaf,
			min_purity_increase,
			min_loss_at_leaf,
			initCondition,
			useRelationAll,
			useRelationId,
			test_operators,
			rng)

		return Tree{T, S}(root, labels, indX, initCondition)
	end
end
