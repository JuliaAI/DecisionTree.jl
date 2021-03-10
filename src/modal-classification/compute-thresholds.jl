	
	# readSogliole: read a specific value of the Sogliole array

	@inline function readSogliole(
		Sogliole            :: AbstractArray{<:AbstractDict{WorldType,NTuple{NTO,T}},3},
		w, i, relation_id, feature) where {NTO,T,WorldType<:AbstractWorld}
		return Sogliole[i, relation_id, feature][w]
	end

	@inline function readSogliole(
		Sogliole            :: AbstractArray{NTuple{NTO,T},N},
		w, i, relation_id, feature) where {N,NTO,T}
		return Sogliole[w.x.x, w.x.y, w.y.x, w.y.y, i, relation_id, feature] # TODO fix and generalize this
	end

	# function computeSogliole(
	# 		X                  :: OntologicalDataset{T, N},
	# 		worldType          :: Type{WorldType},
	# 		# test_operators     :: AbstractVector{<:ModalLogic.TestOperator},
	# 		relationSet        :: Vector{<:ModalLogic.AbstractRelation},
	# 		relationId_id      :: Int,
	# 		relation_ids       :: AbstractVector{Int},
	# 	) where {T, N, WorldType<:AbstractWorld}
	# 	n_instances = n_samples(X)
	# 	n_vars = n_variables(X)
	# 	Sogliole = Array{Dict{X.ontology.worldType,NTuple{NTO,T}}, 3}(undef, n_instances, length(relationSet), n_vars)
	# 	@inbounds for feature in 1:n_vars
	# 		println("feature $(feature)/$(n_vars)")
			
	# 		# Find the highest/lowest thresholds

	# 		for i in 1:n_instances
	# 			# println("instance $(i)/$(n_samples)")

	# 			# Propositional, local
	# 			relation_id = relationId_id
	# 			Sogliole[i,relation_id,feature] = Dict{X.ontology.worldType,NTuple{NTO,T}}()
	# 			@views channel = ModalLogic.getFeature(X.domain, i, feature) # TODO check @views
	# 			for w in ModalLogic.enumAcc(X.ontology.worldType[], ModalLogic.RelationAll, channel)
	# 				# opGeqMaxThresh, opLesMinThresh = ModalLogic.WMin(w, channel), ModalLogic.WMax(w, channel)
	# 				opGeqMaxThresh, opLesMinThresh = ModalLogic.WExtrema(w, channel)
	# 				Sogliole[i,relation_id,feature][w] = (opGeqMaxThresh, opLesMinThresh)
	# 			end

	# 			# Modal
	# 			for relation_id in relation_ids
	# 				relation = relationSet[relation_id]
	# 				# println("relation $(relation)")
	# 				# @info "Relation " relation_id relation
	# 				Sogliole[i,relation_id,feature] = Dict{X.ontology.worldType,NTuple{NTO,T}}()
	# 				# For each world w and each relation R, compute the peaks of v worlds, with w<R>v
	# 				for w in ModalLogic.enumAcc(X.ontology.worldType[], ModalLogic.RelationAll, channel)
	# 					# @info "World " w 
	# 					opGeqMaxThresh, opLesMinThresh = typemin(T), typemax(T)
	# 					for v in ModalLogic.enumAccRepr(w, relation, channel)
	# 					# for v in ModalLogic.enumAcc([w], relation, channel)
	# 						(v_opGeqMaxThresh, v_opLesMinThresh) = Sogliole[i,relationId_id,feature][v]
	# 						# @info "  ->World " v v_opGeqMaxThresh v_opLesMinThresh
	# 						opGeqMaxThresh = max(opGeqMaxThresh, v_opGeqMaxThresh)
	# 						opLesMinThresh = min(opLesMinThresh, v_opLesMinThresh)
	# 					end # worlds
	# 					@info "World " relation_id relation w opGeqMaxThresh opLesMinThresh
	# 					Sogliole[i,relation_id,feature][w] = (opGeqMaxThresh, opLesMinThresh)
	# 				end
	# 			end # relation

	# 		end # instances
	# 	end # feature
	# 	@info "Done." Sogliole[:,[1,relation_ids...],:]
	# 	Sogliole
	# end

	function computeSogliole(
			X                  :: OntologicalDataset{T, N},
			worldType          :: Type{ModalLogic.Interval2D},
			test_operators     :: AbstractVector{<:ModalLogic.TestOperator},
			relationSet        :: Vector{<:ModalLogic.AbstractRelation},
			relationId_id      :: Int,
			relation_ids       :: AbstractVector{Int},
		) where {T, N}
		n_instances = n_samples(X)
		n_vars = n_variables(X)
		x,y = channel_size(X)

		# Prepare Sogliole array
		Sogliole = Array{NTuple{length(test_operators),T}, 7}(undef, x, x+1, y, y+1, n_instances, length(relationSet), n_vars)
		@logmsg DTDebug "Computing Sogliole for Interval2D datasets..." size(X) n_instances n_vars x y test_operators relationSet relationId_id relation_ids size(Sogliole)

		firstWorld = X.ontology.worldType(ModalLogic.firstWorld)

		# @logmsg DTDebug "" test_operators
		# readline()

		# With sorted test_operators
		actual_test_operators = Tuple{Bool,ModalLogic.TestOperator}[]
		nonprimary_test_operators = ModalLogic.TestOperator[]
		for i_test_operator in 1:length(test_operators)
			test_operator = test_operators[i_test_operator]
			# @logmsg DTDebug "" test_operator
			# readline()
			if test_operator in nonprimary_test_operators
				# Skip test_operator
			elseif length(test_operators) >= i_test_operator+1 && ModalLogic.dual_test_operator(test_operator) == test_operators[i_test_operator+1]
				push!(actual_test_operators, (true,ModalLogic.primary_test_operator(test_operator)))
				push!(nonprimary_test_operators,test_operators[i_test_operator+1])
			else
				push!(actual_test_operators, (false,test_operator))
			end
		end

		# @logmsg DTDebug "..." test_operators actual_test_operators
		# readline()
		
		# get_thresholds = (w, channel)->ModalLogic.WExtrema(w, channel)
		# get_thresholds_repr = (w, channel)->ModalLogic.WExtremaRepr(w, channel)

		get_thresholds(w::AbstractWorld, channel::ModalLogic.MatricialChannel{T,N}) = begin
			thresholds = T[]
			# thresholds = similar(test_operators, T)
			for (both,test_operator) in actual_test_operators
				thresholds = if both
						[thresholds..., ModalLogic.WExtrema(test_operator, w, channel)...]
					else
						[thresholds..., ModalLogic.WExtreme(test_operator, w, channel)]
					end
			end
			Tuple(thresholds)
		end

		@inline WExtremaModal(test_operator::ModalLogic.TestOperator, SoglId, w::AbstractWorld, relation::AbstractRelation, channel::ModalLogic.MatricialChannel{T,N}) where {T,N} = begin
			ModalLogic.WExtremaModal(test_operator, w, relation, channel)
			
			# TODO use SoglId...?
			# TODO fix this
			# accrepr = ModalLogic.enumAccRepr(test_operator, w, relation, channel)

			# # TODO use SoglId[w.x.x, w.x.y, w.y.x, w.y.y]
			# # accrepr::Tuple{Bool,AbstractWorldSet{<:AbstractWorld}}
			# inverted, representatives = accrepr
			# opGeqMaxThresh, opLesMinThresh = typemin(T), typemax(T)
			# for w in representatives
			# 	(_wmin, _wmax) = ModalLogic.WExtrema(test_operator, w, channel)
			# 	if inverted
			# 		(_wmax, _wmin) = (_wmin, _wmax)
			# 	end
			# 	opGeqMaxThresh = max(opGeqMaxThresh, _wmin)
			# 	opLesMinThresh = min(opLesMinThresh, _wmax)
			# end
			# return (opGeqMaxThresh, opLesMinThresh)
		end

		@inline WExtremeModal(test_operator::ModalLogic.TestOperator, SoglId, w::AbstractWorld, relation::AbstractRelation, channel::ModalLogic.MatricialChannel{T,N}) where {T,N} = begin
			ModalLogic.WExtremeModal(test_operator, w, relation, channel)
		# 	# TODO fix this
		# 	accrepr = ModalLogic.enumAccRepr(test_operator, w, relation, channel)
			
		# 	# TODO use SoglId[w.x.x, w.x.y, w.y.x, w.y.y]
		# 	# accrepr::Tuple{Bool,AbstractWorldSet{<:AbstractWorld}}
		# 	inverted, representatives = accrepr
		# 	TODO inverted...
		# 	(opExtremeThresh, optimizer) = if ModalLogic.polarity(test_operator)
		# 			typemin(T), max
		# 		else
		# 			typemax(T), min
		# 		end
		# 	for w in representatives
		# 		_wextreme = ModalLogic.WExtreme(test_operator, w, channel)
		# 		opExtremeThresh = optimizer(opExtremeThresh, _wextreme)
		# 	end
		# 	return opExtremeThresh
		end

		get_thresholds_repr(SoglId, w, relation, channel) = begin
			thresholds = T[]
			# thresholds = similar(test_operators, T)
			for (both,test_operator) in actual_test_operators
				thresholds = if both
						[thresholds..., WExtremaModal(test_operator, SoglId, w, relation, channel)...]
					else
						[thresholds..., WExtremeModal(test_operator, SoglId, w, relation, channel)]
					end
			end
			Tuple(thresholds)
		end

		@inbounds for feature in 1:n_vars
			@logmsg DTDebug "Feature $(feature)/$(n_vars)"
			if ((feature-1) % floor(Int, ((n_vars-1)/5))) == 0
				@logmsg DTOverview "Feature $(feature)/$(n_vars)"
			end
			
			# Find the highest/lowest thresholds

			for i in 1:n_instances
				@logmsg DTDebug "Instance $(i)/$(n_instances)"

				# Propositional, local
				@views channel = ModalLogic.getFeature(X.domain, i, feature) # TODO check @views
				# println(channel)
				for w in ModalLogic.enumAcc(X.ontology.worldType[], ModalLogic.RelationAll, channel)
					@logmsg DTDetail "World" w
					# opGeqMaxThresh, opLesMinThresh = ModalLogic.WMin(w, channel), ModalLogic.WMax(w, channel)
					thresholds = get_thresholds(w, channel)
					Sogliole[w.x.x, w.x.y, w.y.x, w.y.y, i,relationId_id,feature] = thresholds
				end # world

				@views SoglId = Sogliole[:,:,:,:, i,relationId_id,feature]
				# Modal
				for relation_id in relation_ids
					relation = relationSet[relation_id]
					@logmsg DTDebug "Relation $(relation) (id: $(relation_id))" # "/$(length(relation_ids))"
					@views Sogl = Sogliole[:,:,:,:, i,relation_id,feature]
					# For each world w and each relation, compute the thresholds of all v worlds, with w<R>v
					worlds = if relation != ModalLogic.RelationAll
							ModalLogic.enumAcc(X.ontology.worldType[], ModalLogic.RelationAll, channel)
						else
							[firstWorld]
						end
					for w in worlds
						thresholds = get_thresholds_repr(SoglId, w, relation, channel)

						# opGeqMaxThresh, opLesMinThresh = typemin(T), typemax(T)
						# inverted,representatives = ModalLogic.enumAccRepr(w, relation, channel)
						# for v in representatives
						# # for v in ModalLogic.enumAcc([w], relation, channel)
						# 	# (v_opGeqMaxThresh, v_opLesMinThresh) = get_thresholds(v, channel)
						# 	(v_opGeqMaxThresh, v_opLesMinThresh) = SoglId[v.x.x, v.x.y, v.y.x, v.y.y]
						# 	if inverted
						# 		(v_opGeqMaxThresh, v_opLesMinThresh) = (v_opLesMinThresh, v_opGeqMaxThresh)
						# 	end
						# 	# @info "  ->World " v v_opGeqMaxThresh v_opLesMinThresh
						# 	opGeqMaxThresh = max(opGeqMaxThresh, v_opGeqMaxThresh)
						# 	opLesMinThresh = min(opLesMinThresh, v_opLesMinThresh)
						# end # worlds
						# thresholds = opGeqMaxThresh, opLesMinThresh
						# Quale e' piu' veloce? TODO use SoglId in Wextrema?
						# @assert (opGeqMaxThresh, opLesMinThresh) == ModalLogic.WExtremaRepr(ModalLogic.enumAccRepr(w, relation, channel), channel) "Wextrema different $((opGeqMaxThresh, opLesMinThresh)) $(get_thresholds(w, channel))"

						@logmsg DTDetail "World" w relation thresholds
						Sogl[w.x.x, w.x.y, w.y.x, w.y.y] = thresholds
					end # world
				end # relation

				# w = firstWorld
				# println(Sogliole[w.x.x, w.x.y, w.y.x, w.y.y, i,2,feature])
				# readline()

			end # instances
		end # feature
		@logmsg DTDebug "Done computing Sogliole" # Sogliole[:,[1,relation_ids...],:]
		Sogliole
	end
