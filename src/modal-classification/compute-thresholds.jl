

	@inline function readSogliole(
		Sogliole            :: AbstractArray{<:AbstractDict{WorldType,Tuple{T,T}},3},
		w, i, relation_id, feature) where {T,WorldType<:AbstractWorld}
		return Sogliole[i, relation_id, feature][w]
	end

	@inline function readSogliole(
		Sogliole            :: AbstractArray{Tuple{T,T},N},
		w, i, relation_id, feature) where {N,T}
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
	# 	Sogliole = Array{Dict{X.ontology.worldType,Tuple{T,T}}, 3}(undef, n_instances, length(relationSet), n_vars)
	# 	@inbounds for feature in 1:n_vars
	# 		println("feature $(feature)/$(n_vars)")
			
	# 		# Find the highest/lowest thresholds

	# 		for i in 1:n_instances
	# 			# println("instance $(i)/$(n_samples)")

	# 			# Propositional, local
	# 			relation_id = relationId_id
	# 			Sogliole[i,relation_id,feature] = Dict{X.ontology.worldType,Tuple{T,T}}()
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
	# 				Sogliole[i,relation_id,feature] = Dict{X.ontology.worldType,Tuple{T,T}}()
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

		Sogliole = Array{Tuple{T,T}, 7}(undef, x, x+1, y, y+1, n_instances, length(relationSet), n_vars)
		@inbounds for feature in 1:n_vars
			println("feature $(feature)/$(n_vars)")
			
			# get_thresholds = (w, channel)->ModalLogic.WExtrema(test_operators, w, channel)
			# get_thresholds_repr = (w, relation, channel)->ModalLogic.WExtremaRepr(test_operators, w, channel)
			get_thresholds = (w, channel)->ModalLogic.WExtrema(w, channel)
			get_thresholds_repr = (w, channel)->ModalLogic.WExtremaRepr(w, channel)
			
			# Find the highest/lowest thresholds

			for i in 1:n_instances
				# println("instance $(i)/$(n_samples)")

				# Propositional, local
				@views channel = ModalLogic.getFeature(X.domain, i, feature) # TODO check @views
				for w in ModalLogic.enumAcc(X.ontology.worldType[], ModalLogic.RelationAll, channel)
					# opGeqMaxThresh, opLesMinThresh = ModalLogic.WMin(w, channel), ModalLogic.WMax(w, channel)
					thresholds = get_thresholds(w, channel)
					Sogliole[w.x.x, w.x.y, w.y.x, w.y.y, i,relationId_id,feature] = thresholds
				end

				@views SoglId = Sogliole[:,:,:,:, i,relationId_id,feature]
				# Modal
				for relation_id in relation_ids
					relation = relationSet[relation_id]
					@views Sogl = Sogliole[:,:,:,:, i,relation_id,feature]
					# println("relation $(relation)")
					# @info "Relation " relation_id relation
					# For each world w and each relation R, compute the peaks of v worlds, with w<R>v
					for w in ModalLogic.enumAcc(X.ontology.worldType[], ModalLogic.RelationAll, channel)
						# @info "World " w 
						thresholds = get_thresholds_repr(ModalLogic.enumAccRepr(w, relation, channel), channel)
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

						@info "World " relation_id relation w thresholds
						Sogl[w.x.x, w.x.y, w.y.x, w.y.y] = thresholds
					end
				end # relation

			end # instances
		end # feature
		@info "Done." Sogliole[:,[1,relation_ids...],:]
		Sogliole
	end
