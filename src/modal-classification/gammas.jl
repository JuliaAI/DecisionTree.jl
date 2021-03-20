
# gammas is a structure holding threshold values that are on the verge of truth of propositional split formulas.

# For generic worldTypes, gammas is an n-dim array of dictionaries indicized on the world itself.
#  Instead, when the structure of a world is known, its attributes are unrolled as
#  array dimensions; gammas then becomes an (n+k)-dim array,
#  where k is the complexity of the worldType.

# TODO test with array-only gammas = Array{T, 4}(undef, 2, n_worlds(X.ontology.worldType, channel_size(X)), n_instances, n_variables(X))
# TODO try something like gammas = fill(No: Dict{X.ontology.worldType,NTuple{NTO,T}}(), n_instances, n_variables(X))
# gammas = Vector{Dict{ModalLogic.AbstractRelation,Vector{Dict{X.ontology.worldType,NTuple{NTO,T}}}}}(undef, n_variables(X))		
# TODO maybe use offset-arrays? https://docs.julialang.org/en/v1/devdocs/offset-arrays/

# TODO make the test_operators tuple part of the array. ?

@inline initGammas(worldType::Type{AbstractWorld}, T::Type, channel_size::Tuple, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_vars::Integer) =
	Array{Dict{worldType,NTuple{n_test_operators,T}}, 3}(undef, n_instances, n_relations, n_vars)
@inline initGammas(worldType::Type{ModalLogic.Interval2D}, T::Type, (X,Y)::NTuple{2,Integer}, n_test_operators::Integer, n_instances::Integer, n_relations::Integer, n_vars::Integer) =
	Array{NTuple{n_test_operators,T}, 7}(undef, X, X+1, Y, Y+1, n_instances, n_relations, n_vars)

@inline setGamma(gammas::Array{Dict{WorldType,NTuple{NTO,T}}, 3}, w::WorldType, i_instances::Integer, i_relations::Integer, i_vars::Integer, thresholds::NTuple{NTO,T}) where {WorldType<:AbstractWorld,NTO,T} =
	gammas[i_instances, i_relations, i_vars][w] = thresholds
@inline setGamma(gammas::Array{NTuple{NTO,T}, 7}, w::ModalLogic.Interval2D, i_instances::Integer, i_relations::Integer, i_vars::Integer, thresholds::NTuple{NTO,T}) where {NTO,T} =
	gammas[w.x.x, w.x.y, w.y.x, w.y.y, i_instances, i_relations, i_vars] = thresholds

@inline initGammaSlice(worldType::Type{AbstractWorld}, gammas::Array{Dict{WorldType,NTuple{NTO,T}}, 3}, n_instances::Integer, n_relations::Integer, n_vars::Integer) where {WorldType<:AbstractWorld,NTO,T} =
	gammas[i_instances, i_relations, i_vars] = Dict{WorldType,NTuple{NTO,T}}()
@inline initGammaSlice(worldType::Type{ModalLogic.Interval2D}, gammas::Array{NTuple{NTO,T}, 7}, n_instances::Integer, n_relations::Integer, n_vars::Integer) where {NTO,T} =
	nothing

@inline sliceGamma(worldType::Type{AbstractWorld}, gammas::Array{Dict{WorldType,NTuple{NTO,T}}, 3}, i_instances::Integer, i_relations::Integer, i_vars::Integer) where {WorldType<:AbstractWorld,NTO,T} =
	gammas[i_instances, i_relations, i_vars]
@inline sliceGamma(worldType::Type{ModalLogic.Interval2D}, gammas::Array{NTuple{NTO,T}, 7}, i_instances::Integer, i_relations::Integer, i_vars::Integer) where {NTO,T} =
	@view gammas[:,:,:,:, i_instances, i_relations, i_vars]

@inline setGammaSlice(gammasSlice::Dict{WorldType,NTuple{NTO,T}}, w::WorldType, thresholds::NTuple{NTO,T}) where {WorldType<:AbstractWorld,NTO,T} =
	gammasSlice[w] = thresholds
@inline setGammaSlice(gammasSlice::AbstractArray{NTuple{NTO,T}, 4}, w::ModalLogic.Interval2D, thresholds::NTuple{NTO,T}) where {NTO,T} =
	gammasSlice[w.x.x, w.x.y, w.y.x, w.y.y] = thresholds

# readGammas: read a specific value of the gammas array

@inline function readGammas(
	gammas     :: AbstractArray{<:AbstractDict{WorldType,NTuple{NTO,T}},3},
	w          :: WorldType,
	i, relation_id, feature) where {NTO,T,WorldType<:AbstractWorld}
	return gammas[i, relation_id, feature][w]
end

@inline function readGammas(
	gammas     :: AbstractArray{NTuple{NTO,T},N},
	w          :: ModalLogic.Interval,
	i, relation_id, feature) where {N,NTO,T}
	return gammas[w.x, w.y, i, relation_id, feature]
end

@inline function readGammas(
	gammas     :: AbstractArray{NTuple{NTO,T},N},
	w          :: ModalLogic.Interval2D,
	i, relation_id, feature) where {N,NTO,T}
	return gammas[w.x.x, w.x.y, w.y.x, w.y.y, i, relation_id, feature]
end



function computeGammas(
		X                  :: OntologicalDataset{T, N},
		worldType          :: Type{WorldType},
		test_operators     :: AbstractVector{<:ModalLogic.TestOperator},
		relationSet        :: Vector{<:ModalLogic.AbstractRelation},
		relationId_id      :: Int,
		relation_ids       :: AbstractVector{Int},
	) where {T, N, WorldType<:AbstractWorld}
	
	n_instances = n_samples(X)
	n_vars = n_variables(X)
	n_relations = length(relationSet)
	n_test_operators = length(test_operators)

	# Prepare gammas array
	gammas = initGammas(worldType, T, channel_size(X), n_test_operators, n_instances, n_relations, n_vars)

	@logmsg DTDebug "Computing gammas..." size(X) worldType channel_size(X) test_operators n_instances n_relations n_vars relationSet relationId_id relation_ids size(gammas)

	firstWorld = worldType(ModalLogic.firstWorld)

	# With sorted test_operators
	# TODO fix
	actual_test_operators = Tuple{Integer,Union{<:ModalLogic.TestOperator,Vector{<:ModalLogic.TestOperator}}}[]
	already_inserted_test_operators = ModalLogic.TestOperator[]
	i_test_operator = 1
	while i_test_operator <= length(test_operators)
		test_operator = test_operators[i_test_operator]
		# println(i_test_operator, test_operators[i_test_operator])
		# @logmsg DTDebug "" test_operator
		# readline()
		if test_operator in already_inserted_test_operators
			# Skip test_operator
		elseif length(test_operators) >= i_test_operator+1 && ModalLogic.dual_test_operator(test_operator) != ModalLogic.TestOpNone && ModalLogic.dual_test_operator(test_operator) == test_operators[i_test_operator+1]
			push!(actual_test_operators, (1,ModalLogic.primary_test_operator(test_operator))) # "prim/dual"
			push!(already_inserted_test_operators,test_operators[i_test_operator+1])
		else
			siblings_present = intersect(test_operators,ModalLogic.siblings(test_operator))
			if length(siblings_present) > 1
				# TODO test if this is actually better
				push!(actual_test_operators, (2,siblings_present)) # "batch"
				for sibling in siblings_present
					push!(already_inserted_test_operators,sibling)
				end
			else
				push!(actual_test_operators, (0,test_operator)) # "single"
			end
		end
		i_test_operator+=1
	end
	
	# print(actual_test_operators)
	# readline()

	@inline WExtremaModal(test_operator::ModalLogic.TestOperator, gammasId, w::AbstractWorld, relation::AbstractRelation, channel::ModalLogic.MatricialChannel{T,N}) where {T,N} = begin
		# TODO use gammasId[w.x.x, w.x.y, w.y.x, w.y.y]...?
		ModalLogic.WExtremaModal(test_operator, w, relation, channel)

		# TODO fix this
		# accrepr = ModalLogic.enumAccRepr(test_operator, w, relation, channel)

		# # TODO use 
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

	@inline WExtremeModal(test_operator::ModalLogic.TestOperator, gammasId, w::AbstractWorld, relation::AbstractRelation, channel::ModalLogic.MatricialChannel{T,N}) where {T,N} = begin
		ModalLogic.WExtremeModal(test_operator, w, relation, channel)
	# 	# TODO fix this
	# 	accrepr = ModalLogic.enumAccRepr(test_operator, w, relation, channel)
		
	# 	# TODO use gammasId[w.x.x, w.x.y, w.y.x, w.y.y]
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

	@inline WExtremeModalMany(test_operators::Vector{<:ModalLogic.TestOperator}, gammasId, w::AbstractWorld, relation::AbstractRelation, channel::ModalLogic.MatricialChannel{T,N}) where {T,N} = begin
		# TODO use gammasId[w.x.x, w.x.y, w.y.x, w.y.y]...?
		ModalLogic.WExtremeModalMany(test_operators, w, relation, channel)
	end

	@inbounds Threads.@threads for feature in 1:n_vars
		@logmsg DTDebug "Feature $(feature)/$(n_vars)"
		if feature == 1 || ((feature+1) % (floor(Int, ((n_vars)/5))+1)) == 0
			@logmsg DTOverview "Feature $(feature)/$(n_vars)"
		end
		
		# Find the highest/lowest thresholds

		for i in 1:n_instances
			@logmsg DTDebug "Instance $(i)/$(n_instances)"

			# Propositional, local
			channel = ModalLogic.getFeature(X.domain, i, feature) # TODO check that @views actually avoids copying
			initGammaSlice(worldType, gammas, i, relationId_id, feature)
			# println(channel)
			for w in ModalLogic.enumAcc(worldType[], ModalLogic.RelationAll, channel)
				@logmsg DTDetail "World" w
				thresholds = T[]
				# thresholds = similar(test_operators, T)
				for (mode,test_operator) in actual_test_operators
					thresholds = if mode == 0
							[thresholds..., ModalLogic.WExtreme(test_operator, w, channel)]
						elseif mode == 1
							[thresholds..., ModalLogic.WExtrema(test_operator, w, channel)...]
						elseif mode == 2
							[thresholds..., ModalLogic.WExtremeMany(test_operator, w, channel)...]
						else
							error("Unexpected mode flag for test_operator $(test_operator): $(mode)\n$(test_operators)")
						end
				end

				setGamma(gammas, w, i, relationId_id, feature, Tuple(thresholds))
			end # world

			@views gammasId = sliceGamma(worldType, gammas, i, relationId_id, feature)
			# Modal
			for relation_id in relation_ids
				relation = relationSet[relation_id]
				initGammaSlice(worldType, gammas, i, relation_id, feature)
				@logmsg DTDebug "Relation $(relation) (id: $(relation_id))" # "/$(length(relation_ids))"
				# TOD Check if cur_gammas improves performances
				@views cur_gammas = sliceGamma(worldType, gammas, i, relation_id, feature)
				# For each world w and each relation, compute the thresholds of all v worlds, with w<R>v
				worlds = if relation != ModalLogic.RelationAll
						ModalLogic.enumAcc(worldType[], ModalLogic.RelationAll, channel)
					else
						[firstWorld]
					end
				for w in worlds
					thresholds  = Vector{T}(undef, length(test_operators))

					# TODO use gammasId, TODO gammasId[v]
					t=1
					for (mode,test_operator) in actual_test_operators
						if mode == 0
							thresholds[t] = WExtremeModal(test_operator, gammasId, w, relation, channel)
							t+=1
						elseif mode == 1
							thresholds[t:t+1] .= WExtremaModal(test_operator, gammasId, w, relation, channel)
							t+=2
						elseif mode == 2
							thresholds[t:t+length(test_operator)-1] .= WExtremeModalMany(test_operator, gammasId, w, relation, channel)
							t+=length(test_operator)
						else
							error("Unexpected mode flag for test_operator $(test_operator): $(mode)\n$(test_operators)")
						end
					end
					# Quale e' piu' veloce? TODO use gammasId in Wextrema?
					# @assert (opGeqMaxThresh, opLesMinThresh) == ModalLogic.WExtremaRepr(ModalLogic.enumAccRepr(w, relation, channel), channel) "Wextrema different $((opGeqMaxThresh, opLesMinThresh)) $(get_thresholds(w, channel))"

					@logmsg DTDetail "World" w relation Tuple(thresholds)

					# setGamma(gammas, w, i, relation_id, feature, Tuple(thresholds))
					setGammaSlice(cur_gammas, w, Tuple(thresholds))
				end # world
			end # relation

			# w = firstWorld
			# println(gammas[w.x.x, w.x.y, w.y.x, w.y.y, i,2,feature])
			# readline()

		end # instances
	end # feature
	@logmsg DTDebug "Done computing gammas" # gammas[:,[1,relation_ids...],:]
	gammas
end
