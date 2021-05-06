# TODO note that these splitting functions simply cut the dataset in two,
#  and they don't necessarily produce balanced cuts. To produce balanced cuts,
#  one must manually stratify the dataset beforehand
traintestsplit((X,Y)::Tuple{MatricialDataset{D,N},AbstractVector{String}}, split_threshold::AbstractFloat; gammas = nothing, worldType = nothing, is_balanced = true) where {D,N} = begin
	num_instances = length(Y)
	spl = ceil(Int, num_instances*split_threshold)
	# In the binary case, make it even
	if length(unique(Y)) == 2 && is_balanced
		spl = isodd(spl) ? (spl-1) : spl
	end
	X_train = ModalLogic.sliceDomainByInstances(X, 1:spl)
	Y_train = Y[1:spl]
	gammas_train = 
		if isnothing(gammas) # || isnothing(worldType)
			gammas
		else
			DecisionTree.sliceGammasByInstances(worldType, gammas, 1:spl; return_view = true)
		end
	X_test  = ModalLogic.sliceDomainByInstances(X, spl+1:num_instances)
	Y_test  = Y[spl+1:end]
	(X_train,Y_train), (X_test,Y_test), gammas_train
	# end
end

# Scale and round dataset to fit into a certain datatype's range:
# For integers: find minimum and maximum (ignoring Infs), and rescale the dataset
# For floating-point, numbers, round
# TODO roundDataset 
roundDataset((X,Y)::Tuple, type::Type = UInt8) = (mapArrayToDataType(type, X),Y)
roundDataset(((X_train,Y_train),(X_test,Y_test))::Tuple, type::Type = UInt8) = begin
	X_train, X_test = mapArrayToDataType(type, (X_train, X_test))
	(X_train,Y_train),(X_test,Y_test)
end

mapArrayToDataType(type::Type{<:Integer}, array::AbstractArray; minVal = minimum(array), maxVal = maximum(array)) = begin
	normalized_array = (array.-minVal)./(maxVal-minVal)
	typemin(type) .+ round.(type, (big(typemax(type))-big(typemin(type)))*normalized_array)
end

mapArrayToDataType(type::Type{<:Integer}, arrays::Tuple) = begin
	minVal, maxVal = minimum(minimum.(array)), maximum(maximum.(array))
	map((array)->mapArrayToDataType(type,array), arrays; minVal = minVal, maxVal = maxVal)
end

mapArrayToDataType(type::Type{<:AbstractFloat}, array::AbstractArray) = begin
	# TODO worry about eps of the target type and the magnitude of values in array
	#  (and eventually scale up or down the array). Also change mapArrayToDataType(type, Xs::Tuple) then
	type.(array)
end

mapArrayToDataType(type::Type{<:AbstractFloat}, arrays::Tuple) = begin
	map((array)->mapArrayToDataType(type,array), arrays)
end

