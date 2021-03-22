using MAT
import Base: getindex, values
import Random

data_dir = "/home/gio/Desktop/SpatialDecisionTree/"
# data_dir = "/home/gpagliarini/ModalDecisionTrees/"

mapArrayToDataType(type::Type, array::AbstractArray) = begin
	minVal = minimum(array)
	maxVal = maximum(array)
	normalized_array = (array.-minVal)./(maxVal-minVal)
	typemin(type) .+ round.(type, (big(typemax(type))-big(typemin(type)))*normalized_array)
end

scaleDataset(dataset::Tuple, type::Type = UInt8) =
	(mapArrayToDataType(type, dataset[1]),dataset[2],dataset[3])

readDataset(filepath::String, ::Val{N}) where {N} = open(filepath, "r") do io
	insts = Array{Array{Float64}}[]
	labels = String[]

	numattributes = 0

	lines = readlines(io)
	for line ∈ lines
		split_line = split(line, "\',")
		split_line[1] = split_line[1][2:end]    # removing the first '
		series = split(split_line[1], "\\n")    # splitting the series based on \n
		class = split_line[2]                   # the class

		if length(series) ≥ 1
			numserie = [parse(Float64, strval) for strval ∈ split(series[1], ",")]
			numseries = Array{Float64}[]
			push!(numseries, numserie)

			for i ∈ 2:length(series)
				numserie = [parse(Float64, strval) for strval ∈ split(series[i], ",")]
				push!(numseries, numserie)
			end

			push!(insts, numseries)
			push!(labels, class)
		end

		if numattributes === 0
			numattributes = length(series)
		end
	end
	(insts,labels)
	# domains = Array{Float64}[]
	# for j ∈ 1:numattributes
	# 	# sort all values for each timeserie (i.e., row of size(insts)[1] rows) for the j-th attribute; that is, compute the sorted domain of each attribute
	# 	push!(domains, sort!(collect(Set([val for i ∈ 1:size(insts)[1] for val ∈ insts.values[i][j]]))))
	# end

	# classes = collect(Set(map((a,b)->(b), insts)))
end

EduardDataset(N) = begin
	insts,classes = readDataset("datasets/test-da-Eduard/Train-$N.txt", Val(N))

	n_samples = length(insts)
	n_vars = 2
	
	X_train = Array{Float64,3}(undef, N, n_samples, n_vars)

	for i in 1:length(insts)
		X_train[:, i, :] .= hcat(insts[i]...)
	end

	Y_train = map((x)-> parse(Int, x), classes)
	
	insts,classes = readDataset("datasets/test-da-Eduard/Test-$N.txt", Val(N))

	n_samples = length(insts)
	n_vars = 2
	
	X_test = Array{Float64,3}(undef, N, n_samples, n_vars)

	for i in 1:length(insts)
		X_test[:, i, :] .= hcat(insts[i]...)
	end

	Y_test = map((x)-> parse(Int, x), classes)

	(X_train,Y_train,X_test,Y_test)
	# [val for i ∈ 1:3 for val ∈ ds.values[i][:,1]] <-- prendo tutti i valori del primo attributo
	# sort!(collect(Set([val for i ∈ 1:3 for val ∈ ds.values[i][:,1]]))) <-- ordino il dominio
	# size(ds.values[1])    <-- dimensione della prima serie
end

simpleDataset(n_samp::Int, N::Int, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	X = Array{Int,3}(undef, N, n_samp, 1);
	Y = Array{Int,1}(undef, n_samp);
	for i in 1:n_samp
		instance = fill(2, N)
		y = rand(rng, 0:1)
		if y == 0
			instance[3] = 1
		else
			instance[3] = 2
		end
		X[:,i,1] .= instance
		Y[i] = y
	end
	(X,Y)
end

simpleDataset2(n_samp::Int, N::Int, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	X = Array{Int,3}(undef, N, n_samp, 1);
	Y = Array{Int,1}(undef, n_samp);
	for i in 1:n_samp
		instance = fill(0, N)
		y = rand(rng, 0:1)
		if y == 0
			instance[3] += 1
		else
			instance[1] += 1
		end
		X[:,i,1] .= instance
		Y[i] = y
	end
	(X,Y)
end

IndianPinesDataset() = begin
	X = matread(data_dir * "datasets/indian-pines/Indian_pines_corrected.mat")["indian_pines_corrected"]
	Y = matread(data_dir * "datasets/indian-pines/Indian_pines_gt.mat")["indian_pines_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SalinasDataset() = begin
	X = matread(data_dir * "datasets/salinas/Salinas_corrected.mat")["salinas_corrected"]
	Y = matread(data_dir * "datasets/salinas/Salinas_gt.mat")["salinas_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SalinasADataset() = begin
	X = matread(data_dir * "datasets/salinas-A/SalinasA_corrected.mat")["salinasA_corrected"]
	Y = matread(data_dir * "datasets/salinas-A/SalinasA_gt.mat")["salinasA_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

PaviaCentreDataset() = begin
	X = matread(data_dir * "datasets/paviaC/Pavia.mat")["pavia"]
	Y = matread(data_dir * "datasets/paviaC/Pavia_gt.mat")["pavia_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

PaviaDataset() = begin
	X = matread(data_dir * "datasets/paviaU/PaviaU.mat")["paviaU"]
	Y = matread(data_dir * "datasets/paviaU/PaviaU_gt.mat")["paviaU_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SampleLandCoverDataset(dataset::String, n_samples_per_label::Int, sample_size::Union{Int,NTuple{2,Int}}; n_variables::Int = -1, flattened::Bool = false, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	if typeof(sample_size) <: Int
		sample_size = (sample_size, sample_size)
	end
	@assert isodd(sample_size[1]) && isodd(sample_size[2])
	(Xmap, Ymap), class_labels_map = 	if dataset == "IndianPines"
									IndianPinesDataset(),["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
								elseif dataset == "Salinas"
									SalinasDataset(),["Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained", "Vinyard_vertical_trellis"]
								elseif dataset == "Salinas-A"
									SalinasADataset(),Dict(1 => "Brocoli_green_weeds_1", 10 => "Corn_senesced_green_weeds", 11 => "Lettuce_romaine_4wk", 12 => "Lettuce_romaine_5wk", 13 => "Lettuce_romaine_6wk", 14 => "Lettuce_romaine_7wk")
								elseif dataset == "PaviaCentre"
									PaviaCentreDataset(),["Water", "Trees", "Asphalt", "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows", "Meadows", "Bare Soil"]
								elseif dataset == "Pavia"
									PaviaDataset(),["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]
								else
									error("Unknown land cover dataset")
	end

	# print(size(Xmap))
	# readline()
	existingLabels = sort(filter!(l->l≠0, unique(Ymap)))
	# print(existingLabels)
	# readline()
	n_labels = length(existingLabels)

	if n_labels != length(class_labels_map)
		error("Unexpected number of labels in dataset: $(n_labels) != $(length(class_labels_map))")
	end

	class_counts = Dict(y=>0 for y in existingLabels)
	no_class_counts = 0
	for y in Ymap
		if y ≠ 0
			class_counts[y] += 1
		else
			no_class_counts += 1
		end
	end
	# println(zip(class_labels_map,class_counts) |> collect)
	# println(no_class_counts)
	# readline()

	class_is_to_ignore = Dict(y=>(class_counts[y] < n_samples_per_label) for y in existingLabels)

	if sum(values(class_is_to_ignore)) != 0
		# println("Warning! The following classes will be ignored in order to balance the dataset:")
		ignored_existingLabels = filter(y->(class_is_to_ignore[y]), existingLabels)
		non_ignored_existingLabels = map(y->!(class_is_to_ignore[y]), existingLabels)
		filter(y->(class_is_to_ignore[y]), existingLabels)
		# println([class_labels_map[y] for y in ignored_existingLabels])
		# println([class_counts[y] for y in ignored_existingLabels])
		n_labels = sum(non_ignored_existingLabels)
		# println(class_labels_map)
		# println(class_counts)
		# println(n_labels)
		# readline()
	end

	n_samples = n_samples_per_label*n_labels
	# print(n_samples)
	# cols = sort!([1:size(M,2);], by=i->(v1[i],v2[i]));

	X,Y = size(Xmap)[1], size(Xmap)[2]
	tot_variables = size(Xmap)[3]
	inputs = Array{eltype(Xmap),4}(undef, sample_size[1], sample_size[2], n_samples, tot_variables)
	labels = Vector{eltype(Ymap)}(undef, n_samples)
	sampled_class_counts = Dict(y=>0 for y in existingLabels)

	already_sampled = fill(false, X, Y)
	for i in 1:n_samples
		# print(i)
		while (x = rand(rng, 1:(X-sample_size[1])+1);
					y = rand(rng, 1:(Y-sample_size[2])+1);
					exLabel = Ymap[x+floor(Int,sample_size[1]/2),y+floor(Int,sample_size[2]/2)];
					exLabel == 0 || class_is_to_ignore[exLabel] || already_sampled[x,y] || sampled_class_counts[exLabel] == n_samples_per_label
					)
		end
		# print( Xmap[x:x+sample_size[1]-1,y:y+sample_size[2]-1,:] )
		# print( size(inputs[:,:,i,:]) )
		# readline()
		# print(label)
		# print(sampled_class_counts)
		# println(x,y)
		# println(x,x+sample_size[1]-1)
		# println(y,y+sample_size[2]-1)
		# println(already_sampled[x,y])
		# readline()

		inputs[:,:,i,:] .= Xmap[x:x+sample_size[1]-1,y:y+sample_size[2]-1,:]
		already_sampled[x,y] = true
		labels[i] = findfirst(x->x==exLabel, existingLabels)
		sampled_class_counts[exLabel] += 1
		# readline()
	end
	if n_variables != -1
		# new_inputs = Array{eltype(Xmap),4}(undef, sample_size[1], sample_size[2], n_samples, n_variables)
		n_variables
		inputs = inputs[:,:,:,1:floor(Int, tot_variables/n_variables):tot_variables]
		# new_inputs[:,:,:,:] = inputs[:,:,:,:]
		# inputs = new_inputs
	end

	if (sum(already_sampled) != n_samples)
		error("ERROR! Sampling failed! $(n_samples) $(sum(already_sampled))")
	end
	# println(labels)

	sp = sortperm(labels)
	labels = labels[sp]
	inputs = inputs[:,:,sp,:]
	
	labels = reshape(transpose(reshape(labels, (n_samples_per_label,n_labels))), n_samples)
	inputs = reshape(permutedims(reshape(inputs, (sample_size[1],sample_size[2],n_samples_per_label,n_labels,size(inputs, 4))), [1,2,4,3,5]), (sample_size[1],sample_size[2],n_samples,size(inputs, 4)))

	if flattened
		inputs = reshape(inputs, (1,1,n_samples,(sample_size[1]*sample_size[2]*size(inputs, 4))))
	end
	# println([class_labels_map[y] for y in existingLabels])
	# println(labels)
	inputs,labels,[class_labels_map[y] for y in existingLabels]
end

traintestsplit(data::Tuple{MatricialDataset{D,3},AbstractVector{T},AbstractVector{String}},threshold) where {D,T} = begin
	(X,Y,class_labels) = data
	spl = floor(Int, length(Y)*threshold)
	X_train = X[:,1:spl,:]
	Y_train = Y[1:spl]
	X_test  = X[:,spl+1:end,:]
	Y_test  = Y[spl+1:end]
	(X_train,Y_train),(X_test,Y_test),class_labels
end

traintestsplit(data::Tuple{MatricialDataset{D,4},AbstractVector{T},AbstractVector{String}},threshold) where {D,T} = begin
	(X,Y,class_labels) = data
	spl = floor(Int, length(Y)*threshold)
	X_train = X[:,:,1:spl,:]
	Y_train = Y[1:spl]
	X_test  = X[:,:,spl+1:end,:]
	Y_test  = Y[spl+1:end]
	(X_train,Y_train),(X_test,Y_test),class_labels
end

