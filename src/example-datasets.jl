using MAT
import Base: getindex, values
import Random

# data_dir = "/home/gio/Desktop/SpatialDecisionTree/"
data_dir = "/home/gpagliarini/ModalDecisionTrees/"

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

PaviaDataset() = begin
	paviaU = matread(data_dir * "datasets/paviaU/PaviaU.mat")["paviaU"]
	paviaU_gt = matread(data_dir * "datasets/paviaU/PaviaU_gt.mat")["paviaU_gt"]
	(paviaU, paviaU_gt) = map(((x)->round.(Int,x)), (paviaU, paviaU_gt))
end

IndianPinesCorrectedDataset() = begin
	Indian_pines_corrected = matread(data_dir * "datasets/indian-pines/Indian_pines_corrected.mat")["indian_pines_corrected"]
	Indian_pines_gt = matread(data_dir * "datasets/indian-pines/Indian_pines_gt.mat")["indian_pines_gt"]
	(Indian_pines_corrected, Indian_pines_gt) = map(((x)->round.(Int,x)), (Indian_pines_corrected, Indian_pines_gt))
end

SampleLandCoverDataset(n_samples::Int, sample_size::Union{Int,NTuple{2,Int}}, dataset::String; n_variables::Int = -1, flattened::Bool = false, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	if typeof(sample_size) <: Int
		sample_size = (sample_size, sample_size)
	end
	@assert isodd(sample_size[1]) && isodd(sample_size[2])
	(Xmap, Ymap), class_labels = 	if dataset == "Pavia"
									PaviaDataset(),["Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]
								elseif dataset == "IndianPinesCorrected"
									IndianPinesCorrectedDataset(),["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
								else
									error("Unknown land cover dataset")
	end
	# print(size(Xmap))
	# readline()
	existingLabels = filter!(l->l≠0, unique(Ymap))
	# print(existingLabels)
	# readline()
	n_labels = length(existingLabels)
	n_samples_per_label = round(Int, n_samples/n_labels)
	n_samples = n_samples_per_label*n_labels
	# print(n_samples)
	# cols = sort!([1:size(M,2);], by=i->(v1[i],v2[i]));

	X,Y = size(Xmap)[1], size(Xmap)[2]
	tot_variables = size(Xmap)[3]
	inputs = Array{eltype(Xmap),4}(undef, sample_size[1], sample_size[2], n_samples, tot_variables)
	labels = Vector{eltype(Ymap)}(undef, n_samples)
	classCounts = Dict(i => 0 for i in existingLabels)
	
	for i in 1:n_samples
		# print(i)
		while (x = rand(rng, 1:(X-sample_size[1])+1);
					y = rand(rng, 1:(Y-sample_size[2])+1);
					label = Ymap[x+floor(Int,sample_size[1]/2),y+floor(Int,sample_size[2]/2)];
					label == 0 || classCounts[label] == n_samples_per_label
					)
		end
		# print( Xmap[x:x+sample_size[1]-1,y:y+sample_size[2]-1,:] )
		# print( size(inputs[:,:,i,:]) )
		# readline()
		# print(label)
		# print(classCounts)
		inputs[:,:,i,:] .= Xmap[x:x+sample_size[1]-1,y:y+sample_size[2]-1,:]
		labels[i] = label
		classCounts[label] += 1
		# readline()
	end
	if n_variables != -1
		# new_inputs = Array{eltype(Xmap),4}(undef, sample_size[1], sample_size[2], n_samples, n_variables)
		n_variables
		inputs = inputs[:,:,:,1:floor(Int, tot_variables/n_variables):tot_variables]
		# new_inputs[:,:,:,:] = inputs[:,:,:,:]
		# inputs = new_inputs
	end

	sp = sortperm(labels)
	labels = labels[sp]
	inputs = inputs[:,:,sp,:]
	
	labels = reshape(transpose(reshape(labels, (n_samples_per_label,n_labels))), n_samples)
	inputs = reshape(permutedims(reshape(inputs, (sample_size[1],sample_size[2],n_samples_per_label,n_labels,size(inputs, 4))), [1,2,4,3,5]), (sample_size[1],sample_size[2],n_samples,size(inputs, 4)))

	if flattened
		inputs = reshape(inputs, (1,1,n_samples,(sample_size[1]*sample_size[2]*size(inputs, 4))))
	end
	inputs,labels,class_labels
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

