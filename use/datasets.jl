using MAT
import Base: getindex, values
import Random

include("local.jl")
include("wav2stft_time_series.jl")

# Generate a new rng from a random pick from a given one.
spawn_rng(rng) = Random.MersenneTwister(abs(rand(rng, Int)))

SplatEduardDataset(N) = begin

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

	insts,classes = readDataset(data_dir * "test-Eduard/Train-$N.txt", Val(N))

	n_samples = length(insts)
	n_vars = 2
	
	X_train = Array{Float64,3}(undef, N, n_samples, n_vars)

	for i in 1:length(insts)
		X_train[:, i, :] .= hcat(insts[i]...)
	end

	Y_train = copy(classes)
	
	insts,classes = readDataset(data_dir * "test-Eduard/Test-$N.txt", Val(N))

	n_samples = length(insts)
	n_vars = 2
	
	X_test = Array{Float64,3}(undef, N, n_samples, n_vars)

	for i in 1:length(insts)
		X_test[:, i, :] .= hcat(insts[i]...)
	end

	Y_test = copy(classes)

	(X_train,Y_train),(X_test,Y_test)
	# [val for i ∈ 1:3 for val ∈ ds.values[i][:,1]] <-- prendo tutti i valori del primo attributo
	# sort!(collect(Set([val for i ∈ 1:3 for val ∈ ds.values[i][:,1]]))) <-- ordino il dominio
	# size(ds.values[1])    <-- dimensione della prima serie
end


# function testSplatSiemensDataset_not_stratified()
# 	for seed ∈ [1,2,3,4,5,6,7,8,9,10]
# 		for nmeans ∈ [5,10,15]
# 			for h ∈ [1,2]
# 				for distance ∈ [-19,-20,-21,-22,-23,-24]
# 					SplatSiemensDataset_not_stratified(seed, nmeans, h, distance)
# 				end
# 			end
# 		end
# 	end
# end

SplatSiemensDataset(rseed::Int, nmeans::Int, hour::Int, distance::Int; subdir = "Siemens") = begin

	readDataset(filepath::String) = open(filepath, "r") do io
		insts = Array{Array{Float64}}[]
		# labels = Int64[]
		labels = String[]

		numattributes = 0

		flag = false

		lines = readlines(io)

		i = 1

		if flag == false
			while isempty(lines[i]) || split(lines[i])[1] != "@data"
				i = i + 1
				if !isempty(lines[i]) && split(lines[i])[1] == "@data"
					break
				end
			end
			flag = true
			i = i+1
		end

		for j = i:length(lines)
			# line = lines[j]
			# # println("line=" * line)
			# split_line = split(line, "\',\'")
			# split_line[1] = split_line[1][2:end]    # removing the first '
			#
			# # println("split_line[1]")
			# # println(split_line[1])
			# # println("split_line[2]")
			# # println(split_line[2])
			# # println("split_line[10]")
			# # println(split_line[10])
			# # series = split(split_line[1], "','")    # splitting the series based on \n
			# #class = split_line[11]                   # the class
			#
			# temp = split(split_line[10], "\',")
			# class = temp[2]
			# split_line[10] = temp[1]
			#
			# series = [split_line[i] for i ∈ 1:10]

			line = lines[j]
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
				# @show class
				push!(insts, numseries)
				# push!(labels, parse(Int, class))
				push!(labels, class)
			end

			if numattributes === 0
				numattributes = length(series)
			end
		end

		(insts,labels)
	end

	fileWithPath = data_dir * subdir * "/TRAIN_seed" * string(rseed) * "_nMeans" * string(nmeans) * "_hour" * string(hour) * "_distanceFromEvent" * string(distance) * ".arff"
	println(fileWithPath)
	insts,classes = readDataset(fileWithPath)

	n_vars = 10

	X_train = Array{Float64,3}(undef, nmeans*hour, length(insts), n_vars)

	for i in 1:length(insts)
		# println()
		# println("i=" * string(i))
		# println("size(X_train)=" * string(size(X_train)))
		# println("size(insts[i])=" * string(size(insts[i])))
		# println("size(hcat(insts[i]...))=" * string(size(hcat(insts[i]...))))
		# println("size(X_train[:, i, :])=" * string(size(X_train[:, i, :])))
		X_train[:, i, :] .= hcat(insts[i]...)
	end

	Y_train = classes

	fileWithPath = data_dir * subdir * "/TEST_seed" * string(rseed) * "_nMeans" * string(nmeans) * "_hour" * string(hour) * "_distanceFromEvent" * string(distance) * ".arff"
	insts,classes = readDataset(fileWithPath)

	X_test = Array{Float64,3}(undef, nmeans*hour, length(insts), n_vars)

	for i in 1:length(insts)
		X_test[:, i, :] .= hcat(insts[i]...)
	end

	Y_test = classes

	(X_train,Y_train),(X_test,Y_test)
end

SiemensDataset_not_stratified(nmeans::Int, hour::Int, distance::Int; subdir = "Siemens") = begin

	readDataset(filepath::String) = open(filepath, "r") do io
		insts = Array{Array{Float64}}[]
		labels = Int64[]

		numattributes = 0

		flag = false

		lines = readlines(io)

		i = 1

		if flag == false
			while isempty(lines[i]) || split(lines[i])[1] != "@data"
				i = i + 1
				if !isempty(lines[i]) && split(lines[i])[1] == "@data"
					break
				end
			end
			flag = true
			i = i+1
		end

		for j = i:length(lines)
			# line = lines[j]
			# # println("line=" * line)
			# split_line = split(line, "\',\'")
			# split_line[1] = split_line[1][2:end]    # removing the first '
			#
			# # println("split_line[1]")
			# # println(split_line[1])
			# # println("split_line[2]")
			# # println(split_line[2])
			# # println("split_line[10]")
			# # println(split_line[10])
			# # series = split(split_line[1], "','")    # splitting the series based on \n
			# #class = split_line[11]                   # the class
			#
			# temp = split(split_line[10], "\',")
			# class = temp[2]
			# split_line[10] = temp[1]
			#
			# series = [split_line[i] for i ∈ 1:10]

			line = lines[j]
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
				# @show class
				push!(insts, numseries)
				push!(labels, parse(Int, class))
			end

			if numattributes === 0
				numattributes = length(series)
			end
		end

		(insts,labels)
	end

	fileWithPath = data_dir * subdir * "/nMeans" * string(nmeans) * "_hour" * string(hour) * "_distanceFromEvent" * string(distance) * ".arff"
	println(fileWithPath)
	insts,classes = readDataset(fileWithPath)

	n_vars = 10

	X = Array{Float64,3}(undef, nmeans*hour, length(insts), n_vars)
	Y = Vector{String}(undef, length(insts))

	pos_idx = []
	neg_idx = []
	for (i,class) in enumerate(classes)
		if class == 0
			push!(neg_idx, i)
		elseif class == 1
			push!(pos_idx, i)
		else
			throw("Unknown class: $(class)")
		end
	end

	progressive_i = 1
	for i in [pos_idx..., neg_idx...]
		X[:, progressive_i, :] .= hcat(insts[i]...)
		Y[progressive_i] = (i == 0 ? "negative" : "positive")
		progressive_i+=1
	end

	((X,Y), length(pos_idx), length(neg_idx))
end

################################################################################
################################################################################
################################################################################
# task 1: YES/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY
#   ( 66 user (141 sample) / 220 users (298 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# task 2: YES_WITH_COUGH/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY
#   ( 23 user (54 sample) / 29 users (32 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH
# task 3: YES_WITH_COUGH/NO_CLEAN_HISTORY_AND_LOW_PROBABILITY_WITH_ASTHMA_AND_COUGH_REPORTED
#   ( 23 user (54 sample) / 18 users (20 samples) in total)
# - v1: USING COUGH
# - v2: USING BREATH

using JSON
KDDDataset_not_stratified((n_task,n_version),
		audio_kwargs; ma_size = 1,
		ma_step = 1,
		max_points = -1,
		use_full_mfcc = false,
		preprocess_wavs = [],
		# rng = Random.GLOBAL_RNG :: Random.AbstractRNG
	) = begin
	@assert n_task in [1,2,3] "KDDDataset: invalid n_task: {$n_task}"
	@assert n_version in [1,2] "KDDDataset: invalid n_version: {$n_version}"

	kdd_data_dir = data_dir * "KDD/"

	task_to_folders = [
		[
			["covidandroidnocough", "covidandroidwithcough", "covidwebnocough", "covidwebwithcough"],
			["healthyandroidnosymp", "healthywebnosymp"],
			["YES", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"]
		],
		[
			["covidandroidwithcough", "covidwebwithcough"],
			["healthyandroidwithcough", "healthywebwithcough"],
			["YES_WITH_COUGH", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY"]
		],
		[
			["covidandroidwithcough", "covidwebwithcough"],
			["asthmaandroidwithcough", "asthmawebwithcough"],
			["YES_WITH_COUGH", "NO_CLEAN_HISTORY_AND_LOW_PROBABILITY_WITH_ASTHMA_AND_COUGH_REPORTED"]
		],
	]

	subfolder,file_suffix,file_prefix = (n_version == 1 ? ("cough","cough","cough_") : ("breath","breathe","breaths_"))

	folders_Y, folders_N, class_labels = task_to_folders[n_task]

	files_map = JSON.parsefile(kdd_data_dir * "files.json")

	function readFiles(folders)
		# https://stackoverflow.com/questions/59562325/moving-average-in-julia
		moving_average(vs::AbstractArray{T,1},n,st=1) where {T} = [sum(@view vs[i:(i+n-1)])/n for i in 1:st:(length(vs)-(n-1))]
		moving_average(vs::AbstractArray{T,2},n,st=1) where {T} = mapslices((x)->(@views moving_average(x,n,st)), vs, dims=2)
		# (sum(w) for w in partition(1:9, 3, 2))
		# moving_average_np(vs,num_out_points,st) = moving_average(vs,length(vs)-num_out_points*st+1,st)
		# moving_average_np(vs,num_out_points,o) = (w = length(vs)-num_out_points*(1-o/w)+1; moving_average(vs,w,1-o/w))
		# moving_average_np(vs,t,o) = begin
		# 	N = length(vs);
		# 	s = floor(Int, (N+1)/(t+(1/(1-o))))
		# 	w = ceil(Int, s/(1-o))
		# 	# moving_average(vs,w,1-ceil(Int, o/w))
		# end
		n_samples = 0
		timeseries = []
		for folder in folders
			cur_folder_timeseries = Array{Array{Array{Float64, 2}, 1}, 1}(undef, length(files_map[folder]))

			# collect is necessary because the threads macro only supports arrays
			# https://stackoverflow.com/questions/57633477/multi-threading-julia-shows-error-with-enumerate-iterator
			Threads.@threads for (i_samples, samples) in collect(enumerate(files_map[folder]))
				samples = 
					if folder in ["asthmawebwithcough", "covidwebnocough", "covidwebwithcough", "healthywebnosymp", "healthywebwithcough"]
						map((subfoldname)->"$folder/$subfoldname/audio_file_$(file_suffix).wav", samples)
					else
						filter!((filename)->startswith(filename,file_prefix), samples)
						map((filename)->"$folder/$subfolder/$filename", samples)
					end

				cur_file_timeseries = Array{Array{Float64, 2}, 1}(undef, length(samples))
				valid_i_filenames = []
				for (i_filename, filename) in collect(enumerate(samples))
					filepath = kdd_data_dir * "$filename"
					ts = wav2stft_time_series(filepath, audio_kwargs; preprocess_sample = preprocess_wavs, use_full_mfcc = use_full_mfcc)
					# Ignore instances with NaN (careful! this may leave with just a few instances)
					# if any(isnan.(ts))
					# 	@warn "Instance with NaN values was ignored"
					# 	continue
					# end
					ts = moving_average(ts, ma_size, ma_step)
					# Drop first point
					ts = @views ts[:,2:end]
					# println(size(ts))
					if max_points != -1 && size(ts,2)>max_points
						ts = ts[:,1:max_points]
					end
					# println(size(ts))
					# readline()
					# println(size(wav2stft_time_series(filepath, audio_kwargs)))
					cur_file_timeseries[i_filename] = ts
					push!(valid_i_filenames, i_filename)
					n_samples += 1
				end
				cur_folder_timeseries[i_samples] = cur_file_timeseries[valid_i_filenames]
				# break
			end
			append!(timeseries, [j for i in cur_folder_timeseries for j in i])
		end
		# @assert n_samples == tot_n_samples "KDDDataset: unmatching tot_n_samples: {$n_samples} != {$tot_n_samples}"
		timeseries
	end

	pos = readFiles(folders_Y)
	neg = readFiles(folders_N)

	println("POS={$(length(pos))}, NEG={$(length(neg))}")
	#n_per_class = min(length(pos), length(neg))

	#pos = pos[Random.randperm(rng, length(pos))[1:n_per_class]]
	#neg = neg[Random.randperm(rng, length(neg))[1:n_per_class]]

	#println("Balanced -> {$n_per_class}+{$n_per_class}")

	# Stratify
	# timeseries = vec(hcat(pos,neg)')
	# Y = vec(hcat(ones(Int,length(pos)),zeros(Int,length(neg)))')

	# print(size(pos))
	# print(size(neg))
	timeseries = [[p' for p in pos]..., [n' for n in neg]...]
	# print(size(timeseries))
	# print(size(timeseries[1]))
	# Y = [ones(Int, length(pos))..., zeros(Int, length(neg))...]
	# Y = [zeros(Int, length(pos))..., ones(Int, length(neg))...]
	Y = [fill(class_labels[1], length(pos))..., fill(class_labels[2], length(neg))...]
	# print(size(Y))

	# println([size(ts, 1) for ts in timeseries])
	max_timepoints = maximum(size(ts, 1) for ts in timeseries)
	n_unique_freqs = unique(size(ts, 2) for ts in timeseries)
	@assert length(n_unique_freqs) == 1 "KDDDataset: length(n_unique_freqs) != 1: {$n_unique_freqs} != 1"
	n_unique_freqs = n_unique_freqs[1]
	X = zeros((max_timepoints, length(timeseries), n_unique_freqs))
	for (i,ts) in enumerate(timeseries)
		# println(size(ts))
		X[1:size(ts, 1),i,:] = ts
	end

	((X,Y), length(pos), length(neg))
end
################################################################################
################################################################################
################################################################################



simpleDataset(n_samp::Int, N::Int, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	X = Array{Int,3}(undef, N, n_samp, 1);
	Y = Vector{String}(undef, n_samp);
	for i in 1:n_samp
		instance = fill(2, N)
		y = rand(rng, 0:1)
		if y == 0
			instance[3] = 1
		else
			instance[3] = 2
		end
		X[:,i,1] .= instance
		Y[i] = string(y)
	end
	(X,Y)
end

simpleDataset2(n_samp::Int, N::Int, rng = Random.GLOBAL_RNG :: Random.AbstractRNG) = begin
	X = Array{Int,3}(undef, N, n_samp, 1);
	Y = Vector{String}(undef, n_samp);
	for i in 1:n_samp
		instance = fill(0, N)
		y = rand(rng, 0:1)
		if y == 0
			instance[3] += 1
		else
			instance[1] += 1
		end
		X[:,i,1] .= instance
		Y[i] = string(y)
	end
	(X,Y)
end

IndianPinesDataset() = begin
	X = matread(data_dir * "indian-pines/Indian_pines_corrected.mat")["indian_pines_corrected"]
	Y = matread(data_dir * "indian-pines/Indian_pines_gt.mat")["indian_pines_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SalinasDataset() = begin
	X = matread(data_dir * "salinas/Salinas_corrected.mat")["salinas_corrected"]
	Y = matread(data_dir * "salinas/Salinas_gt.mat")["salinas_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

SalinasADataset() = begin
	X = matread(data_dir * "salinas-A/SalinasA_corrected.mat")["salinasA_corrected"]
	Y = matread(data_dir * "salinas-A/SalinasA_gt.mat")["salinasA_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

PaviaCentreDataset() = begin
	X = matread(data_dir * "paviaC/Pavia.mat")["pavia"]
	Y = matread(data_dir * "paviaC/Pavia_gt.mat")["pavia_gt"]
	(X, Y) = map(((x)->round.(Int,x)), (X, Y))
end

PaviaDataset() = begin
	X = matread(data_dir * "paviaU/PaviaU.mat")["paviaU"]
	Y = matread(data_dir * "paviaU/PaviaU_gt.mat")["paviaU_gt"]
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
	class_labels = [class_labels_map[y] for y in existingLabels]
	inputs, [class_labels[y] for y in labels]
end
