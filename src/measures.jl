# CM[actual,predicted]
struct ConfusionMatrix
	classes::Vector
	matrix::Matrix{Int}
	# TODO only keep classes and matrix, the others can be inferred at a later time?
	overall_accuracy::Float64
	kappa::Float64
	mean_accuracy::Float64

	accuracies::Vector{Float64}
	F1s::Vector{Float64}
	sensitivities::Vector{Float64}
	specificities::Vector{Float64}
	PPVs::Vector{Float64}
	NPVs::Vector{Float64}
	ConfusionMatrix(classes::Vector, matrix::Matrix{Int}, overall_accuracy::Float64, kappa::Float64) = begin
		ConfusionMatrix(classes, matrix)
	end
	ConfusionMatrix(matrix::Matrix{Int}) = begin
		ConfusionMatrix(fill("", size(matrix, 1)), matrix)
	end
	ConfusionMatrix(classes::Vector, matrix::Matrix{Int}) = begin
		ALL = sum(matrix)
		T = LinearAlgebra.tr(matrix)
		F = ALL-T

		@assert size(matrix,1) == size(matrix,2) "Can't instantiate ConfusionMatrix with matrix of size ($(size(matrix))"

		n_classes = size(matrix,1)
		@assert length(classes) == n_classes "Can't instantiate ConfusionMatrix with mismatching n_classes ($(n_classes)) and classes $(classes)"
		overall_accuracy = T / ALL
		prob_chance = (sum(matrix,dims=1) * sum(matrix,dims=2))[1] / ALL^2
		kappa = (overall_accuracy - prob_chance) / (1.0 - prob_chance)

		TPs = Vector{Float64}(undef, n_classes)
		TNs = Vector{Float64}(undef, n_classes)
		FPs = Vector{Float64}(undef, n_classes)
		FNs = Vector{Float64}(undef, n_classes)

		for i in 1:n_classes
			class = i
			other_classes = [(1:i-1)..., (i+1:n_classes)...]
			TPs[i] = sum(matrix[class,class])
			TNs[i] = sum(matrix[other_classes,other_classes])
			FNs[i] = sum(matrix[class,other_classes])
			FPs[i] = sum(matrix[other_classes,class])
		end

		# https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
		accuracies = (TPs .+ TNs)./ALL
		mean_accuracy = Statistics.mean(accuracies)

		# https://en.wikipedia.org/wiki/Sensitivity_and_specificity
		F1s           = TPs./(TPs.+.5*(FPs.+FNs))
		# https://en.wikipedia.org/wiki/F-score
		sensitivities = TPs./(TPs.+FNs)
		specificities = TNs./(TNs.+FPs)
		PPVs          = TPs./(TPs.+FPs)
		NPVs          = TNs./(TNs.+FNs)

		new(classes, matrix, overall_accuracy, kappa, mean_accuracy, accuracies, F1s, sensitivities, specificities, PPVs, NPVs)
	end
end

class_counts(cm::ConfusionMatrix) = sum(cm.matrix,dims=2)
macro_F1(cm::ConfusionMatrix) = Statistics.mean(cm.F1s)
macro_weighted_F1(cm::ConfusionMatrix) = Statistics.sum(cm.F1s.*class_counts(cm))./sum(cm.matrix)
macro_sensitivity(cm::ConfusionMatrix) = Statistics.mean(cm.sensitivities)
macro_weighted_sensitivity(cm::ConfusionMatrix) = Statistics.sum(cm.sensitivities.*class_counts(cm))./sum(cm.matrix)
macro_specificity(cm::ConfusionMatrix) = Statistics.mean(cm.specificities)
macro_weighted_specificity(cm::ConfusionMatrix) = Statistics.sum(cm.specificities.*class_counts(cm))./sum(cm.matrix)
mean_PPV(cm::ConfusionMatrix) = Statistics.mean(cm.PPVs)
macro_weighted_PPV(cm::ConfusionMatrix) = Statistics.sum(cm.PPVs.*class_counts(cm))./sum(cm.matrix)
mean_NPV(cm::ConfusionMatrix) = Statistics.mean(cm.NPVs)
macro_weighted_NPV(cm::ConfusionMatrix) = Statistics.sum(cm.NPVs.*class_counts(cm))./sum(cm.matrix)

function show(io::IO, cm::ConfusionMatrix)
	print(io, "classes:  ")
	show(io, cm.classes)
	print(io, "\nmatrix:   ")
	display(cm.matrix)
	print(io, "\noverall_accuracy: ")
	show(io, cm.overall_accuracy)
	print(io, "\nkappa:    ")
	show(io, cm.kappa)
	print(io, "\nmean_accuracy:    ")
	show(io, cm.mean_accuracy)
	print(io, "\naccuracies:    ")
	show(io, cm.accuracies)
	print(io, "\nF1s:    ")
	show(io, cm.F1s)
	print(io, "\nsensitivities:    ")
	show(io, cm.sensitivities)
	print(io, "\nspecificities:    ")
	show(io, cm.specificities)
	print(io, "\nPPVs:    ")
	show(io, cm.PPVs)
	print(io, "\nNPVs:    ")
	show(io, cm.NPVs)
end

function _hist_add!(counts::Dict{T, Int}, labels::AbstractVector{T}, region::UnitRange{Int}) where T
	for i in region
		lbl = labels[i]
		counts[lbl] = get(counts, lbl, 0) + 1
	end
	return counts
end

_hist(labels::AbstractVector{T}, region::UnitRange{Int} = 1:lastindex(labels)) where T =
	_hist_add!(Dict{T,Int}(), labels, region)

function _weighted_error(actual::AbstractVector, predicted::AbstractVector, weights::AbstractVector{T}) where T <: Real
	mismatches = actual .!= predicted
	err = sum(weights[mismatches]) / sum(weights)
	return err
end

function majority_vote(labels::AbstractVector)
	if length(labels) == 0
		return nothing
	end
	counts = _hist(labels)
	top_vote = labels[1]
	top_count = -1
	for (k,v) in counts
		if v > top_count
			top_vote = k
			top_count = v
		end
	end
	return top_vote
end

function best_score(labels::AbstractVector{T}, weights::Union{Nothing,AbstractVector{N}}) where {T, N<:Real}
	if isnothing(weights)
		return majority_vote(labels)
	end

	@assert length(labels) === length(weights) "Each label must have a corresponding weight: labels length is $(length(labels)) and weights length is $(length(weights))."

	counts = Dict{T,AbstractFloat}()
	for i in 1:length(labels)
		l = labels[i]
		counts[l] = get(counts, l, 0) + weights[i]
	end

	top_vote = labels[1]
	top_score = -1

	for (k,v) in counts
		if v > top_score
			top_vote = k
			top_score = v
		end
	end

	return top_vote
end

### Classification ###

function confusion_matrix(actual::AbstractVector, predicted::AbstractVector)
	@assert length(actual) == length(predicted)
	N = length(actual)
	_actual = zeros(Int,N)
	_predicted = zeros(Int,N)
	classes = sort(unique([actual; predicted]))
	N = length(classes)
	for i in 1:N
		_actual[actual .== classes[i]] .= i
		_predicted[predicted .== classes[i]] .= i
	end
	# CM[actual,predicted]
	CM = zeros(Int,N,N)
	for i in zip(_actual, _predicted)
		CM[i[1],i[2]] += 1
	end
	return ConfusionMatrix(classes, CM)
end

function _nfoldCV(classifier::Symbol, labels::AbstractVector{T}, features::AbstractMatrix{S}, args...; verbose, rng) where {S, T}
	_rng = mk_rng(rng)::Random.AbstractRNG
	nfolds = args[1]
	if nfolds < 2
		throw("number of folds must be greater than 1")
	end
	if classifier == :tree
		pruning_purity      = args[2]
		max_depth           = args[3]
		min_samples_leaf    = args[4]
		min_samples_split   = args[5]
		min_purity_increase = args[6]
	elseif classifier == :forest
		n_subfeatures       = args[2]
		n_trees             = args[3]
		partial_sampling    = args[4]
		max_depth           = args[5]
		min_samples_leaf    = args[6]
		min_samples_split   = args[7]
		min_purity_increase = args[8]
	elseif classifier == :stumps
		n_iterations        = args[2]
	end
	N = length(labels)
	ntest = floor(Int, N / nfolds)
	inds = Random.randperm(_rng, N)
	accuracy = zeros(nfolds)
	for i in 1:nfolds
		test_inds = falses(N)
		test_inds[(i - 1) * ntest + 1 : i * ntest] .= true
		train_inds = (!).(test_inds)
		test_features = features[inds[test_inds],:]
		test_labels = labels[inds[test_inds]]
		train_features = features[inds[train_inds],:]
		train_labels = labels[inds[train_inds]]

		if classifier == :tree
			n_subfeatures = 0
			model = build_tree(train_labels, train_features,
				   n_subfeatures,
				   max_depth,
				   min_samples_leaf,
				   min_samples_split,
				   min_purity_increase;
				   rng = rng)
			if pruning_purity < 1.0
				model = prune_tree(model, pruning_purity)
			end
			predictions = apply_tree(model, test_features)
		elseif classifier == :forest
			model = build_forest(
						train_labels, train_features,
						n_subfeatures,
						n_trees,
						partial_sampling,
						max_depth,
						min_samples_leaf,
						min_samples_split,
						min_purity_increase;
						rng = rng)
			predictions = apply_forest(model, test_features)
		elseif classifier == :stumps
			model, coeffs = build_adaboost_stumps(
				train_labels, train_features, n_iterations)
			predictions = apply_adaboost_stumps(model, coeffs, test_features)
		end
		cm = confusion_matrix(test_labels, predictions)
		accuracy[i] = cm.accuracy
		if verbose
			println("\nFold ", i)
			println(cm)
		end
	end
	println("\nMean Accuracy: ", mean(accuracy))
	return accuracy
end

function nfoldCV_tree(
		labels              :: AbstractVector{T},
		features            :: AbstractMatrix{S},
		n_folds             :: Integer,
		pruning_purity      :: Float64 = 1.0,
		max_depth           :: Integer = -1,
		min_samples_leaf    :: Integer = 1,
		min_samples_split   :: Integer = 2,
		min_purity_increase :: Float64 = 0.0;
		verbose             :: Bool = true,
		rng                 = Random.GLOBAL_RNG) where {S, T}
	_nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
				min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
function nfoldCV_forest(
		labels              :: AbstractVector{T},
		features            :: AbstractMatrix{S},
		n_folds             :: Integer,
		n_subfeatures       :: Integer = -1,
		n_trees             :: Integer = 10,
		partial_sampling    :: Float64 = 0.7,
		max_depth           :: Integer = -1,
		min_samples_leaf    :: Integer = 1,
		min_samples_split   :: Integer = 2,
		min_purity_increase :: Float64 = 0.0;
		verbose             :: Bool = true,
		rng                 = Random.GLOBAL_RNG) where {S, T}
	_nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
				max_depth, min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
function nfoldCV_stumps(
		labels       ::AbstractVector{T},
		features     ::AbstractMatrix{S},
		n_folds      ::Integer,
		n_iterations ::Integer = 10;
		verbose             :: Bool = true,
		rng          = Random.GLOBAL_RNG) where {S, T}
	_nfoldCV(:stumps, labels, features, n_folds, n_iterations; verbose=verbose, rng=rng)
end

### Regression ###

function mean_squared_error(actual, predicted)
	@assert length(actual) == length(predicted)
	return mean((actual - predicted).^2)
end

function R2(actual, predicted)
	@assert length(actual) == length(predicted)
	ss_residual = sum((actual - predicted).^2)
	ss_total = sum((actual .- mean(actual)).^2)
	return 1.0 - ss_residual/ss_total
end

function _nfoldCV(regressor::Symbol, labels::AbstractVector{T}, features::AbstractMatrix, args...; verbose, rng) where T <: Float64
	_rng = mk_rng(rng)::Random.AbstractRNG
	nfolds = args[1]
	if nfolds < 2
		throw("number of folds must be greater than 1")
	end
	if regressor == :tree
		pruning_purity      = args[2]
		max_depth           = args[3]
		min_samples_leaf    = args[4]
		min_samples_split   = args[5]
		min_purity_increase = args[6]
	elseif regressor == :forest
		n_subfeatures       = args[2]
		n_trees             = args[3]
		partial_sampling    = args[4]
		max_depth           = args[5]
		min_samples_leaf    = args[6]
		min_samples_split   = args[7]
		min_purity_increase = args[8]
	end
	N = length(labels)
	ntest = floor(Int, N / nfolds)
	inds = Random.randperm(_rng, N)
	R2s = zeros(nfolds)
	for i in 1:nfolds
		test_inds = falses(N)
		test_inds[(i - 1) * ntest + 1 : i * ntest] .= true
		train_inds = (!).(test_inds)
		test_features = features[inds[test_inds],:]
		test_labels = labels[inds[test_inds]]
		train_features = features[inds[train_inds],:]
		train_labels = labels[inds[train_inds]]
		if regressor == :tree
			n_subfeatures = 0
			model = build_tree(train_labels, train_features,
				   n_subfeatures,
				   max_depth,
				   min_samples_leaf,
				   min_samples_split,
				   min_purity_increase;
				   rng = rng)
			if pruning_purity < 1.0
				model = prune_tree(model, pruning_purity)
			end
			predictions = apply_tree(model, test_features)
		elseif regressor == :forest
			model = build_forest(
						train_labels, train_features,
						n_subfeatures,
						n_trees,
						partial_sampling,
						max_depth,
						min_samples_leaf,
						min_samples_split,
						min_purity_increase;
						rng = rng)
			predictions = apply_forest(model, test_features)
		end
		err = mean_squared_error(test_labels, predictions)
		corr = cor(test_labels, predictions)
		r2 = R2(test_labels, predictions)
		R2s[i] = r2
		if verbose
			println("\nFold ", i)
			println("Mean Squared Error:     ", err)
			println("Correlation Coeff:      ", corr)
			println("Coeff of Determination: ", r2)
		end
	end
	println("\nMean Coeff of Determination: ", mean(R2s))
	return R2s
end

function nfoldCV_tree(
	labels              :: AbstractVector{T},
	features            :: AbstractMatrix{S},
	n_folds             :: Integer,
	pruning_purity      :: Float64 = 1.0,
	max_depth           :: Integer = -1,
	min_samples_leaf    :: Integer = 5,
	min_samples_split   :: Integer = 2,
	min_purity_increase :: Float64 = 0.0;
	verbose             :: Bool = true,
	rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
_nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
			min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
function nfoldCV_forest(
	labels              :: AbstractVector{T},
	features            :: AbstractMatrix{S},
	n_folds             :: Integer,
	n_subfeatures       :: Integer = -1,
	n_trees             :: Integer = 10,
	partial_sampling    :: Float64 = 0.7,
	max_depth           :: Integer = -1,
	min_samples_leaf    :: Integer = 5,
	min_samples_split   :: Integer = 2,
	min_purity_increase :: Float64 = 0.0;
	verbose             :: Bool = true,
	rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
_nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
			max_depth, min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
