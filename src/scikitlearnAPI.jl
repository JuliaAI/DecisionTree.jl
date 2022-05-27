import ScikitLearnBase: BaseClassifier, BaseRegressor, predict, predict_proba,
                        fit!, get_classes, @declare_hyperparameters

################################################################################
# Classifier

"""
    DecisionTreeClassifier(; pruning_purity_threshold=0.0,
                           max_depth::Int=-1,
                           min_samples_leaf::Int=1,
                           min_samples_split::Int=2,
                           min_purity_increase::Float=0.0,
                           n_subfeatures::Int=0,
                           rng=Random.GLOBAL_RNG,
                           calc_fi::Bool=true)

Decision tree classifier. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `pruning_purity_threshold`: (post-pruning) merge leaves having `>=thresh` combined purity (default: no pruning)
- `max_depth`: maximum depth of the decision tree (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 1)
- `min_samples_split`: the minimum number of samples in needed for a split (default: 2)
- `min_purity_increase`: minimum purity needed for a split (default: 0.0)
- `n_subfeatures`: number of features to select at random (default: keep all)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.
- `calc_fi`: whether to calculate feature importances using `Mean Decrease in Impurity (MDI)`

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct DecisionTreeClassifier <: BaseClassifier
    pruning_purity_threshold::Float64 # no pruning if 1.0
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Int
    rng::Random.Random.AbstractRNG
    calc_fi::Bool
    root::Union{LeafOrNode, Nothing}
    classes::Union{Vector, Nothing}
    DecisionTreeClassifier(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=1, min_samples_split=2,
                           min_purity_increase=0.0, n_subfeatures=0, rng=Random.GLOBAL_RNG, calc_fi=true, root=nothing, classes=nothing) =
        new(pruning_purity_threshold, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, n_subfeatures, mk_rng(rng), calc_fi, root, classes)
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
@declare_hyperparameters(DecisionTreeClassifier,
                         [:pruning_purity_threshold, :max_depth, :min_samples_leaf,
                          :min_samples_split, :min_purity_increase, :rng, :calc_fi])

function fit!(dt::DecisionTreeClassifier, X, y)
    n_samples, n_features = size(X)
    dt.root = build_tree(
        y, X,
        dt.n_subfeatures,
        dt.max_depth,
        dt.min_samples_leaf,
        dt.min_samples_split,
        dt.min_purity_increase;
        rng = dt.rng,
        calc_fi = dt.calc_fi)

    dt.root = prune_tree(dt.root, dt.pruning_purity_threshold)
    dt.classes = sort(unique(y))
    dt
end

predict(dt::DecisionTreeClassifier, X) = apply_tree(dt.root, X)

predict_proba(dt::DecisionTreeClassifier, X) =
    apply_tree_proba(dt.root, X, dt.classes)

predict_log_proba(dt::DecisionTreeClassifier, X) =
    log(predict_proba(dt, X)) # this will yield -Inf when p=0. Hmmm...

function show(io::IO, dt::DecisionTreeClassifier)
    println(io, "DecisionTreeClassifier")
    println(io, "max_depth:                $(dt.max_depth)")
    println(io, "min_samples_leaf:         $(dt.min_samples_leaf)")
    println(io, "min_samples_split:        $(dt.min_samples_split)")
    println(io, "min_purity_increase:      $(dt.min_purity_increase)")
    println(io, "pruning_purity_threshold: $(dt.pruning_purity_threshold)")
    println(io, "n_subfeatures:            $(dt.n_subfeatures)")
    print(io,   "classes:                  ") ; show(io, dt.classes) ; println(io, "")
    print(io,   "root:                     ") ; show(io, dt.root)
end

################################################################################
# Regression

"""
    DecisionTreeRegressor(; pruning_purity_threshold=0.0,
                          max_depth::Int-1,
                          min_samples_leaf::Int=5,
                          min_samples_split::Int=2,
                          min_purity_increase::Float=0.0,
                          n_subfeatures::Int=0,
                          rng=Random.GLOBAL_RNG,
                          calc_fi::Bool=true)
Decision tree regression. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `pruning_purity_threshold`: (post-pruning) merge leaves having `>=thresh` combined purity (default: no pruning)
- `max_depth`: maximum depth of the decision tree (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 5)
- `min_samples_split`: the minimum number of samples in needed for a split (default: 2)
- `min_purity_increase`: minimum purity needed for a split (default: 0.0)
- `n_subfeatures`: number of features to select at random (default: keep all)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.
- `calc_fi`: whether to calculate feature importances using `Mean Decrease in Impurity (MDI)`

Implements `fit!`, `predict`, `get_classes`
"""
mutable struct DecisionTreeRegressor <: BaseRegressor
    pruning_purity_threshold::Float64
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    n_subfeatures::Int
    rng::Random.AbstractRNG
    calc_fi::Bool
    root::Union{LeafOrNode, Nothing}
    DecisionTreeRegressor(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=5,
                          min_samples_split=2, min_purity_increase=0.0, n_subfeatures=0, rng=Random.GLOBAL_RNG, calc_fi=true, root=nothing) =
        new(pruning_purity_threshold,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            n_subfeatures,
            mk_rng(rng),
            calc_fi,
            root)
end

@declare_hyperparameters(DecisionTreeRegressor,
                         [:pruning_purity_threshold, :min_samples_leaf, :n_subfeatures,
                          :max_depth, :min_samples_split, :min_purity_increase, :rng, :calc_fi])

function fit!(dt::DecisionTreeRegressor, X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    dt.root = build_tree(
        float.(y), X,
        dt.n_subfeatures,
        dt.max_depth,
        dt.min_samples_leaf,
        dt.min_samples_split,
        dt.min_purity_increase;
        rng = dt.rng,
        calc_fi = dt.calc_fi)
    
    dt.root = prune_tree(dt.root, dt.pruning_purity_threshold)
    dt
end

predict(dt::DecisionTreeRegressor, X) = apply_tree(dt.root, X)

function show(io::IO, dt::DecisionTreeRegressor)
    println(io, "DecisionTreeRegressor")
    println(io, "max_depth:                $(dt.max_depth)")
    println(io, "min_samples_leaf:         $(dt.min_samples_leaf)")
    println(io, "min_samples_split:        $(dt.min_samples_split)")
    println(io, "min_purity_increase:      $(dt.min_purity_increase)")
    println(io, "pruning_purity_threshold: $(dt.pruning_purity_threshold)")
    println(io, "n_subfeatures:            $(dt.n_subfeatures)")
    print(io,   "root:                     ") ; show(io, dt.root)
end

################################################################################
# Random Forest Classification

"""
    RandomForestClassifier(; n_subfeatures::Int=-1,
                           n_trees::Int=10,
                           partial_sampling::Float=0.7,
                           max_depth::Int=-1,
                           rng=Random.GLOBAL_RNG,
                           calc_fi::Bool=true)
Random forest classification. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `n_subfeatures`: number of features to consider at random per split (default: -1, sqrt(# features))
- `n_trees`: number of trees to train (default: 10)
- `partial_sampling`: fraction of samples to train each tree on (default: 0.7)
- `max_depth`: maximum depth of the decision trees (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have
- `min_samples_split`: the minimum number of samples in needed for a split
- `min_purity_increase`: minimum purity needed for a split
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator. Multi-threaded forests must be seeded with an `Int`
- `calc_fi`: whether to calculate feature importances using `Mean Decrease in Impurity (MDI)`

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct RandomForestClassifier <: BaseClassifier
    n_subfeatures::Int
    n_trees::Int
    partial_sampling::Float64
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    rng::Union{Random.AbstractRNG, Int}
    calc_fi:: Bool
    ensemble::Union{Ensemble, Nothing}
    classes::Union{Vector, Nothing}
    RandomForestClassifier(; n_subfeatures=-1, n_trees=10, partial_sampling=0.7,
                           max_depth=-1, min_samples_leaf=1, min_samples_split=2, min_purity_increase=0.0,
                           rng=Random.GLOBAL_RNG, calc_fi=true,ensemble=nothing, classes=nothing) =
        new(n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, rng, calc_fi, ensemble, classes)
end

get_classes(rf::RandomForestClassifier) = rf.classes
@declare_hyperparameters(RandomForestClassifier,
                         [:n_subfeatures, :n_trees, :partial_sampling, :max_depth,
                          :min_samples_leaf, :min_samples_split, :min_purity_increase,
                          :rng, :calc_fi])

function fit!(rf::RandomForestClassifier, X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    rf.ensemble = build_forest(
        y, X,
        rf.n_subfeatures,
        rf.n_trees,
        rf.partial_sampling,
        rf.max_depth,
        rf.min_samples_leaf,
        rf.min_samples_split,
        rf.min_purity_increase;
        rng = rf.rng,
        calc_fi = rf.calc_fi)
    rf.classes = sort(unique(y))
    rf
end

predict_proba(rf::RandomForestClassifier, X) =
    apply_forest_proba(rf.ensemble, X, rf.classes)

predict(rf::RandomForestClassifier, X) = apply_forest(rf.ensemble, X)

function show(io::IO, rf::RandomForestClassifier)
    println(io, "RandomForestClassifier")
    println(io, "n_trees:             $(rf.n_trees)")
    println(io, "n_subfeatures:       $(rf.n_subfeatures)")
    println(io, "partial_sampling:    $(rf.partial_sampling)")
    println(io, "max_depth:           $(rf.max_depth)")
    println(io, "min_samples_leaf:    $(rf.min_samples_leaf)")
    println(io, "min_samples_split:   $(rf.min_samples_split)")
    println(io, "min_purity_increase: $(rf.min_purity_increase)")
    print(io,   "classes:             ") ; show(io, rf.classes)  ; println(io, "")
    print(io,   "ensemble:            ") ; show(io, rf.ensemble)
end

################################################################################
# Random Forest Regression

"""
    RandomForestRegressor(; n_subfeatures::Int=-1,
                          n_trees::Int=10,
                          partial_sampling::Float=0.7,
                          max_depth::Int=-1,
                          min_samples_leaf::Int=5,
                          rng=Random.GLOBAL_RNG,
                          calc_fi::Bool=true)
Random forest regression. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `n_subfeatures`: number of features to consider at random per split (default: -1, sqrt(# features))
- `n_trees`: number of trees to train (default: 10)
- `partial_sampling`: fraction of samples to train each tree on (default: 0.7)
- `max_depth`: maximum depth of the decision trees (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 5)
- `min_samples_split`: the minimum number of samples in needed for a split
- `min_purity_increase`: minimum purity needed for a split
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator. Multi-threaded forests must be seeded with an `Int`
- `calc_fi`: whether to calculate feature importances using `Mean Decrease in Impurity (MDI)`

Implements `fit!`, `predict`, `get_classes`
"""
mutable struct RandomForestRegressor <: BaseRegressor
    n_subfeatures::Int
    n_trees::Int
    partial_sampling::Float64
    max_depth::Int
    min_samples_leaf::Int
    min_samples_split::Int
    min_purity_increase::Float64
    rng::Union{Random.AbstractRNG, Int}
    calc_fi::Bool
    ensemble::Union{Ensemble, Nothing}
    RandomForestRegressor(; n_subfeatures=-1, n_trees=10, partial_sampling=0.7,
                            max_depth=-1, min_samples_leaf=5, min_samples_split=2, min_purity_increase=0.0,
                            rng=Random.GLOBAL_RNG, calc_fi=true, ensemble=nothing) =
        new(n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, rng, calc_fi, ensemble)
end

@declare_hyperparameters(RandomForestRegressor,
                         [:n_subfeatures, :n_trees, :partial_sampling,
                          :min_samples_leaf, :min_samples_split, :min_purity_increase,
                          # I'm not crazy about :rng being a hyperparameter,
                          # since it'll change throughout fitting, but it works
                          :max_depth, :rng, :calc_fi])

function fit!(rf::RandomForestRegressor, X::AbstractMatrix, y::AbstractVector)
    n_samples, n_features = size(X)
    rf.ensemble = build_forest(
        float.(y), X,
        rf.n_subfeatures,
        rf.n_trees,
        rf.partial_sampling,
        rf.max_depth,
        rf.min_samples_leaf,
        rf.min_samples_split,
        rf.min_purity_increase;
        rng = rf.rng,
        calc_fi = rf.calc_fi)
    rf
end

predict(rf::RandomForestRegressor, X) = apply_forest(rf.ensemble, X)

function show(io::IO, rf::RandomForestRegressor)
    println(io, "RandomForestRegressor")
    println(io, "n_trees:             $(rf.n_trees)")
    println(io, "n_subfeatures:       $(rf.n_subfeatures)")
    println(io, "partial_sampling:    $(rf.partial_sampling)")
    println(io, "max_depth:           $(rf.max_depth)")
    println(io, "min_samples_leaf:    $(rf.min_samples_leaf)")
    println(io, "min_samples_split:   $(rf.min_samples_split)")
    println(io, "min_purity_increase: $(rf.min_purity_increase)")
    print(io,   "ensemble:            ") ; show(io, rf.ensemble)
end

################################################################################
# AdaBoost Stump Classifier

"""
    AdaBoostStumpClassifier(; n_iterations::Int=10,
                            rng=Random.GLOBAL_RNG,
                            calc_fi::Bool=true)
Adaboosted decision tree stumps. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `n_iterations`: number of iterations of AdaBoost
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.
- `calc_fi`: whether to calculate feature importances using `Mean Decrease in Impurity (MDI)`

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct AdaBoostStumpClassifier <: BaseClassifier
    n_iterations::Int
    rng::Random.AbstractRNG
    calc_fi::Bool
    ensemble::Union{Ensemble, Nothing}
    coeffs::Union{Vector{Float64}, Nothing}
    classes::Union{Vector, Nothing}
    AdaBoostStumpClassifier(; n_iterations=10, rng=Random.GLOBAL_RNG, calc_fi=true, ensemble=nothing, coeffs=nothing, classes=nothing) =
        new(n_iterations, mk_rng(rng), calc_fi, ensemble, coeffs, classes)
end

@declare_hyperparameters(AdaBoostStumpClassifier, [:n_iterations, :rng, :calc_fi])
get_classes(ada::AdaBoostStumpClassifier) = ada.classes

function fit!(ada::AdaBoostStumpClassifier, X, y)
    ada.ensemble, ada.coeffs =
        build_adaboost_stumps(y, X, ada.n_iterations; rng=ada.rng, calc_fi=ada.calc_fi)
    ada.classes = sort(unique(y))
    ada
end

predict(ada::AdaBoostStumpClassifier, X) =
    apply_adaboost_stumps(ada.ensemble, ada.coeffs, X)

predict_proba(ada::AdaBoostStumpClassifier, X) =
    apply_adaboost_stumps_proba(ada.ensemble, ada.coeffs, X, ada.classes)

function show(io::IO, ada::AdaBoostStumpClassifier)
    println(io, "AdaBoostStumpClassifier")
    println(io, "n_iterations: $(ada.n_iterations)")
    print(io,   "classes:      ") ; show(io, ada.classes)  ; println(io, "")
    print(io,   "ensemble:     ") ; show(io, ada.ensemble)
end

################################################################################
# Common functions

depth(dt::DecisionTreeClassifier)   = depth(dt.root)
depth(dt::DecisionTreeRegressor)    = depth(dt.root)

length(dt::DecisionTreeClassifier)  = length(dt.root)
length(dt::DecisionTreeRegressor)   = length(dt.root)

print_tree(dt::DecisionTreeClassifier, depth=-1; kwargs...) = print_tree(dt.root, depth; kwargs...)
print_tree(dt::DecisionTreeRegressor,  depth=-1; kwargs...) = print_tree(dt.root, depth; kwargs...)
print_tree(n::Nothing, depth=-1; kwargs...)                 = show(n)

#################################################################################
# Trait functions
y_convert(::Type{<: Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier}}, y) = y
y_convert(::Type{<: Union{DecisionTreeRegressor, RandomForestRegressor}}, y) = float.(y)

model(dt::Union{DecisionTreeClassifier, DecisionTreeRegressor}) = dt.root
model(rf::Union{RandomForestClassifier, RandomForestRegressor}) = rf.ensemble
model(ada::AdaBoostStumpClassifier) = ada.ensemble

score_fn(::Type{<: Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier}}) = accuracy
score_fn(::Type{<: Union{DecisionTreeRegressor, RandomForestRegressor}}) = R2

# score functino
R2(model::T, X::AbstractMatrix, y::AbstractVector) where {T <: Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier, DecisionTreeRegressor, RandomForestRegressor}}= 
    R2(y, predict(model, X))
accuracy(model::T, X::AbstractMatrix, y::AbstractVector) where {T <: Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier, DecisionTreeRegressor, RandomForestRegressor}}= 
    accuracy(y, predict(model, X))

const DecisionTreeEstimator = Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier, DecisionTreeRegressor, RandomForestRegressor}

"""
    feature_importances(trees::T; normalize::Bool = false) where {T <: DecisionTreeEstimator}

Return an vector of feature importances calculated by `Mean Decrease in Impurity (MDI)`.
If `calc_fi` was set false, this function returns an empty vector.

!!! warn
    The feature importnaces might be misleading because it is a biased methods.

See [Beware Default Random Forest Importances](https://explained.ai/rf-importance/index.html) for more detailed dicussion.
"""
feature_importances(trees::T; 
    normalize::Bool = false) where {T <: DecisionTreeEstimator} = 
    feature_importances(model(trees), normalize = normalize)

function sampling(X::AbstractMatrix, y::AbstractVector, n_sample::Int, sampling_rate::Float64 = 1.0)
    if iszero(n_sample)
        n_sample = ceil(Int, length(y) * sampling_rate)
    else
        n_sample > length(y) && @warn "Sampling size is larger than total number of samples; calculating with all samples."
        n_sample = min(n_sample, length(y))
    end
    samples = sample(1:length(y), n_sample, replace = false)
    if n_sample != length(y)
        y = y[samples]
        X = X[samples, :]
    end
    X, y
end

"""
    permutation_importances(
                            trees::T, 
                            X::AbstractMatrix,
                            y::AbstractVector; 
                            score::Function = score_fn(T),
                            n_iter::Int = 3, 
                            cv_score = nothing,
                            cv = nothing, 
                            n_sample::Int = 0,
                            sampling_rate::Float64 = 1.0
                            ) where {T <: DecisionTreeEstimator}

Calculate feature importances by shuffling each features.
* Keyword arguments:
1. `score`: a function to evaluating model performance with the form of `score(model, X, y)`. The default value is determined by function `score_fn`.
2. `cv_score`: a function to do cross validation. It's designed to be `cross_val_score` from `ScikitLearn.jl`, but any function in the form of `cv_score(model, X, y; scoring = score, cv = cv)` is okay.
3. `cv`: keyword argument for `cv_score`.
4. `n_sample`: maximum number of samples. The default is zero. 
5. `sampling_rate`: the proportion of sampling. If `n_sample` is zero, `sampling_rate` is used. The default is one.

# Return an `NamedTuple`
* Fields 
1. `mean`: mean of feature importances of each shuffle.
2. `std`: std of feature importances of each shuffle.
3. `scores`: scores of each shuffles
4. `base_scores`: scores of base models if using cross validation.

Evaluating with multiple shuffles is generally quite robust and much more efficient than using cross validation.
For algorithm details, please see [Permutation feature importanc](https://scikit-learn.org/stable/modules/permutation_importance.html).
"""
function permutation_importances(
                                trees::T, 
                                X::AbstractMatrix,
                                y::AbstractVector; 
                                score::Function = score_fn(T),
                                n_iter::Int = 3, 
                                cv_score = nothing,
                                cv = nothing, 
                                n_sample::Int = 0,
                                sampling_rate::Float64 = 1.0
                                ) where {T <: Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier, DecisionTreeRegressor, RandomForestRegressor}}
    
    length(y) == size(X, 1) || error(DimensionMismatch("X and y should have the same number of rows."))               
    X, y = sampling(X, y, n_sample, sampling_rate)
    y = y_convert(T, y)
    n_feat = size(X, 2)
    if cv_score === nothing
        base = score(trees, X, y)
        scores = Matrix{Float64}(undef, n_feat, n_iter)
        for i in 1:n_feat
            col = @view X[:, i]
            origin = copy(col)
            scores[i, :] = map(1:n_iter) do i
                shuffle!(col)
                score(trees, X, y)
            end
            X[:, i] = origin
        end
        (mean = reshape(mapslices(scores, dims = 2) do im
            base - mean(im)
        end, :), 
        std = reshape(mapslices(scores, dims = 2) do im
            std(im)
        end, :), 
        scores = scores)
    else
        n_iter = isa(cv, Number) ? cv : length(cv)
        base_scores = cv_score(trees, X, y; scoring = score, cv = cv)
        scores = Matrix{Float64}(undef, n_feat, n_iter)
        for i in 1:n_feat
            col = @view X[:, i]
            origin = copy(col)
            scores[i, :] = cv_score(trees, X, y; scoring = score, cv = cv)
            X[:, i] = origin
        end
        (mean = reshape(mapslices(scores, dims = 2) do im
            mean(base_scores) - mean(im)
        end, :), 
        std = reshape(mapslices(scores, dims = 2) do im
            sqrt((var(scores) + var(base_scores))/2)
        end, :), 
        scores = scores,
        base_scores = base_scores)
    end
end

"""
    dropcol_importances(
                        trees::T, 
                        X::AbstractMatrix,
                        y::AbstractVector;    
                        score::Function = score_fn(T),
                        cv_score = nothing,
                        cv = nothing, 
                        rng = nothing
                        ) where {T <: DecisionTreeEstimator}

Calculate feature importances by dropping each features.
* Keyword arguments:
1. `score`: a function to evaluating model performance with the form of `score(model, X, y)`. The default value is determined by function `score_fn`.
2. `cv_score`: a function to do cross validation. It's designed to be `cross_val_score` from `ScikitLearn.jl`, but any function in the form of `cv_score(model, X, y; scoring = score, cv = cv)` is okay.
3. `cv`: keyword argument for `cv_score`.
4. `rng`: specific seed for training new models. The default is `nothing` and a random number is selected.

# Return an `NamedTuple`
* Fields 
1. `mean`: mean of feature importances.
2. `std`: std of feature importances.
3. `scores`: scores of each feature if using cross validation.
4. `base_scores`: scores of base models if using cross validation.

!!! warn 
    The importnaces without cross validation may be quite biased if the model is overfitting.

See [Beware Default Random Forest Importances](https://explained.ai/rf-importance/index.html) and [rfpimp](https://github.com/parrt/random-forest-importances) for more detailed dicussion and alogrithm.
"""
function dropcol_importances(
                            trees::T, 
                            X::AbstractMatrix,
                            y::AbstractVector;    
                            score::Function = score_fn(T),
                            cv_score = nothing,
                            cv = nothing, 
                            rng = nothing
                            ) where {T <: Union{DecisionTreeClassifier, RandomForestClassifier, AdaBoostStumpClassifier, DecisionTreeRegressor, RandomForestRegressor}}
    
    length(y) == size(X, 1) || error(DimensionMismatch("X and y should have the same number of rows."))               
    y = y_convert(T, y)
    n_feat = size(X, 2)
    trees_ = deepcopy(trees)
    if isnothing(rng)
        rng = rand(1:typemax(Int))
    end
    trees_.rng = mk_rng(rng)
    trees_.calc_fi = false
    if cv_score === nothing
        fit!(trees_, X, y)
        base = score(trees_, X, y)
        im = Vector{Float64}(undef, n_feat)
        for i in 1:n_feat
            inds = deleteat!(collect(1:n_feat), i)
            X_new = X[:, inds]
            trees_.rng = mk_rng(rng)
            fit!(trees_, X_new, y)
            im[i] = base - score(trees_, X_new, y)
        end
        (mean = im, 
        std = std.(im))
    else
        n_iter = isa(cv, Number) ? cv : length(cv)
        base_scores = cv_score(trees, X, y; scoring = score, cv = cv)
        scores = Matrix{Float64}(undef, n_feat, n_iter)
        for i in 1:n_feat
            inds = deleteat!(collect(1:n_feat), i)
            X_new = X[:, inds]
            trees_.rng = mk_rng(rng)
            fit!(trees_, X_new, y)
            scores[i, :] = cv_score(trees_, X_new, y; scoring = score,  cv = cv)
        end
        (mean = reshape(mapslices(scores, dims = 2) do im
            mean(base_scores) - mean(im)
        end, :), 
        std = reshape(mapslices(scores, dims = 2) do im
            sqrt((var(scores) + var(base_scores))/2)
        end, :), 
        scores = scores,
        base_scores = base_scores)
    end
end