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
                           rng=Random.GLOBAL_RNG)

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
    root::Union{LeafOrNode, Nothing}
    classes::Union{Vector, Nothing}
    DecisionTreeClassifier(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=1, min_samples_split=2,
                           min_purity_increase=0.0, n_subfeatures=0, rng=Random.GLOBAL_RNG, root=nothing, classes=nothing) =
        new(pruning_purity_threshold, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, n_subfeatures, mk_rng(rng), root, classes)
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
@declare_hyperparameters(DecisionTreeClassifier,
                         [:pruning_purity_threshold, :max_depth, :min_samples_leaf,
                          :min_samples_split, :min_purity_increase, :rng])

function fit!(dt::DecisionTreeClassifier, X, y)
    n_samples, n_features = size(X)
    dt.root = build_tree(
        y, X,
        dt.n_subfeatures,
        dt.max_depth,
        dt.min_samples_leaf,
        dt.min_samples_split,
        dt.min_purity_increase;
        rng = dt.rng)

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
    print(io,   "classes:                  ") ; show(dt.classes) ; println()
    print(io,   "root:                     ") ; show(dt.root)
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
                          rng=Random.GLOBAL_RNG)
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
    root::Union{LeafOrNode, Nothing}
    DecisionTreeRegressor(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=5,
                          min_samples_split=2, min_purity_increase=0.0, n_subfeatures=0, rng=Random.GLOBAL_RNG, root=nothing) =
        new(pruning_purity_threshold,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            n_subfeatures,
            mk_rng(rng),
            root)
end

@declare_hyperparameters(DecisionTreeRegressor,
                         [:pruning_purity_threshold, :min_samples_leaf, :n_subfeatures,
                          :max_depth, :min_samples_split, :min_purity_increase, :rng])

function fit!(dt::DecisionTreeRegressor, X::Matrix, y::Vector)
    n_samples, n_features = size(X)
    dt.root = build_tree(
        float.(y), X,
        dt.n_subfeatures,
        dt.max_depth,
        dt.min_samples_leaf,
        dt.min_samples_split,
        dt.min_purity_increase;
        rng = dt.rng)
    dt.pruning_purity_threshold
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
    print(io,   "root:                     ") ; show(dt.root)
end

################################################################################
# Random Forest Classification

"""
    RandomForestClassifier(; n_subfeatures::Int=-1,
                           n_trees::Int=10,
                           partial_sampling::Float=0.7,
                           max_depth::Int=-1,
                           rng=Random.GLOBAL_RNG)
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
  to seed and create a new random number generator.

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
    rng::Random.AbstractRNG
    ensemble::Union{Ensemble, Nothing}
    classes::Union{Vector, Nothing}
    RandomForestClassifier(; n_subfeatures=-1, n_trees=10, partial_sampling=0.7,
                           max_depth=-1, min_samples_leaf=1, min_samples_split=2, min_purity_increase=0.0,
                           rng=Random.GLOBAL_RNG, ensemble=nothing, classes=nothing) =
        new(n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, mk_rng(rng), ensemble, classes)
end

get_classes(rf::RandomForestClassifier) = rf.classes
@declare_hyperparameters(RandomForestClassifier,
                         [:n_subfeatures, :n_trees, :partial_sampling, :max_depth,
                          :min_samples_leaf, :min_samples_split, :min_purity_increase,
                          :rng])

function fit!(rf::RandomForestClassifier, X::Matrix, y::Vector)
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
        rng = rf.rng)
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
    print(io,   "classes:             ") ; show(rf.classes)  ; println()
    print(io,   "ensemble:            ") ; show(rf.ensemble)
end

################################################################################
# Random Forest Regression

"""
    RandomForestRegressor(; n_subfeatures::Int=-1,
                          n_trees::Int=10,
                          partial_sampling::Float=0.7,
                          max_depth::Int=-1,
                          min_samples_leaf::Int=5,
                          rng=Random.GLOBAL_RNG)
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
  to seed and create a new random number generator.

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
    rng::Random.AbstractRNG
    ensemble::Union{Ensemble, Nothing}
    RandomForestRegressor(; n_subfeatures=-1, n_trees=10, partial_sampling=0.7,
                            max_depth=-1, min_samples_leaf=5, min_samples_split=2, min_purity_increase=0.0,
                            rng=Random.GLOBAL_RNG, ensemble=nothing) =
        new(n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, mk_rng(rng), ensemble)
end

@declare_hyperparameters(RandomForestRegressor,
                         [:n_subfeatures, :n_trees, :partial_sampling,
                          :min_samples_leaf, :min_samples_split, :min_purity_increase,
                          # I'm not crazy about :rng being a hyperparameter,
                          # since it'll change throughout fitting, but it works
                          :max_depth, :rng])

function fit!(rf::RandomForestRegressor, X::Matrix, y::Vector)
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
        rng = rf.rng)
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
    print(io,   "ensemble:            ") ; show(rf.ensemble)
end

################################################################################
# AdaBoost Stump Classifier

"""
    AdaBoostStumpClassifier(; n_iterations::Int=0)

Adaboosted decision tree stumps. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `n_iterations`: number of iterations of AdaBoost
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct AdaBoostStumpClassifier <: BaseClassifier
    n_iterations::Int
    rng::Random.AbstractRNG
    ensemble::Union{Ensemble, Nothing}
    coeffs::Union{Vector{Float64}, Nothing}
    classes::Union{Vector, Nothing}
    AdaBoostStumpClassifier(; n_iterations=10, rng=Random.GLOBAL_RNG, ensemble=nothing, coeffs=nothing, classes=nothing) =
        new(n_iterations, mk_rng(rng), ensemble, coeffs, classes)
end

@declare_hyperparameters(AdaBoostStumpClassifier, [:n_iterations, :rng])
get_classes(ada::AdaBoostStumpClassifier) = ada.classes

function fit!(ada::AdaBoostStumpClassifier, X, y)
    ada.ensemble, ada.coeffs =
        build_adaboost_stumps(y, X, ada.n_iterations; rng=ada.rng)
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
    print(io,   "classes:      ") ; show(ada.classes)  ; println()
    print(io,   "ensemble:     ") ; show(ada.ensemble)
end

################################################################################
# Common functions

depth(dt::DecisionTreeClassifier)   = depth(dt.root)
depth(dt::DecisionTreeRegressor)    = depth(dt.root)

length(dt::DecisionTreeClassifier)  = length(dt.root)
length(dt::DecisionTreeRegressor)   = length(dt.root)

print_tree(dt::DecisionTreeClassifier, depth=-1) = print_tree(dt.root, depth)
print_tree(dt::DecisionTreeRegressor,  depth=-1) = print_tree(dt.root, depth)
print_tree(n::Nothing, depth=-1)                 = show(n)
