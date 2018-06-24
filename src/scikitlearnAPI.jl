import ScikitLearnBase: BaseClassifier, BaseRegressor, predict, predict_proba,
                        fit!, get_classes, @declare_hyperparameters

################################################################################
# Classifier

"""
    DecisionTreeClassifier(; pruning_purity_threshold=1.0,
                           max_depth::Int=-1,
                           min_samples_leaf::Int=1,
                           min_samples_split::Int=2,
                           min_purity_increase::Float=0.0,
                           n_subfeatures::Int=0,
                           rng=Base.GLOBAL_RNG)
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
    rng::AbstractRNG
    root::Union{LeafOrNode, Void}
    classes::Union{Vector, Void}
    DecisionTreeClassifier(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=1, min_samples_split=2,
                           min_purity_increase=0.0, n_subfeatures=0, rng=Base.GLOBAL_RNG, root=nothing, classes=nothing) =
        new(pruning_purity_threshold, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, n_subfeatures, mk_rng(rng), root, classes)
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
@declare_hyperparameters(DecisionTreeClassifier,
                         [:pruning_purity_threshold, :max_depth, :min_samples_leaf,
                          :min_samples_split, :min_purity_increase, :rng])

function fit!(dt::DecisionTreeClassifier, X, y)
    dt.root = build_tree(y, X, dt.n_subfeatures, dt.max_depth, dt.min_samples_leaf, dt.min_samples_split, dt.min_purity_increase; rng=dt.rng)
    if dt.pruning_purity_threshold < 1.0
        dt.root = prune_tree(dt.root, dt.pruning_purity_threshold)
    end
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
    println(io, "classes:                  $(dt.classes)")
    println(io, "root:                     $(dt.root)")
end

################################################################################
# Regression

"""
    DecisionTreeRegressor(; pruning_purity_threshold=1.0,
                          max_depth::Int-1,
                          min_samples_leaf::Int=5,
                          min_samples_split::Int=2,
                          min_purity_increase::Float=0.0,
                          n_subfeatures::Int=0,
                          rng=Base.GLOBAL_RNG)
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
    rng::AbstractRNG
    root::Union{LeafOrNode, Void}
    DecisionTreeRegressor(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=5,
                          min_samples_split=2, min_purity_increase=0.0, n_subfeatures=0, rng=Base.GLOBAL_RNG, root=nothing) =
        new(pruning_purity_threshold, max_depth, min_samples_leaf,
            min_samples_split, min_purity_increase, n_subfeatures, mk_rng(rng), root)
end

@declare_hyperparameters(DecisionTreeRegressor,
                         [:pruning_purity_threshold, :min_samples_leaf, :n_subfeatures,
                          :max_depth, :min_samples_split, :min_purity_increase, :rng])

function fit!(dt::DecisionTreeRegressor, X::Matrix, y::Vector)
    dt.root = build_tree(float.(y), X, dt.min_samples_leaf,
                         dt.n_subfeatures, dt.max_depth, dt.min_samples_split, dt.min_purity_increase; rng=dt.rng)
    if dt.pruning_purity_threshold < 1.0
        dt.root = prune_tree(dt.root, dt.pruning_purity_threshold)
    end
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
    println(io, "root:                     $(dt.root)")
end

################################################################################
# Random Forest Classification

"""
    RandomForestClassifier(; n_subfeatures::Int=0,
                           n_trees::Int=10,
                           partial_sampling::Float=0.7,
                           max_depth::Int=-1,
                           rng=Base.GLOBAL_RNG)
Random forest classification. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `n_subfeatures`: number of features to consider at random per split (default: keep all)
- `n_trees`: number of trees to train (default: 10)
- `partial_sampling`: fraction of samples to train each tree on (default: 0.7)
- `max_depth`: maximum depth of the decision trees (default: no maximum)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct RandomForestClassifier <: BaseClassifier
    n_subfeatures::Int
    n_trees::Int
    partial_sampling::Float64
    max_depth::Int
    rng::AbstractRNG
    ensemble::Union{Ensemble, Void}
    classes::Union{Vector, Void}
    RandomForestClassifier(; n_subfeatures=0, n_trees=10, partial_sampling=0.7,
                           max_depth=-1, rng=Base.GLOBAL_RNG, ensemble=nothing, classes=nothing) =
        new(n_subfeatures, n_trees, partial_sampling, max_depth, mk_rng(rng), ensemble, classes)
end

get_classes(rf::RandomForestClassifier) = rf.classes
@declare_hyperparameters(RandomForestClassifier,
                         [:n_subfeatures, :n_trees, :partial_sampling, :max_depth,
                          :rng])

function fit!(rf::RandomForestClassifier, X::Matrix, y::Vector)
    rf.ensemble = build_forest(y, X, rf.n_subfeatures, rf.n_trees,
                               rf.partial_sampling, rf.max_depth; rng=rf.rng)
    rf.classes = sort(unique(y))
    rf
end

predict_proba(rf::RandomForestClassifier, X) = 
    apply_forest_proba(rf.ensemble, X, rf.classes)

predict(rf::RandomForestClassifier, X) = apply_forest(rf.ensemble, X)

function show(io::IO, rf::RandomForestClassifier)
    println(io, "RandomForestClassifier")
    println(io, "n_trees:          $(rf.n_trees)")
    println(io, "n_subfeatures:    $(rf.n_subfeatures)")
    println(io, "max_depth:        $(rf.max_depth)")
    println(io, "partial_sampling: $(rf.partial_sampling)")
    println(io, "classes:          $(rf.classes)")
    println(io, "ensemble:         $(rf.ensemble)")
end

################################################################################
# Random Forest Regression

"""
    RandomForestRegressor(; n_subfeatures::Int=0,
                          n_trees::Int=10,
                          partial_sampling::Float=0.7,
                          max_depth::Int=-1,
                          min_samples_leaf::Int=5,
                          rng=Base.GLOBAL_RNG)
Random forest regression. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `n_subfeatures`: number of features to consider at random per split (default: keep all)
- `n_trees`: number of trees to train (default: 10)
- `partial_sampling`: fraction of samples to train each tree on (default: 0.7)
- `max_depth`: maximum depth of the decision trees (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 5)
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
    rng::AbstractRNG
    ensemble::Union{Ensemble, Void}
    RandomForestRegressor(; n_subfeatures=0, n_trees=10, partial_sampling=0.7,
                            max_depth=-1, min_samples_leaf=5, rng=Base.GLOBAL_RNG, ensemble=nothing) =
        new(n_subfeatures, n_trees, partial_sampling, max_depth, min_samples_leaf, mk_rng(rng), ensemble)
end

@declare_hyperparameters(RandomForestRegressor,
                         [:n_subfeatures, :n_trees, :min_samples_leaf, :partial_sampling,
                          # I'm not crazy about :rng being a hyperparameter,
                          # since it'll change throughout fitting, but it works
                          :max_depth, :rng])

function fit!(rf::RandomForestRegressor, X::Matrix, y::Vector)
    rf.ensemble = build_forest(float.(y), X, rf.n_subfeatures,
                               rf.n_trees, rf.min_samples_leaf, rf.partial_sampling,
                               rf.max_depth; rng=rf.rng)
    rf
end

predict(rf::RandomForestRegressor, X) = apply_forest(rf.ensemble, X)

function show(io::IO, rf::RandomForestRegressor)
    println(io, "RandomForestRegressor")
    println(io, "n_trees:          $(rf.n_trees)")
    println(io, "n_subfeatures:    $(rf.n_subfeatures)")
    println(io, "max_depth:        $(rf.max_depth)")
    println(io, "min_samples_leaf: $(rf.min_samples_leaf)")
    println(io, "partial_sampling: $(rf.partial_sampling)")
    println(io, "ensemble:         $(rf.ensemble)")
end

################################################################################
# AdaBoost Stump Classifier

"""
    AdaBoostStumpClassifier(; niterations::Int=0)

Adaboosted decision tree stumps. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `niterations`: number of iterations of AdaBoost
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct AdaBoostStumpClassifier <: BaseClassifier
    niterations::Int
    rng::AbstractRNG
    ensemble::Union{Ensemble, Void}
    coeffs::Union{Vector{Float64}, Void}
    classes::Union{Vector, Void}
    AdaBoostStumpClassifier(; niterations=10, rng=Base.GLOBAL_RNG, ensemble=nothing, coeffs=nothing, classes=nothing) =
        new(niterations, mk_rng(rng), ensemble, coeffs, classes)
end
@declare_hyperparameters(AdaBoostStumpClassifier, [:niterations, :rng])
get_classes(ada::AdaBoostStumpClassifier) = ada.classes

function fit!(ada::AdaBoostStumpClassifier, X, y)
    ada.ensemble, ada.coeffs =
        build_adaboost_stumps(y, X, ada.niterations; rng=ada.rng)
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
    println(io, "coeffs:       $(ada.coeffs)")
    println(io, "classes:      $(ada.classes)")
    println(io, "ensemble:     $(ada.ensemble)")
end

################################################################################
# Common functions

depth(dt::DecisionTreeClassifier) = depth(dt.root)
depth(dt::DecisionTreeRegressor) = depth(dt.root)
