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
                           nsubfeatures::Int=0,
                           rng=Base.GLOBAL_RNG)
Decision tree classifier. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `pruning_purity_threshold`: (post-pruning) merge leaves having `>=thresh` combined purity (default: no pruning)
- `max_depth`: maximum depth of the decision tree (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 1)
- `min_samples_split`: the minimum number of samples in needed for a split (default: 2)
- `min_purity_increase`: minimum purity needed for a split (default: 0.0)
- `nsubfeatures`: number of features to select at random (default: keep all)
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
    nsubfeatures::Int
    rng::AbstractRNG
    root::LeafOrNode
    # classes (in scikit-learn) === labels (in DecisionTree.jl)
    classes::Vector   # an arbitrary ordering of the labels 
    DecisionTreeClassifier(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=1, min_samples_split=2,
                           min_purity_increase=0.0, nsubfeatures=0, rng=Base.GLOBAL_RNG) =
        new(pruning_purity_threshold, max_depth, min_samples_leaf, min_samples_split,
            min_purity_increase, nsubfeatures, mk_rng(rng))
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
@declare_hyperparameters(DecisionTreeClassifier,
                         [:pruning_purity_threshold, :max_depth, :min_samples_leaf,
                          :min_samples_split, :min_purity_increase, :rng])

function fit!(dt::DecisionTreeClassifier, X, y)
    dt.root = build_tree(y, X, dt.nsubfeatures, dt.max_depth, dt.min_samples_leaf, dt.min_samples_split, dt.min_purity_increase; rng=dt.rng)
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

################################################################################
# Regression

"""
    DecisionTreeRegressor(; pruning_purity_threshold=1.0,
                          max_depth::Int-1,
                          min_samples_leaf::Int=5,
                          min_samples_split::Int=2,
                          min_purity_increase::Float=0.0,
                          nsubfeatures::Int=0,
                          rng=Base.GLOBAL_RNG)
Decision tree regression. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `pruning_purity_threshold`: (post-pruning) merge leaves having `>=thresh` combined purity (default: no pruning)
- `max_depth`: maximum depth of the decision tree (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 5)
- `min_samples_split`: the minimum number of samples in needed for a split (default: 2)
- `min_purity_increase`: minimum purity needed for a split (default: 0.0)
- `nsubfeatures`: number of features to select at random (default: keep all)
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
    nsubfeatures::Int
    rng::AbstractRNG
    root::LeafOrNode
    DecisionTreeRegressor(;pruning_purity_threshold=1.0, max_depth=-1, min_samples_leaf=5,
                          min_samples_split=2, min_purity_increase=0.0, nsubfeatures=0, rng=Base.GLOBAL_RNG) =
        new(pruning_purity_threshold, max_depth, min_samples_leaf,
            min_samples_split, min_purity_increase, nsubfeatures, mk_rng(rng))
end

@declare_hyperparameters(DecisionTreeRegressor,
                         [:pruning_purity_threshold, :min_samples_leaf, :nsubfeatures,
                          :max_depth, :min_samples_split, :min_purity_increase, :rng])

function fit!{T<:Real}(dt::DecisionTreeRegressor, X::Matrix, y::Vector{T})
    dt.root = build_tree(y, convert(Matrix{Float64}, X), dt.min_samples_leaf,
                         dt.nsubfeatures, dt.max_depth, dt.min_samples_split, dt.min_purity_increase; rng=dt.rng)
    if dt.pruning_purity_threshold < 1.0
        dt.root = prune_tree(dt.root, dt.pruning_purity_threshold)
    end
    dt
end

predict(dt::DecisionTreeRegressor, X) = apply_tree(dt.root, X)

################################################################################
# Random Forest Classification

"""
    RandomForestClassifier(; nsubfeatures::Int=0,
                           ntrees::Int=10,
                           partialsampling::Float=0.7,
                           max_depth::Int=-1,
                           rng=Base.GLOBAL_RNG)
Random forest classification. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `nsubfeatures`: number of features to select in each tree at random (default: keep all)
- `ntrees`: number of trees to train
- `partialsampling`: fraction of samples to train each tree on
- `max_depth`: maximum depth of the decision trees (default: no maximum)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
mutable struct RandomForestClassifier <: BaseClassifier
    nsubfeatures::Int
    ntrees::Int
    partialsampling::Float64
    max_depth::Int
    rng::AbstractRNG
    ensemble::Ensemble
    classes::Vector
    RandomForestClassifier(; nsubfeatures=0, ntrees=10, partialsampling=0.7,
                           max_depth=-1, rng=Base.GLOBAL_RNG) =
        new(nsubfeatures, ntrees, partialsampling, max_depth, mk_rng(rng))
end

get_classes(rf::RandomForestClassifier) = rf.classes
@declare_hyperparameters(RandomForestClassifier,
                         [:nsubfeatures, :ntrees, :partialsampling, :max_depth,
                          :rng])

function fit!(rf::RandomForestClassifier, X::Matrix, y::Vector)
    rf.ensemble = build_forest(y, X, rf.nsubfeatures, rf.ntrees,
                               rf.partialsampling, rf.max_depth; rng=rf.rng)
    rf.classes = sort(unique(y))
    rf
end

predict_proba(rf::RandomForestClassifier, X) = 
    apply_forest_proba(rf.ensemble, X, rf.classes)

predict(rf::RandomForestClassifier, X) = apply_forest(rf.ensemble, X)

################################################################################
# Random Forest Regression

"""
    RandomForestRegressor(; nsubfeatures::Int=0,
                          ntrees::Int=10,
                          partialsampling::Float=0.7,
                          max_depth::Int=-1,
                          min_samples_leaf::Int=5,
                          rng=Base.GLOBAL_RNG)
Random forest regression. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `nsubfeatures`: number of features to select in each tree at random (default:
  keep all)
- `ntrees`: number of trees to train
- `partialsampling`: fraction of samples to train each tree on
- `max_depth`: maximum depth of the decision trees (default: no maximum)
- `min_samples_leaf`: the minimum number of samples each leaf needs to have (default: 5)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `get_classes`
"""
mutable struct RandomForestRegressor <: BaseRegressor
    nsubfeatures::Int
    ntrees::Int
    partialsampling::Float64
    max_depth::Int
    min_samples_leaf::Int
    rng::AbstractRNG
    ensemble::Ensemble
    RandomForestRegressor(; nsubfeatures=0, ntrees=10, partialsampling=0.7, max_depth=-1, min_samples_leaf=5, rng=Base.GLOBAL_RNG) =
        new(nsubfeatures, ntrees, partialsampling, max_depth, min_samples_leaf, mk_rng(rng))
end

@declare_hyperparameters(RandomForestRegressor,
                         [:nsubfeatures, :ntrees, :min_samples_leaf, :partialsampling,
                          # I'm not crazy about :rng being a hyperparameter,
                          # since it'll change throughout fitting, but it works
                          :max_depth, :rng])

function fit!{T<:Real}(rf::RandomForestRegressor, X::Matrix, y::Vector{T})
    rf.ensemble = build_forest(y, convert(Matrix{Float64}, X), rf.nsubfeatures,
                               rf.ntrees, rf.min_samples_leaf, rf.partialsampling,
                               rf.max_depth; rng=rf.rng)
    rf
end

predict(rf::RandomForestRegressor, X) = apply_forest(rf.ensemble, X)

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
    ensemble::Ensemble
    coeffs::Vector{Float64}
    classes::Vector
    AdaBoostStumpClassifier(; niterations=10, rng=Base.GLOBAL_RNG) =
        new(niterations, mk_rng(rng))
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

################################################################################
# Common functions

depth(dt::DecisionTreeClassifier) = depth(dt.root)
depth(dt::DecisionTreeRegressor) = depth(dt.root)
