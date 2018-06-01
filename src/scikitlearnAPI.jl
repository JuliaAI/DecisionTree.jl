import ScikitLearnBase: BaseClassifier, BaseRegressor, predict, predict_proba,
                        fit!, get_classes, @declare_hyperparameters

################################################################################
# Classifier

"""
    DecisionTreeClassifier(; pruning_purity_threshold=nothing,
                           nsubfeatures::Int=0, maxdepth::Int=-1,
                           rng=Base.GLOBAL_RNG)
Decision tree classifier. See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `pruning_purity_threshold`: merge leaves having `>=thresh` combined purity (default: no pruning)
- `nsubfeatures`: number of features to select at random (default: keep all)
- `maxdepth`: maximum depth of the decision tree (default: no maximum)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
struct DecisionTreeClassifier <: BaseClassifier
    pruning_purity_threshold::Nullable{Float64} # no pruning if null
    # Does nsubfeatures make sense for a stand-alone decision tree?
    nsubfeatures::Int
    maxdepth::Int
    rng::AbstractRNG
    root::LeafOrNode
    # classes (in scikit-learn) === labels (in DecisionTree.jl)
    classes::Vector   # an arbitrary ordering of the labels 
    # No pruning by default
    DecisionTreeClassifier(;pruning_purity_threshold=nothing, nsubfeatures=0,
                           maxdepth=-1, rng=Base.GLOBAL_RNG) =
        new(convert(Nullable{Float64}, pruning_purity_threshold), nsubfeatures,
            maxdepth, mk_rng(rng))
end

get_classes(dt::DecisionTreeClassifier) = dt.classes
@declare_hyperparameters(DecisionTreeClassifier,
                         [:pruning_purity_threshold, :nsubfeatures, :maxdepth,
                          :rng])

function fit!(dt::DecisionTreeClassifier, X, y)
    dt.root = build_tree(y, X, dt.nsubfeatures, dt.maxdepth; rng=dt.rng)
    if !isnull(dt.pruning_purity_threshold)
        dt.root = prune_tree(dt.root, get(dt.pruning_purity_threshold))
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
    DecisionTreeRegressor(; pruning_purity_threshold=nothing,
                          maxlabels::Int=5,
                          nsubfeatures::Int=0,
                          rng=Base.GLOBAL_RNG)
Decision tree regression. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `pruning_purity_threshold`: merge leaves having `>=thresh` combined purity (default: no pruning)
- `maxlabels`: maximum number of samples per leaf, split leaf if exceeded
- `nsubfeatures`: number of features to select at random (default: keep all)
- `maxdepth`: maximum depth of the decision tree (default: no maximum)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `get_classes`
"""
struct DecisionTreeRegressor <: BaseRegressor
    pruning_purity_threshold::Nullable{Float64}
    maxlabels::Int
    nsubfeatures::Int
    maxdepth::Int
    rng::AbstractRNG
    root::LeafOrNode
    # No pruning by default (I think purity_threshold=1.0 is a no-op, maybe
    # we could use that)
    DecisionTreeRegressor(;pruning_purity_threshold=nothing, maxlabels=5,
                          nsubfeatures=0, maxdepth=-1, rng=Base.GLOBAL_RNG) =
        new(convert(Nullable{Float64}, pruning_purity_threshold), maxlabels,
            nsubfeatures, maxdepth, mk_rng(rng))
end

@declare_hyperparameters(DecisionTreeRegressor,
                         [:pruning_purity_threshold, :maxlabels, :nsubfeatures,
                          :maxdepth, :rng])

function fit!{T<:Real}(dt::DecisionTreeRegressor, X::Matrix, y::Vector{T})
    # build_tree knows that its a regression problem by its argument types. I'm
    # not sure why X has to be Float64, but the method signature requires it
    # (as of April 2016).
    dt.root = build_tree(y, convert(Matrix{Float64}, X), dt.maxlabels,
                         dt.nsubfeatures, dt.maxdepth; rng=dt.rng)
    if !isnull(dt.pruning_purity_threshold)
        dt.root = prune_tree(dt.root, get(dt.pruning_purity_threshold))
    end
    dt
end

predict(dt::DecisionTreeRegressor, X) = apply_tree(dt.root, X)

################################################################################
# Random Forest Classification

"""
    RandomForestClassifier(; nsubfeatures::Int=0,
                           ntrees::Int=10,
                           partialsampling=0.7,
                           maxdepth=-1,
                           rng=Base.GLOBAL_RNG)
Random forest classification. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `nsubfeatures`: number of features to select in each tree at random (default:
  keep all)
- `ntrees`: number of trees to train
- `partialsampling`: fraction of samples to train each tree on
- `maxdepth`: maximum depth of the decision trees (default: no maximum)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `predict_proba`, `get_classes`
"""
struct RandomForestClassifier <: BaseClassifier
    nsubfeatures::Int
    ntrees::Int
    partialsampling::Float64
    maxdepth::Int
    rng::AbstractRNG
    ensemble::Ensemble
    classes::Vector
    RandomForestClassifier(; nsubfeatures=0, ntrees=10, partialsampling=0.7,
                           maxdepth=-1, rng=Base.GLOBAL_RNG) =
        new(nsubfeatures, ntrees, partialsampling, maxdepth, mk_rng(rng))
end

get_classes(rf::RandomForestClassifier) = rf.classes
@declare_hyperparameters(RandomForestClassifier,
                         [:nsubfeatures, :ntrees, :partialsampling, :maxdepth,
                          :rng])

function fit!(rf::RandomForestClassifier, X::Matrix, y::Vector)
    rf.ensemble = build_forest(y, X, rf.nsubfeatures, rf.ntrees,
                               rf.partialsampling, rf.maxdepth; rng=rf.rng)
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
                          maxlabels::Int=5,
                          ntrees::Int=10,
                          partialsampling=0.7,
                          maxdepth=-1,
                          rng=Base.GLOBAL_RNG)
Random forest regression. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:

- `nsubfeatures`: number of features to select in each tree at random (default:
  keep all)
- `maxlabels`: maximum number of samples per leaf, split leaf if exceeded
- `ntrees`: number of trees to train
- `partialsampling`: fraction of samples to train each tree on
- `maxdepth`: maximum depth of the decision trees (default: no maximum)
- `rng`: the random number generator to use. Can be an `Int`, which will be used
  to seed and create a new random number generator.

Implements `fit!`, `predict`, `get_classes`
"""
struct RandomForestRegressor <: BaseRegressor
    nsubfeatures::Int
    maxlabels::Int
    ntrees::Int
    partialsampling::Float64
    maxdepth::Int
    rng::AbstractRNG
    ensemble::Ensemble
    RandomForestRegressor(; nsubfeatures=0, ntrees=10, maxlabels=5, partialsampling=0.7, maxdepth=-1, rng=Base.GLOBAL_RNG) =
        new(nsubfeatures, maxlabels, ntrees, partialsampling, maxdepth,
            mk_rng(rng))
end

@declare_hyperparameters(RandomForestRegressor,
                         [:nsubfeatures, :ntrees, :maxlabels, :partialsampling,
                          # I'm not crazy about :rng being a hyperparameter,
                          # since it'll change throughout fitting, but it works
                          :maxdepth, :rng])

function fit!{T<:Real}(rf::RandomForestRegressor, X::Matrix, y::Vector{T})
    rf.ensemble = build_forest(y, convert(Matrix{Float64}, X), rf.nsubfeatures,
                               rf.ntrees, rf.maxlabels, rf.partialsampling,
                               rf.maxdepth; rng=rf.rng)
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
struct AdaBoostStumpClassifier <: BaseClassifier
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

# Cannot use Union{} in Julia 0.3
depth(dt::DecisionTreeClassifier) = depth(dt.root)
depth(dt::DecisionTreeRegressor) = depth(dt.root)
