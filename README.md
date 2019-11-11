# DecisionTree.jl

[![Build Status](https://travis-ci.org/bensadeghi/DecisionTree.jl.svg?branch=master)](https://travis-ci.org/bensadeghi/DecisionTree.jl)
[![Coverage Status](https://coveralls.io/repos/bensadeghi/DecisionTree.jl/badge.svg?branch=master)](https://coveralls.io/r/bensadeghi/DecisionTree.jl?branch=master)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pkg.julialang.org/docs/DecisionTree/pEDeB/0.8.1/)

Julia implementation of Decision Tree (CART) and Random Forest algorithms

Available via:
* [CombineML.jl](https://github.com/ppalmes/CombineML.jl) - a heterogeneous ensemble learning package
* [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) - a pure Julia machine learning framework
* [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) - Julia implementation of the scikit-learn API

## Classification
* pre-pruning (max depth, min leaf size)
* post-pruning (pessimistic pruning)
* multi-threaded bagging (random forests)
* adaptive boosting (decision stumps)
* cross validation (n-fold)
* support for mixed categorical and numerical data

## Regression
* pre-pruning (max depth, min leaf size)
* multi-threaded bagging (random forests)
* cross validation (n-fold)
* support for numerical features

**Note that regression is implied if labels/targets are of type Array{Float}**

## Installation
You can install DecisionTree.jl using Julia's package manager
```julia
Pkg.add("DecisionTree")
```

## ScikitLearn.jl API
DecisionTree.jl supports the [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) interface and algorithms (cross-validation, hyperparameter tuning, pipelines, etc.)

Available models: `DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, AdaBoostStumpClassifier`.
See each model's help (eg. `?DecisionTreeRegressor` at the REPL) for more information

### Classification Example
Load DecisionTree package
```julia
using DecisionTree
```
Separate Fisher's Iris dataset features and labels
```julia
features, labels = load_data("iris")    # also see "adult" and "digits" datasets

# the data loaded are of type Array{Any}
# cast them to concrete types for better performance
features = float.(features)
labels   = string.(labels)
```
Pruned Tree Classifier
```julia
# train depth-truncated classifier
model = DecisionTreeClassifier(max_depth=2)
fit!(model, features, labels)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model, 5)
# apply learned model
predict(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
predict_proba(model, [5.9,3.0,5.1,1.9])
println(get_classes(model)) # returns the ordering of the columns in predict_proba's output
# run n-fold cross validation over 3 CV folds
# See ScikitLearn.jl for installation instructions
using ScikitLearn.CrossValidation: cross_val_score
accuracy = cross_val_score(model, features, labels, cv=3)
```

Also, have a look at these [classification](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Classifier_Comparison_Julia.ipynb) and [regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Decision_Tree_Regression_Julia.ipynb) notebooks.

## Native API
### Classification Example
Decision Tree Classifier
```julia
# train full-tree classifier
model = build_tree(labels, features)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model, 5)
# apply learned model
apply_tree(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_tree_proba(model, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# run 3-fold cross validation of pruned tree,
n_folds=3
accuracy = nfoldCV_tree(labels, features, n_folds)

# set of classification parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 1)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)
n_subfeatures=0; max_depth=-1; min_samples_leaf=1; min_samples_split=2
min_purity_increase=0.0; pruning_purity = 1.0

model    =   build_tree(labels, features,
                        n_subfeatures,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase)

accuracy = nfoldCV_tree(labels, features,
                        n_folds,
                        pruning_purity,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase)
```
Random Forest Classifier
```julia
# train random forest classifier
# using 2 random features, 10 trees, 0.5 portion of samples per tree, and a maximum tree depth of 6
model = build_forest(labels, features, 2, 10, 0.5, 6)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_forest_proba(model, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# run 3-fold cross validation for forests, using 2 random features per split
n_folds=3; n_subfeatures=2
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

# set of classification parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
n_subfeatures=-1; n_trees=10; partial_sampling=0.7; max_depth=-1
min_samples_leaf=5; min_samples_split=2; min_purity_increase=0.0

model    =   build_forest(labels, features,
                          n_subfeatures,
                          n_trees,
                          partial_sampling,
                          max_depth,
                          min_samples_leaf,
                          min_samples_split,
                          min_purity_increase)

accuracy = nfoldCV_forest(labels, features,
                          n_folds,
                          n_subfeatures,
                          n_trees,
                          partial_sampling,
                          max_depth,
                          min_samples_leaf,
                          min_samples_split,
                          min_purity_increase)
```
Adaptive-Boosted Decision Stumps Classifier
```julia
# train adaptive-boosted stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_adaboost_stumps_proba(model, coeffs, [5.9,3.0,5.1,1.9], ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
# run 3-fold cross validation for boosted stumps, using 7 iterations
n_iterations=7; n_folds=3
accuracy = nfoldCV_stumps(labels, features,
                          n_folds,
                          n_iterations)
```

### Regression Example
```julia
n, m = 10^3, 5
features = randn(n, m)
weights = rand(-2:2, m)
labels = features * weights
```
Regression Tree
```julia
# train regression tree
model = build_tree(labels, features)
# apply learned model
apply_tree(model, [-0.9,3.0,5.1,1.9,0.0])
# run 3-fold cross validation, returns array of coefficients of determination (R^2)
n_folds = 3
r2 = nfoldCV_tree(labels, features, n_folds)

# set of regression parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)
n_subfeatures = 0; max_depth = -1; min_samples_leaf = 5
min_samples_split = 2; min_purity_increase = 0.0; pruning_purity = 1.0

model = build_tree(labels, features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase)

r2 =  nfoldCV_tree(labels, features,
                   n_folds,
                   pruning_purity,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase)
```
Regression Random Forest
```julia
# train regression forest, using 2 random features, 10 trees,
# averaging of 5 samples per leaf, and 0.7 portion of samples per tree
model = build_forest(labels, features, 2, 10, 0.7, 5)
# apply learned model
apply_forest(model, [-0.9,3.0,5.1,1.9,0.0])
# run 3-fold cross validation on regression forest, using 2 random features per split
n_subfeatures=2; n_folds=3
r2 = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

# set of regression build_forest() parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
n_subfeatures=-1; n_trees=10; partial_sampling=0.7; max_depth=-1
min_samples_leaf=5; min_samples_split=2; min_purity_increase=0.0

model = build_forest(labels, features,
                     n_subfeatures,
                     n_trees,
                     partial_sampling,
                     max_depth,
                     min_samples_leaf,
                     min_samples_split,
                     min_purity_increase)

r2 =  nfoldCV_forest(labels, features,
                     n_folds,
                     n_subfeatures,
                     n_trees,
                     partial_sampling,
                     max_depth,
                     min_samples_leaf,
                     min_samples_split,
                     min_purity_increase)
```

## Saving Models
Models can be saved to disk and loaded back with the use of the [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) package.
```julia
using JLD2
@save "model_file.jld2" model
```
Note that even though features and labels of type `Array{Any}` are supported, it is highly recommended that data be cast to explicit types (ie with `float.(), string.()`, etc). This significantly improves model training and prediction execution times, and also drastically reduces the size of saved models.
