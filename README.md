# DecisionTree.jl

[![Build Status](https://travis-ci.org/bensadeghi/DecisionTree.jl.svg?branch=master)](https://travis-ci.org/bensadeghi/DecisionTree.jl)
[![Coverage Status](https://coveralls.io/repos/bensadeghi/DecisionTree.jl/badge.svg?branch=master)](https://coveralls.io/r/bensadeghi/DecisionTree.jl?branch=master)

[![DecisionTree](http://pkg.julialang.org/badges/DecisionTree_0.5.svg)](http://pkg.julialang.org/?pkg=DecisionTree&ver=0.5)
[![DecisionTree](http://pkg.julialang.org/badges/DecisionTree_0.6.svg)](http://pkg.julialang.org/?pkg=DecisionTree&ver=0.6)
[![DecisionTree](http://pkg.julialang.org/badges/DecisionTree_0.7.svg)](http://pkg.julialang.org/?pkg=DecisionTree&ver=0.7)

Julia implementation of Decision Trees & Random Forests

## Classification
* pre-pruning (max depth, min leaf size)
* post-pruning (pessimistic pruning)
* parallelized bagging (random forests)
* adaptive boosting (decision stumps)
* cross validation (n-fold)
* support for mixed categorical and numerical data

## Regression
* pre-pruning (max depth, min leaf size)
* post-pruning (pessimistic pruning)
* parallelized bagging (random forests)
* cross validation (n-fold)
* support for numerical features

**Note that regression is implied if labels/targets are of type float**

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
Load RDatasets and DecisionTree packages
```julia
using RDatasets: dataset
using DecisionTree
```
Separate Fisher's Iris dataset features and labels
```julia
iris = dataset("datasets", "iris")
features = convert(Array, iris[:, 1:4]);
labels = convert(Array, iris[:, 5]);
```
Pruned Tree Classifier
```julia
# train depth-truncated classifier
model = DecisionTreeClassifier(max_depth=2)
fit!(model, features, labels)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model.root, 5)
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

Also have a look at these [classification](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Classifier_Comparison_Julia.ipynb), and [regression](https://github.com/cstjean/ScikitLearn.jl/blob/master/examples/Decision_Tree_Regression_Julia.ipynb) notebooks.

## Native API
### Classification Example
Pruned Tree Classifier
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
apply_tree_proba(model, [5.9,3.0,5.1,1.9], ["setosa", "versicolor", "virginica"])
# run n-fold cross validation for pruned tree,
# using 90% purity threshold pruning, and 3 CV folds
accuracy = nfoldCV_tree(labels, features, 0.9, 3)
```
Random Forest Classifier
```julia
# train random forest classifier
# using 2 random features, 10 trees, 0.5 portion of samples per tree (optional), and a maximum tree depth of 6 (optional)
model = build_forest(labels, features, 2, 10, 0.5, 6)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_forest_proba(model, [5.9,3.0,5.1,1.9], ["setosa", "versicolor", "virginica"])
# run n-fold cross validation for forests
# using 2 random features, 10 trees, 3 folds, and 0.5 portion of samples per tree (optional)
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5)
```
Adaptive-Boosted Decision Stumps Classifier
```julia
# train adaptive-boosted stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# get the probability of each label
apply_adaboost_stumps_proba(model, coeffs, [5.9,3.0,5.1,1.9], ["setosa", "versicolor", "virginica"])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
accuracy = nfoldCV_stumps(labels, features, 7, 3)
```

### Regression Example
```julia
n, m = 10^3, 5 ;
features = randn(n, m);
weights = rand(-2:2, m);
labels = features * weights;
```
Regression Tree
```julia
# train regression tree, using an averaging of 5 samples per leaf (optional)
model = build_tree(labels, features, 5)
# apply learned model
apply_tree(model, [-0.9,3.0,5.1,1.9,0.0])
# run n-fold cross validation, using 3 folds and averaging of 5 samples per leaf (optional)
# returns array of coefficients of determination (R^2)
r2 = nfoldCV_tree(labels, features, 3, 5)
```
Regression Random Forest
```julia
# train regression forest, using 2 random features, 10 trees,
# averaging of 5 samples per leaf (optional), and 0.7 portion of samples per tree (optional)
model = build_forest(labels, features, 2, 10, 5, 0.7)
# apply learned model
apply_forest(model, [-0.9,3.0,5.1,1.9,0.0])
# run n-fold cross validation on regression forest
# using 2 random features, 10 trees, 3 folds, averaging of 5 samples per leaf (optional),
# and 0.7 porition of samples per tree (optional)
# returns array of coefficients of determination (R^2)
r2 = nfoldCV_forest(labels, features, 2, 10, 3, 5, 0.7)
```
