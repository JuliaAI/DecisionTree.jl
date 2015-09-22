# DecisionTree.jl

[![Build Status](https://travis-ci.org/bensadeghi/DecisionTree.jl.svg?branch=master)](https://travis-ci.org/bensadeghi/DecisionTree.jl)
[![Coverage Status](https://coveralls.io/repos/bensadeghi/DecisionTree.jl/badge.svg?branch=master)](https://coveralls.io/r/bensadeghi/DecisionTree.jl?branch=master)

[![DecisionTree](http://pkg.julialang.org/badges/DecisionTree_0.3.svg)](http://pkg.julialang.org/?pkg=DecisionTree&ver=0.3)
[![DecisionTree](http://pkg.julialang.org/badges/DecisionTree_0.4.svg)](http://pkg.julialang.org/?pkg=DecisionTree&ver=0.4)

Decision Tree Classifier and Regressor in Julia

Implementation of the [ID3 algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

## Classifier
Includes:
* post pruning (pessimistic pruning)
* parallelized bagging (random forests)
* adaptive boosting (decision stumps)
* cross validation (n-fold)
* support for mixed nominal and numerical data

Adapted from [MILK: Machine Learning Toolkit](https://github.com/luispedro/milk)

## Regressor
Includes:
* parallelized bagging (random forests)
* cross validation (n-fold)
* support for numerical features

Note that regression is implied if labels/targets are of type float

## Installation
You can install DecisionTree.jl using Julia's package manager
```julia
Pkg.add("DecisionTree")
```

## Classification Example
Load RDatasets and DecisionTree packages
```julia
using RDatasets
using DecisionTree
```
Separate Fisher's Iris dataset features and labels
```julia
iris = dataset("datasets", "iris")
# on Julia v0.2
features = matrix(iris[:, 1:4]);
labels = vector(iris[:, 5]);
# on Julia v0.3
features = array(iris[:, 1:4]);
labels = array(iris[:, 5]);
```
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
# run n-fold cross validation for pruned tree,
# using 90% purity threshold purning, and 3 CV folds
accuracy = nfoldCV_tree(labels, features, 0.9, 3)
```
Random Forest Classifier
```julia
# train random forest classifier
# using 2 random features, 10 trees, and 0.5 portion of samples per tree (optional)
model = build_forest(labels, features, 2, 10, 0.5)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests
# using 2 random features, 10 trees, 3 folds and 0.5 of samples per tree (optional)
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5)
```
Adaptive-Boosted Decision Stumps Classifier
```julia
# train adaptive-boosted stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
accuracy = nfoldCV_stumps(labels, features, 7, 3)
```

## Regression Example
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
# run n-fold cross validation, using 3 folds, averaging of 5 samples per leaf (optional)
# returns array of coefficients of determination (R^2)
r2 = nfoldCV_tree(labels, features, 3, 5)
```
Regression Random Forest
```julia
# train regression forest, using 2 random features, 10 trees,
# averaging of 5 samples per leaf (optional), 0.7 of samples per tree (optional)
model = build_forest(labels,features, 2, 10, 5, 0.7)
# apply learned model
apply_forest(model, [-0.9,3.0,5.1,1.9,0.0])
# run n-fold cross validation on regression forest
# using 2 random features, 10 trees, 3 folds, averaging of 5 samples/leaf (optional),
# and 0.7 porition of samples per tree (optional)
# returns array of coefficients of determination (R^2)
r2 = nfoldCV_forest(labels, features, 2, 10, 3, 5, 0.7)
```
