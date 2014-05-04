# DecisionTree.jl

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
* currently, only features of type float are supported

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
features = matrix(iris[:, 1:4]);
labels = vector(iris[:, 5]);
```
Pruned Tree Classifier
```julia
# train full-tree classifier
model = build_tree(labels, features)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)
# pretty print of the tree
print_tree(model)
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
# train regression forest, using 2 random features, 10 trees, 3 folds,
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
