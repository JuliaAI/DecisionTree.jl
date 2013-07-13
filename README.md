# DecisionTree.jl

Decision Tree Classifier in Julia

Implementation of the [ID3 algorithm](http://en.wikipedia.org/wiki/ID3_algorithm), with
* post pruning (pessimistic pruning)
* parallelized bagging (random forests)
* adaptive boosting (decision stumps)
* cross validation (n-fold)
* support for nominal and numerical datasets

Adapted from [MILK: Machine Learning Toolkit](https://github.com/luispedro/milk)

## Installation
You can install DecisionTree.jl using Julia's package manager
```julia
Pkg.add("DecisionTree")
```

## Usage Example
Load RDatasets and DecisionTree packages
```julia
using RDatasets
using DecisionTree
```
Separate Fisher's Iris dataset features and labels
```julia
iris = data("datasets", "iris")
features = matrix(iris[:, 2:5]);
labels = vector(iris[:, "Species"]);
```
Pruned tree classifier
```julia
# train full-tree classifier
model = build_tree(labels, features);
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9);
# pretty print of the tree
print_tree(model)
# apply learned model
apply_tree(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for pruned tree,
# using 90% purity threshold purning, and 3 CV folds
accuray = nfoldCV_tree(labels, features, 0.9, 3)
```
Random forest classifier
```julia
# train random forest classifier, using 2 random features and 10 trees
model = build_forest(labels, features, 2, 10);
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests,
# using 2 random features, 10 trees and 3 folds
accuray = nfoldCV_forest(labels, features, 2, 10, 3)
```
Adaptive-boosted decision stumps classifier
```julia
# train adaptive-boosted stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
accuray = nfoldCV_stumps(labels, features, 7, 3)
```

