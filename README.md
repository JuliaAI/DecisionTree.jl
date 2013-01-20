DecisionTree.jl
========

Decision Tree Classifier in Julia

Implementation of the greedy ID3 algorithm, with
* post pruning (pessimistic pruning)
* parallelized bagging (random forests)
* adaptive boosting (decision stumps)
* cross validation

Adapted from [MILK: Machine Learning Toolkit](https://github.com/luispedro/milk)

# Usage Example
```julia
using RDatasets
using DecisionTree

iris = data("datasets", "iris")
features = matrix(iris[:, 2:5]);
labels = vector(iris[:, "Species"]);

# train full-tree classifier
model = build_tree(labels, features);
# prune tree: merge leaves having > 90% combined purity (default 100%)
model = prune_tree(model, 0.9);
# apply learned model
apply_tree(model, [5.9,3.0,5.1,1.9])

# train random forest classifier, using 2 random features and 10 trees
model = build_forest(labels, features, 2, 10);
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests, using 2 random features, 10 trees and 3 folds
nfoldCV_forest(labels, features, 2, 10, 3)

# train adaptive-boosted decision stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
nfoldCV_stumps(labels, features, 7, 3)
```

# Coming Soon

* Support for missing values, DataFrames
