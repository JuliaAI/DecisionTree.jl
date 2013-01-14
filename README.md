DecisionTree.jl
========

Decision Tree Classifier in Julia

Implementation of the greedy ID3 algorithm with
* parallelized bagging (random forests)
* adaptive boosting (decision stumps)
* cross validation
* but no pruning

Adapted from [MILK: Machine Learning Toolkit](https://github.com/luispedro/milk)

# Usage Example
```julia
using DataFrames
using RDatasets
using DecisionTree

iris = data("datasets", "iris")
features = convert(Array{Float64,2}, matrix(iris[:, 2:5]));
labels = convert(Array{UTF8String,1}, iris[:, "Species"]);

# train full-tree classifier
model = build_tree(features, labels);
# apply learned model
apply_tree(model, [5.9,3.0,5.1,1.9])

# train random forest classifier, using 2 random features and 10 trees
model = build_forest(features, labels, 2, 10);
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests, using 2 random features, 10 trees and 3 folds
nfoldCV_forest(features, labels, 2, 10, 3)

# train adaptive-boosted decision stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(features, labels, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
nfoldCV_stumps(features, labels, 7, 3)
```

# Coming Soon

* Support for DataFrames
