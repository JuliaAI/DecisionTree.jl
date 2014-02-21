using RDatasets
using DecisionTree

iris = dataset("datasets", "iris")
features = array(iris[:, 1:4]);
labels = array(iris[:, 5]);

# train full-tree classifier
model = build_tree(labels, features)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)
# pretty print of the tree
print_tree(model)
# apply learned model
apply_tree(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for pruned tree, using 90% purity threshold purning, and 3 CV folds
nfoldCV_tree(labels, features, 0.9, 3)

# train random forest classifier, using 2 random features, 10 trees and 0.5 of samples per tree (optional, defaults to 0.7)
model = build_forest(labels, features, 2, 10, 0.5)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests, using 2 random features, 10 trees, 3 folds, 0.5 of samples per tree (optional, defaults to 0.7)
nfoldCV_forest(labels, features, 2, 10, 3, 0.5)

# train adaptive-boosted decision stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
nfoldCV_stumps(labels, features, 7, 3)

