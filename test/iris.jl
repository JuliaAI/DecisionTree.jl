using Base.Test
using RDatasets: dataset
using DecisionTree

iris = dataset("datasets", "iris");
features = convert(Array, iris[:, 1:4]);
labels = convert(Array, iris[:, 5]);

# train full-tree classifier
model = build_tree(labels, features)
# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model, 5)
# apply learned model
apply_tree(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for pruned tree, using 90% purity threshold purning, and 3 CV folds
println("\n##### nfoldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3)
@test mean(accuracy) > 0.8

# train random forest classifier, using 2 random features, 10 trees and 0.5 of samples per tree (optional, defaults to 0.7)
model = build_forest(labels, features, 2, 10, 0.5)
# apply learned model
apply_forest(model, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for forests, using 2 random features, 10 trees, 3 folds, 0.5 of samples per tree (optional, defaults to 0.7)
println("\n##### nfoldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5)
@test mean(accuracy) > 0.8

# train adaptive-boosted decision stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
# apply learned model
apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
println("\n##### nfoldCV Classification Adaboosted Stumps #####")
nfoldCV_stumps(labels, features, 7, 3)

