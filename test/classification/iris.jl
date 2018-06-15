download("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", "iris.csv")
iris = readcsv("iris.csv");

features = iris[:, 1:4];
labels = iris[:, 5];

# train full-tree classifier (over-fit)
model = build_tree(labels, features);
preds = apply_tree(model, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.99
@test length(model) == 9
@test depth(model) == 5
print_tree(model)

# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, 0.9);
@test length(model) == 8

# run n-fold cross validation for pruned tree, using 90% purity threshold purning, and 3 CV folds
println("\n##### nfoldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3);
@test mean(accuracy) > 0.8

# train random forest classifier, using 2 random features, 10 trees and 0.5 of samples per tree (optional, defaults to 0.7)
model = build_forest(labels, features, 2, 10, 0.5);
preds = apply_forest(model, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.95

# run n-fold cross validation for forests, using 2 random features, 10 trees, 3 folds, 0.5 of samples per tree (optional, defaults to 0.7)
println("\n##### nfoldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5);
@test mean(accuracy) > 0.8

# train adaptive-boosted decision stumps, using 7 iterations
model, coeffs = build_adaboost_stumps(labels, features, 7);
preds = apply_adaboost_stumps(model, coeffs, features);

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
println("\n##### nfoldCV Classification Adaboosted Stumps #####")
accuracy = nfoldCV_stumps(labels, features, 7, 3);
@test mean(accuracy) > 0.7
