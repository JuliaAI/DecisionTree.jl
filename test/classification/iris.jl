# Classification Test - Iris Data Set
# https://archive.ics.uci.edu/ml/datasets/iris

@testset "iris.jl" begin

import DelimitedFiles

download("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", "iris.csv")
iris = DelimitedFiles.readdlm("iris.csv", ',')

features = iris[:, 1:4]
labels = iris[:, 5]

# train a decision stump (depth=1)
model = build_stump(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.6
@test depth(model) == 1

# train full-tree classifier (over-fit)
model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.99
@test length(model) == 9
@test depth(model) == 5
print_tree(model)

# prune tree: merge leaves having >= 90% combined purity (default: 100%)
model = prune_tree(model, -0.1)
@test length(model) == 8

# run n-fold cross validation for pruned tree
println("\n##### nfoldCV Classification Tree #####")
pruning_purity = 0.9
nfolds = 3
accuracy = nfoldCV_tree(labels, features, pruning_purity, nfolds)
@test mean(accuracy) > 0.8

# train random forest classifier
n_trees = 10
n_subfeatures = 2
partial_sampling = 0.5
model = build_forest(labels, features, n_subfeatures, n_trees, partial_sampling)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95

# run n-fold cross validation for forests
println("\n##### nfoldCV Classification Forest #####")
n_subfeatures = 2
n_trees = 10
n_folds = 3
partial_sampling = 0.5
accuracy = nfoldCV_forest(labels, features, n_subfeatures, n_trees, nfolds, partial_sampling)
@test mean(accuracy) > 0.8

# train adaptive-boosted decision stumps
n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations)
preds = apply_adaboost_stumps(model, coeffs, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
n_iterations = 15
nfolds = 3
println("\n##### nfoldCV Classification Adaboosted Stumps #####")
accuracy = nfoldCV_stumps(labels, features, n_iterations, nfolds)
@test mean(accuracy) > 0.9

end # @testset
