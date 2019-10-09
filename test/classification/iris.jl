# Classification Test - Iris Data Set
# https://archive.ics.uci.edu/ml/datasets/iris

@testset "iris.jl" begin

features, labels = load_data("iris")
labels = String.(labels)
classes = sort(unique(labels))
n = length(labels)

# train a decision stump (depth=1)
model = build_stump(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.6
@test depth(model) == 1
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# train full-tree classifier (over-fit)
model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy == 1.0
@test length(model) == 9
@test depth(model) == 5
@test typeof(preds) == Vector{String}
print_tree(model)
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# prune tree to 8 leaves
pruning_purity = 0.9
pt = prune_tree(model, pruning_purity)
@test length(pt) == 8
preds = apply_tree(pt, features)
cm = confusion_matrix(labels, preds)
@test 0.99 < cm.accuracy < 1.0

# prune tree to 3 leaves
pruning_purity = 0.6
pt = prune_tree(model, pruning_purity)
@test length(pt) == 3
preds = apply_tree(pt, features)
cm = confusion_matrix(labels, preds)
@test 0.95 < cm.accuracy < 1.0
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# prune tree to a stump, 2 leaves
pruning_purity = 0.5
pt = prune_tree(model, pruning_purity)
@test length(pt) == 2
preds = apply_tree(pt, features)
cm = confusion_matrix(labels, preds)
@test 0.66 < cm.accuracy < 1.0


# run n-fold cross validation for pruned tree
println("\n##### nfoldCV Classification Tree #####")
nfolds = 3
accuracy = nfoldCV_tree(labels, features, nfolds)
@test mean(accuracy) > 0.8

# train random forest classifier
n_trees = 10
n_subfeatures = 2
partial_sampling = 0.5
model = build_forest(labels, features, n_subfeatures, n_trees, partial_sampling)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95
@test typeof(preds) == Vector{String}
probs = apply_forest_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# run n-fold cross validation for forests
println("\n##### nfoldCV Classification Forest #####")
n_subfeatures = 2
n_trees = 10
n_folds = 3
partial_sampling = 0.5
accuracy = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees, partial_sampling)
@test mean(accuracy) > 0.9

# train adaptive-boosted decision stumps
n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations)
preds = apply_adaboost_stumps(model, coeffs, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9
@test typeof(preds) == Vector{String}
probs = apply_adaboost_stumps_proba(model, coeffs, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
println("\n##### nfoldCV Classification Adaboosted Stumps #####")
n_iterations = 15
nfolds = 3
accuracy = nfoldCV_stumps(labels, features, nfolds, n_iterations)
@test mean(accuracy) > 0.9

end # @testset
