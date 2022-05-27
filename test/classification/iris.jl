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
f1 = feature_importances(model)
p1 = permutation_importances(model, labels, features, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f1, p1) > 0.99

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
f1 = feature_importances(model)
p1 = permutation_importances(model, labels, features, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f1, p1) > 0.9

# prune tree to 8 leaves
pruning_purity = 0.9
pt = prune_tree(model, pruning_purity)
@test length(pt) == 8
preds = apply_tree(pt, features)
cm = confusion_matrix(labels, preds)
@test 0.99 < cm.accuracy < 1.0
f2 = feature_importances(pt)
p2 = permutation_importances(pt, labels, features, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test isapprox(f2, f1.+ [0, 0, 0, (47*log(47/48) + log(1/48))/150])
@test similarity(f2, p2) > 0.9

# prune tree to 3 leaves
pruning_purity = 0.6
pt = prune_tree(model, pruning_purity)
@test length(pt) == 3
preds = apply_tree(pt, features)
cm = confusion_matrix(labels, preds)
@test 0.95 < cm.accuracy < 1.0
probs = apply_tree_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)
f1 = feature_importances(pt)
p1 = permutation_importances(pt, labels, features, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f1, p1) > 0.9

# prune tree to a stump, 2 leaves
pruning_purity = 0.5
pt = prune_tree(model, pruning_purity)
@test length(pt) == 2
preds = apply_tree(pt, features)
cm = confusion_matrix(labels, preds)
@test 0.66 < cm.accuracy < 1.0
f1 = feature_importances(pt)
p1 = permutation_importances(pt, labels, features, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f1, p1) > 0.9
@test length(findall(x -> !isapprox(x, 0, atol = 1e-7), f1)) == 1

# run n-fold cross validation for pruned tree
println("\n##### nfoldCV Classification Tree #####")
nfolds = 3
accuracy = nfoldCV_tree(labels, features, nfolds; rng=StableRNG(1))
@test mean(accuracy) > 0.8

# train random forest classifier
n_trees = 10
n_subfeatures = 2
partial_sampling = 0.5
model = build_forest(labels, features, n_subfeatures, n_trees, partial_sampling; rng=StableRNG(2))
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95
@test typeof(preds) == Vector{String}
probs = apply_forest_proba(model, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)
f1 = feature_importances(model)
p1 = permutation_importances(model, labels, features, (model, y, X)->accuracy(y, apply_forest(model, X))).mean
@test similarity(f1, p1) > 0.8

# run n-fold cross validation for forests
println("\n##### nfoldCV Classification Forest #####")
n_subfeatures = 2
n_trees = 10
n_folds = 3
partial_sampling = 0.5
accuracy = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees, partial_sampling; rng=StableRNG(1))
@test mean(accuracy) > 0.9

# train adaptive-boosted decision stumps
n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations; rng=StableRNG(1))
preds = apply_adaboost_stumps(model, coeffs, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9
@test typeof(preds) == Vector{String}
probs = apply_adaboost_stumps_proba(model, coeffs, features, classes)
@test reshape(sum(probs, dims=2), n) ≈ ones(n)
f1 = feature_importances(model)
p1 = permutation_importances((model, coeffs), labels, features, (model, y, X)->accuracy(y, apply_adaboost_stumps(model, X))).mean
@test similarity(f1, p1) > 0.9

# run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
println("\n##### nfoldCV Classification Adaboosted Stumps #####")
n_iterations = 15
nfolds = 3
accuracy = nfoldCV_stumps(labels, features, nfolds, n_iterations; rng=StableRNG(1))
@test mean(accuracy) > 0.85

# feature importances
f1 = features[:, 1:3]
model = build_tree(labels, f1)
@test argmax(feature_importances(model)) == argmax(permutation_importances(model, labels, f1).mean) == argmax(dropcol_importances(model, labels, f1).mean)
n_trees = 10
n_subfeatures = 2
partial_sampling = 0.5
model = build_forest(labels, f1, n_subfeatures, n_trees, partial_sampling)
@test argmax(feature_importances(model)) == argmax(permutation_importances(model, labels, f1).mean) == argmax(dropcol_importances(model, labels, f1).mean)
n_iterations = 15
model = build_adaboost_stumps(labels, f1, n_iterations)
@test argmax(feature_importances(model)) == argmax(permutation_importances(model, labels, f1).mean) == argmax(dropcol_importances(model, labels, f1).mean)

end # @testset
