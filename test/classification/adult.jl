# Classification Test - Adult Data Set
# https://archive.ics.uci.edu/ml/datasets/adult

@testset "adult.jl" begin

features, labels = load_data("adult")

model = build_tree(labels, features; rng=StableRNG(1))
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.99

features = string.(features)
labels   = string.(labels)

n_subfeatures = 3
n_trees = 5
model = build_forest(labels, features, n_subfeatures, n_trees; rng=StableRNG(1))
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9
f1 = impurity_importance(model)
p1 = permutation_importance(model, labels, features, (model, y, X)->accuracy(y, apply_forest(model, X)), rng=StableRNG(1)).mean

n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations; rng=StableRNG(1));
preds = apply_adaboost_stumps(model, coeffs, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.8
f2 = impurity_importance(model, coeffs)
p2 = permutation_importance(model, labels, features, (model, y, X)->accuracy(y, apply_forest(model, X)), rng=StableRNG(1)).mean

@test similarity(p2, p2) > 0.8
@test similarity(f1, f2) < 0.8

println("\n##### 3 foldCV Classification Tree #####")
pruning_purity = 0.9
nfolds = 3
accuracy1 = nfoldCV_tree(labels, features, nfolds, pruning_purity; rng=StableRNG(1), verbose=false);
@test mean(accuracy1) > 0.8

println("\n##### 3 foldCV Classification Forest #####")
n_subfeatures = 2
n_trees = 10
n_folds = 3
partial_sampling = 0.5
accuracy1 = nfoldCV_forest(labels, features, n_folds, n_subfeatures, n_trees, partial_sampling; rng=StableRNG(1), verbose=false)
@test mean(accuracy1) > 0.8

println("\n##### nfoldCV Classification Adaboosted Stumps #####")
n_iterations = 15
n_folds = 3
accuracy1 = nfoldCV_stumps(labels, features, n_folds, n_iterations; rng=StableRNG(1), verbose=false);
@test mean(accuracy1) > 0.8

end # @testset
