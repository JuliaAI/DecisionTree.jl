# Classification Test - Adult Data Set
# https://archive.ics.uci.edu/ml/datasets/adult

@testset "adult.jl" begin

features, labels = load_data("adult")

model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.99

features = string.(features)
labels   = string.(labels)

n_subfeatures = 3
n_trees = 5
model = build_forest(labels, features, n_subfeatures, n_trees)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9

n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations);
preds = apply_adaboost_stumps(model, coeffs, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.8

println("\n##### 3 foldCV Classification Tree #####")
pruning_purity = 0.9
nfolds = 3
accuracy = nfoldCV_tree(labels, features, nfolds, pruning_purity; verbose=false);
@test mean(accuracy) > 0.8

println("\n##### 3 foldCV Classification Forest #####")
n_subfeatures = 2
n_trees = 10
n_folds = 3
partial_sampling = 0.5
accuracy = nfoldCV_forest(labels, features, n_folds, n_subfeatures, n_trees, partial_sampling; verbose=false)
@test mean(accuracy) > 0.8

println("\n##### nfoldCV Classification Adaboosted Stumps #####")
n_iterations = 15
n_folds = 3
accuracy = nfoldCV_stumps(labels, features, n_folds, n_iterations; verbose=false);
@test mean(accuracy) > 0.8

end # @testset
