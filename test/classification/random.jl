
@testset "random.jl" begin

Random.srand(16)

n,m = 10^3, 5;
features = rand(n,m);
weights = rand(-1:1,m);
labels = _int(features * weights);

model = build_stump(labels, features)
@test depth(model) == 1

max_depth = 3
model = build_tree(labels, features, 0, max_depth)
@test depth(model) == max_depth
print_tree(model, 3)

model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95
@test typeof(preds) == Vector{Int}

model = build_forest(labels, features)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95
@test typeof(preds) == Vector{Int}

n_subfeatures       = 3
n_trees             = 10
partial_sampling    = 0.7
max_depth           = -1
min_samples_leaf    = 5
min_samples_split   = 2
min_purity_increase = 0.0
model = build_forest(
        labels, features,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95

n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations);
preds = apply_adaboost_stumps(model, coeffs, features);
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7
@test typeof(preds) == Vector{Int}

println("\n##### nfoldCV Classification Tree #####")
pruning_purity = 0.9
nfolds = 3
accuracy = nfoldCV_tree(labels, features, pruning_purity, nfolds)
@test mean(accuracy) > 0.7

println("\n##### nfoldCV Classification Forest #####")
n_subfeatures = 2
n_trees = 10
nfolds = 3
accuracy = nfoldCV_forest(labels, features, n_subfeatures, n_trees, nfolds)
@test mean(accuracy) > 0.7

println("\n##### nfoldCV Adaboosted Stumps #####")
n_iterations = 15
nfolds = 3
accuracy = nfoldCV_stumps(labels, features, n_iterations, nfolds)
@test mean(accuracy) > 0.7

end # @testset
