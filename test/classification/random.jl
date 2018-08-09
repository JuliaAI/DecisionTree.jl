
@testset "random.jl" begin

Random.seed!(16)

n,m = 10^3, 5;
features = rand(n,m);
weights = rand(-1:1,m);
labels = round.(Int, features * weights);

model = build_stump(labels, round.(Int, features))
preds = apply_tree(model, round.(Int, features))
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

# test RNG param of trees
n_subfeatures = 2
t1 = build_tree(labels, features, n_subfeatures; rng=10)
t2 = build_tree(labels, features, n_subfeatures; rng=10)
t3 = build_tree(labels, features, n_subfeatures; rng=5)
@test (length(t1) == length(t2)) && (depth(t1) == depth(t2))
@test (length(t1) != length(t3)) || (depth(t1) != depth(t3))

mt = Random.MersenneTwister(1)
t1 = build_tree(labels, features, n_subfeatures; rng=mt)
t3 = build_tree(labels, features, n_subfeatures; rng=mt)
@test (length(t1) != length(t3)) || (depth(t1) != depth(t3))


model = build_forest(labels, features)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95
@test typeof(preds) == Vector{Int}

n_subfeatures       = 3
n_trees             = 9
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
@test length(model) == n_trees

# test n_subfeatures
n_subfeatures       = 0
m_partial = build_forest(labels, features) # default sqrt(n_features)
m_full    = build_forest(labels, features, n_subfeatures)
@test all( length.(m_full.trees) .< length.(m_partial.trees) )

# test partial_sampling parameter, train on single sample
partial_sampling    = 1 / n
n_subfeatures       = 0
n_trees             = 1         # single tree test
max_depth           = -1
min_samples_leaf    = 1
min_samples_split   = 2
min_purity_increase = 0.0
partial = build_forest(
            labels, features,
            n_subfeatures,
            n_trees,
            partial_sampling,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)
@test typeof(partial.trees[1]) <: Leaf

# test RNG parameter for forests
n_subfeatures       = 2
n_trees             = 5
m1 = build_forest(labels, features,
        n_subfeatures,
        n_trees;
        rng=10)
m2 = build_forest(labels, features,
        n_subfeatures,
        n_trees;
        rng=10)
m3 = build_forest(labels, features,
        n_subfeatures,
        n_trees;
        rng=5)
@test length.(m1.trees) == length.(m2.trees)
@test depth.(m1.trees)  == depth.(m2.trees)
@test length.(m1.trees) != length.(m3.trees)


n_iterations = 15
model, coeffs = build_adaboost_stumps(labels, features, n_iterations);
preds = apply_adaboost_stumps(model, coeffs, features);
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7
@test typeof(preds) == Vector{Int}
@test length(model) == n_iterations

println("\n##### nfoldCV Classification Tree #####")
nfolds          = 3
pruning_purity  = 1.0
max_depth       = 3
accuracy  = nfoldCV_tree(labels, features, nfolds, pruning_purity, max_depth; rng=10)
accuracy2 = nfoldCV_tree(labels, features, nfolds, pruning_purity, max_depth; rng=10)
accuracy3 = nfoldCV_tree(labels, features, nfolds, pruning_purity, max_depth; rng=5)
@test mean(accuracy) > 0.7
@test accuracy == accuracy2
@test accuracy != accuracy3

println("\n##### nfoldCV Classification Forest #####")
nfolds          = 3
n_subfeatures   = 2
n_trees         = 10
accuracy  = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees; rng=10)
accuracy2 = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees; rng=10)
accuracy3 = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees; rng=5)
@test mean(accuracy) > 0.7
@test accuracy == accuracy2
@test accuracy != accuracy3

println("\n##### nfoldCV Adaboosted Stumps #####")
n_iterations = 15
n_folds = 3
accuracy  = nfoldCV_stumps(labels, features, n_folds, n_iterations; rng=10)
accuracy2 = nfoldCV_stumps(labels, features, n_folds, n_iterations; rng=10)
accuracy3 = nfoldCV_stumps(labels, features, n_folds, n_iterations; rng=5)
@test mean(accuracy) > 0.7
@test accuracy == accuracy2
@test accuracy != accuracy3

end # @testset
