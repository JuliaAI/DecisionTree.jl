@testset "random.jl" begin

Random.seed!(5)

n, m = 10^3, 5 ;
features = Array{Any}(undef, n, m);
features[:,:] = randn(n, m);
features[:,1] = round.(Integer, features[:,1]); # convert a column of integers
weights = rand(-2:2,m);
labels = float.(features * weights);            # cast to Array{Float64,1}

model = build_stump(labels, features)
@test depth(model) == 1

# over-fitting
min_samples_leaf    = 1
max_depth           = -1
n_subfeatures       = 0
model = build_tree(
        labels, features,
        n_subfeatures,
        max_depth,
        min_samples_leaf)
preds = apply_tree(model, features);
@test R2(labels, preds) > 0.99      # R2: coeff of determination
@test typeof(preds) <: Vector{Float64}
### @test length(model) == n        # can / should this be enforced ???

# under-fitting
min_samples_leaf    = 100
model = build_tree(
        labels, round.(Int, features),
        n_subfeatures,
        max_depth,
        min_samples_leaf)
preds = apply_tree(model, round.(Int, features));
@test R2(labels, preds) < 0.8

min_samples_leaf    = 5
max_depth           = 3
n_subfeatures       = 0
model = build_tree(
        labels, features,
        n_subfeatures,
        max_depth,
        min_samples_leaf)
@test depth(model) == max_depth

min_samples_leaf    = 1
n_subfeatures       = 0
max_depth           = -1
min_samples_split   = 300
model = build_tree(
        labels, features,
        n_subfeatures,
        max_depth,
        min_samples_leaf,
        min_samples_split)
preds = apply_tree(model, features);
@test R2(labels, preds) < 0.8

n_subfeatures       = 0
max_depth           = -1
min_samples_leaf    = 1
min_samples_split   = 2
min_purity_increase = 0.5
model = build_tree(
        labels, features,
        n_subfeatures,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
preds = apply_tree(model, features);
@test R2(labels, preds) < 0.95

# test RNG param of trees
n_subfeatures       = 2
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
@test R2(labels, preds) > 0.9
@test typeof(preds) <: Vector{Float64}

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
@test R2(labels, preds) > 0.9
@test length(model) == n_trees

# test n_subfeatures
n_trees             = 10
partial_sampling    = 1.0
max_depth           = -1
min_samples_leaf    = 10
n_subfeatures       = 1
m_partial = build_forest(
        labels, features,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf)
n_subfeatures       = 0
m_full = build_forest(
        labels, features,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf)
@test mean(depth.(m_full.trees)) < mean(depth.(m_partial.trees))

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

# test RNG parameter
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


println("\n##### nfoldCV Classification Tree #####")
nfolds          = 3
pruning_purity  = 1.0
max_depth       = 4
r2_1 = nfoldCV_tree(labels, features, nfolds, pruning_purity, max_depth; rng=10)
r2_2 = nfoldCV_tree(labels, features, nfolds, pruning_purity, max_depth; rng=10)
r2_3 = nfoldCV_tree(labels, features, nfolds, pruning_purity, max_depth; rng=5)
@test mean(r2_1) > 0.6
@test r2_1 == r2_2
@test r2_1 != r2_3

println("\n##### nfoldCV Regression Forest #####")
nfolds          = 3
n_subfeatures   = 2
n_trees         = 10
r2_1  = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees; rng=10)
r2_2 = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees; rng=10)
r2_3 = nfoldCV_forest(labels, features, nfolds, n_subfeatures, n_trees; rng=5)
@test mean(r2_1) > 0.8
@test r2_1 == r2_2
@test r2_1 != r2_3

end # @testset
