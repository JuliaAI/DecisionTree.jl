@testset "random.jl" begin

srand(5)

n, m = 10^3, 5 ;
features = Array{Any}(n, m);
features[:,:] = randn(n, m);
features[:,1] = round.(Integer, features[:,1]); # convert a column of integers
weights = rand(-2:2,m);
labels = float.(features * weights);            # cast to Array{Float64,1}

model = build_stump(labels, features)
@test depth(model) == 1

# over-fitting
min_samples_leaf = 1
model = build_tree(labels, features, min_samples_leaf)
preds = apply_tree(model, features);
@test R2(labels, preds) > 0.99      # R2: coeff of determination
### @test length(model) == n        # can / should this be enforced ???

# under-fitting
min_samples_leaf = 100
model = build_tree(labels, features, min_samples_leaf)
preds = apply_tree(model, features);
@test R2(labels, preds) < 0.8

min_samples_leaf = 5; max_depth = 3; nsubfeatures = 0;
model = build_tree(labels, features, min_samples_leaf, nsubfeatures, max_depth)
@test depth(model) == max_depth

min_samples_leaf = 1; nsubfeatures = 0; max_depth = -1; min_samples_split = 100;
model = build_tree(labels, features, min_samples_leaf, nsubfeatures, max_depth, min_samples_split)
preds = apply_tree(model, features);
@test R2(labels, preds) < 0.8

min_samples_leaf = 1; nsubfeatures = 0; max_depth = -1; min_samples_split = 2; min_purity_increase = 0.5;
model = build_tree(labels, features, min_samples_leaf, nsubfeatures, max_depth, min_samples_split, min_purity_increase)
preds = apply_tree(model, features);
@test R2(labels, preds) < 0.95

model = build_forest(labels, features)
preds = apply_forest(model, features)
@test R2(labels, preds) > 0.9

println("\n##### nfoldCV Regression Tree #####")
r2 = nfoldCV_tree(labels, features, 3)
@test mean(r2) > 0.6

println("\n##### nfoldCV Regression Forest #####")
r2 = nfoldCV_forest(labels, features, 2, 10, 3)
@test mean(r2) > 0.8

end # @testset
