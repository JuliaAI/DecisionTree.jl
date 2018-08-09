# Test parallelization of random forests

@testset "parallel.jl" begin

Distributed.addprocs(1)
@test Distributed.nprocs() > 1

Distributed.@everywhere using DecisionTree

Random.seed!(16)

# Classification
n,m = 10^3, 5;
features = rand(n,m);
weights = rand(-1:1,m);
labels = round.(Int, features * weights);

model = build_forest(labels, features, 2, 10);
preds = apply_forest(model, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.8


# Regression
n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;

model = build_forest(labels, features, 2, 10);
preds = apply_forest(model, features);
@test R2(labels, preds) > 0.8

end # @testset
