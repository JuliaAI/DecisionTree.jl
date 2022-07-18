### Classification - Heterogeneously typed features (ints, floats, bools, strings)

@testset "heterogeneous.jl" begin

m, n = 10^2, 5

tf = [trues(Int(m/2)) falses(Int(m/2))]
inds = Random.randperm(StableRNG(1), m)
labels = string.(tf[inds])

features = Array{Any}(undef, m, n)
features[:,:] = randn(StableRNG(1), m, n)
features[:,2] = string.(tf[Random.randperm(StableRNG(1), m)])
features[:,3] = map(t -> round.(Int, t), features[:,3])
features[:,4] = tf[inds]

model = build_tree(labels, features; rng=StableRNG(1))
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9

n_subfeatures = 2
n_trees = 3
model = build_forest(labels, features, n_subfeatures, n_trees; rng=StableRNG(1))
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9

preds_MT = apply_forest(model, features, use_multithreading = true)
cm_MT = confusion_matrix(labels, preds_MT)
@test cm_MT.accuracy > 0.9

n_subfeatures = 7
model, coeffs = build_adaboost_stumps(labels, features, n_subfeatures; rng=StableRNG(1))
preds = apply_adaboost_stumps(model, coeffs, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9

end # @testset
