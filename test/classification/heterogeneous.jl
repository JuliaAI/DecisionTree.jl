### Classification - Heterogeneously typed features (ints, floats, bools, strings)

using Base.Test
using DecisionTree

m, n = 10^2, 5

tf = [trues(Int(m/2)) falses(Int(m/2))]
inds = randperm(m)
labels = string.(tf[inds])

features = Array{Any}(m, n)
features[:,:] = randn(m, n)
features[:,2] = string.(tf[randperm(m)])
features[:,3] = map(t -> round(Int, t), features[:,3]) # can just become round.(Int, features[:,3]) when Julia 0.5 support is dropped
features[:,4] = tf[inds]

model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7

model = build_forest(labels, features, 2, 3)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7

model, coeffs = build_adaboost_stumps(labels, features, 7)
preds = apply_adaboost_stumps(model, coeffs, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7

