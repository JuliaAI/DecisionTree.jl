### Classification - Heterogeneously typed features (ints, floats, bools, strings)

using Base.Test
using DecisionTree

m, n = 10^2, 5;

tf = [trues(Int(m/2)) falses(Int(m/2))];
inds = randperm(m);
labels = string.(tf[inds]);

features = Array{Any}(m, n);
features[:,:] = randn(m, n);
features[:,2] = string.(tf[randperm(m)]);
features[:,3] = round(Int, features[:,3]);
features[:,4] = tf[inds];

model = tree(labels, features)
preds = apply(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7

model = forest(labels, features,2,3)
preds = apply(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7

model, coeffs = adaboost_stumps(labels, features, 7)
preds = apply(model, coeffs, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.7

