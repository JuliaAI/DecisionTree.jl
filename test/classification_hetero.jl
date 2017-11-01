### Classification - Heterogeneously typed features (ints, floats, bools, strings)

using DecisionTree

m, n = 10^2, 5;

tf = [trues(Int(m/2)) falses(Int(m/2))];
inds = randperm(m);
labels = string.(tf[inds]);

features = Array{Any}(m, n);
features[:,:] = randn(m, n);
features[:,2] = string.(tf[randperm(m)]);
features[:,3] = round.(Int, features[:,3]);
features[:,4] = tf[inds];

build_tree(labels, features)
build_forest(labels, features,2,3)
build_stump(labels, features)

