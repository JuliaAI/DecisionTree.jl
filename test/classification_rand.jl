using Base.Test
using DecisionTree

n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = _int(features * weights);

maxdepth = 3
model = build_tree(labels, features, 0, maxdepth)
@test depth(model) == maxdepth

println("\n##### nfoldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3)
@test mean(accuracy) > 0.7

println("\n##### nfoldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3)
@test mean(accuracy) > 0.7

println("\n##### nfoldCV Adaboosted Stumps #####")
nfoldCV_stumps(labels, features, 7, 3)
