using Base.Test
using DecisionTree

n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;

println("\n##### nfoldCV Regression Tree #####")
r2 = nfoldCV_tree(labels, features, 3)
@test mean(r2) > 0.6

println("\n##### nfoldCV Regression Forest #####")
r2 = nfoldCV_forest(labels, features, 2, 10, 3)
@test mean(r2) > 0.8
