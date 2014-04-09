using Base.Test
using DecisionTree

n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;
test_sample = randn(1,m)

# regression tree
model = build_tree(labels, features)
apply_tree(model, test_sample)
println("\n##### nfoldCV Regression Tree #####")
r2 = nfoldCV_tree(labels, features, 3)
@test mean(r2) > 0.6

# regression forest
model = build_forest(labels,features, 3, 10)
apply_forest(model, test_sample)
println("\n##### nfoldCV Regression Forest #####")
r2 = nfoldCV_forest(labels, features, 2, 10, 3)
@test mean(r2) > 0.8

