
@testset "random.jl" begin

Random.srand(16)

n,m = 10^3, 5;
features = rand(n,m);
weights = rand(-1:1,m);
labels = _int(features * weights);

model = build_stump(labels, features)
@test depth(model) == 1

max_depth = 3
model = build_tree(labels, features, 0, max_depth)
@test depth(model) == max_depth
print_tree(model, 3)

model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95

model = build_forest(labels, features)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.95

println("\n##### nfoldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3)
@test mean(accuracy) > 0.7

println("\n##### nfoldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3)
@test mean(accuracy) > 0.7

println("\n##### nfoldCV Adaboosted Stumps #####")
accuracy = nfoldCV_stumps(labels, features, 7, 3)
@test mean(accuracy) > 0.5

end # @testset
