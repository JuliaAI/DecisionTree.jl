# Classification Test - Adult Data Set
# https://archive.ics.uci.edu/ml/datasets/adult

@testset "adult.jl" begin

adult = DelimitedFiles.readdlm("data/adult.csv", ',');

features = adult[:, 1:14];
labels = adult[:, 15];

model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.99

model = build_forest(labels, features, 3, 10)
preds = apply_forest(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.9

model, coeffs = build_adaboost_stumps(labels, features, 15);
preds = apply_adaboost_stumps(model, coeffs, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.8

println("\n##### 3 foldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3);
@test mean(accuracy) > 0.8

println("\n##### 3 foldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5);
@test mean(accuracy) > 0.8

println("\n##### nfoldCV Classification Adaboosted Stumps #####")
accuracy = nfoldCV_stumps(labels, features, 15, 3);
@test mean(accuracy) > 0.8

end # @testset
