using Base.Test
using DataFrames
using RDatasets: dataset
using DecisionTree

iris = dataset("datasets", "iris");

@testset for Test in ["Iris Classification", "Iris Regression"]

    if Test == "Iris Classification"
        featurecols, targetcol = 1:4, 5
    elseif Test == "Iris Regression"
        break # break because heterogenous features are not yet supported for regression
        featurecols, targetcol = 2:5, 1
    end

    @testset "Arrays" begin

        features = convert(Array, iris[:, featurecols]);
        labels = convert(Array, iris[:, targetcol]);

        @testset "Decision Tree" begin
            # train full-tree classifier
            model = build_tree(labels, features)
            # prune tree: merge leaves having >= 90% combined purity (default: 100%)
            model = prune_tree(model, 0.9)
            # pretty print of the tree, to a depth of 5 nodes (optional)
            print_tree(model, 5)
            # apply learned model
            apply_tree(model, [5.9,3.0,5.1,1.9])
            apply_tree(model, features[1:10,:])
            apply_tree_proba(model, features[1:10,:], ["versicolor", "virginica", "setosa"])
            # run n-fold cross validation for pruned tree, using 90% purity threshold purning, and 3 CV folds
            accuracy = nfoldCV_tree(labels, features, 0.9, 3)
            @test mean(accuracy) > 0.8
        end

        @testset "Decision Forest" begin
            # train random forest classifier, using 2 random features, 10 trees and 0.5 of samples per tree (optional, defaults to 0.7)
            model = build_forest(labels, features, 2, 10, 0.5)
            # apply learned model
            apply_forest(model, [5.9,3.0,5.1,1.9])
            apply_forest(model, features[1:10,:])
            apply_forest_proba(model, features[1:10,:], ["versicolor", "virginica", "setosa"])
            # run n-fold cross validation for forests, using 2 random features, 10 trees, 3 folds, 0.5 of samples per tree (optional, defaults to 0.7)
            accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5)
            @test mean(accuracy) > 0.8
        end

        @testset "Adaboost Stumps" begin
            # train adaptive-boosted decision stumps, using 7 iterations
            model, coeffs = build_adaboost_stumps(labels, features, 7);
            # apply learned model
            apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
            apply_adaboost_stumps(model, coeffs, features[1:10,:])
            apply_adaboost_stumps_proba(model, coeffs, features[1:10,:], ["versicolor", "virginica", "setosa"])
            # run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
            accuracy = nfoldCV_stumps(labels, features, 7, 3)
            #@test mean(accuracy) > 0.8
        end

    end # Array

    @testset "DataFrame" begin

        cols = names(iris);
        target, features = cols[targetcol], cols[featurecols];

        @testset "Decision Tree" begin
            # train full-tree classifier
            model = build_tree(iris, target)
            # prune tree: merge leaves having >= 90% combined purity (default: 100%)
            model = prune_tree(model, 0.9)
            # pretty print of the tree, to a depth of 5 nodes (optional)
            print_tree(model, 5)
            # apply learned model
            apply_tree(model, [5.9,3.0,5.1,1.9])
            apply_tree(model, iris[1:10, features])
            apply_tree_proba(model, iris[1:10, features], ["versicolor", "virginica", "setosa"])
            # run n-fold cross validation for pruned tree, using 90% purity threshold purning, and 3 CV folds
            accuracy = nfoldCV_tree(iris, target, 0.9, 3)
            @test mean(accuracy) > 0.8
        end

        @testset "Decision Forest" begin
            # train random forest classifier, using 2 random features, 10 trees and 0.5 of samples per tree (optional, defaults to 0.7)
            model = build_forest(iris, target, 2, 10, 0.5)
            # apply learned model
            apply_forest(model, [5.9,3.0,5.1,1.9])
            apply_forest(model, iris[1:10, features])
            apply_forest_proba(model, iris[1:10, features], ["versicolor", "virginica", "setosa"])
            # run n-fold cross validation for forests, using 2 random features, 10 trees, 3 folds, 0.5 of samples per tree (optional, defaults to 0.7)
            accuracy = nfoldCV_forest(iris, target, 2, 10, 3, 0.5)
            @test mean(accuracy) > 0.8
        end

        @testset "Adaboost Stumps" begin
            # train adaptive-boosted decision stumps, using 7 iterations
            model, coeffs = build_adaboost_stumps(iris, target, 7);
            # apply learned model
            apply_adaboost_stumps(model, coeffs, [5.9,3.0,5.1,1.9])
            apply_adaboost_stumps(model, coeffs, iris[1:10, features])
            apply_adaboost_stumps_proba(model, coeffs, iris[1:10, features], ["versicolor", "virginica", "setosa"])
            # run n-fold cross validation for boosted stumps, using 7 iterations and 3 folds
            nfoldCV_stumps(iris, target, 7, 3)
            #@test mean(accuracy) > 0.8
        end

    end # DataFrame

end