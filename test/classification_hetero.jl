### Classification - Heterogeneously typed features (ints, floats, bools, strings)

using Base.Test
using DecisionTree
using DataFrames

m, n = 10^2, 5

tf = [trues(Int(m/2)) falses(Int(m/2))]
inds = randperm(m)
labels = string.(tf[inds])

features = Array{Any}(m, n)
features[:,:] = randn(m, n)
features[:,2] = string.(tf[randperm(m)])
features[:,3] = round.(Int, features[:,3])
features[:,4] = tf[inds]

@testset "Classification with heterogenous features" begin

    @testset "Arrays" begin

        @testset "Decision Tree" begin
            model = build_tree(labels, features)
            preds = apply_tree(model, features)
            cm = confusion_matrix(labels, preds)
            @test cm.accuracy > 0.7
        end

        @testset "Decision Forest" begin
            model = build_forest(labels, features,2,3)
            preds = apply_forest(model, features)
            cm = confusion_matrix(labels, preds)
            @test cm.accuracy > 0.7
        end

        @testset "Adaboost Stumps" begin
            model, coeffs = build_adaboost_stumps(labels, features, 7)
            preds = apply_adaboost_stumps(model, coeffs, features)
            cm = confusion_matrix(labels, preds)
            @test cm.accuracy > 0.7
        end

    end

    @testset "DataFrame" begin

        df = hcat(DataFrame(features), DataFrame(target = labels))

        @testset "Decision Tree" begin
            model = build_tree(df, :target)
            preds = apply_tree(model, df)
            cm = confusion_matrix(df[:target], preds)
            @test cm.accuracy > 0.7
        end

        @testset "Decision Forest" begin
            model = build_forest(df, :target,2,3)
            preds = apply_forest(model, df)
            cm = confusion_matrix(df[:target], preds)
            @test cm.accuracy > 0.7
        end

        @testset "Adaboost Stumps" begin
            model, coeffs = build_adaboost_stumps(df, :target, 7)
            preds = apply_adaboost_stumps(model, coeffs, df)
            cm = confusion_matrix(df[:target], preds)
            #@test cm.accuracy > 0.7
        end

    end

end