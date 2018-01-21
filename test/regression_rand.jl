using Base.Test
using DecisionTree
using DataFrames

n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;

maxdepth = 3

@testset "nfoldCV - Regression" begin

    @testset "Arrays" begin

        @testset "nfoldCV Tree" begin
            model = build_tree(labels, features, 5, 0, maxdepth)
            @test depth(model) == maxdepth

            r2 = nfoldCV_tree(labels, features, 3)
            @test mean(r2) > 0.6
        end

        @testset "nfoldCV Forest" begin
            r2 = nfoldCV_forest(labels, features, 2, 10, 3)
            @test mean(r2) > 0.8
        end

    end

    @testset "DataFrame" begin

        df = hcat(DataFrame(features), DataFrame(target = labels))

        @testset "nfoldCV Tree" begin
            model = build_tree(df, :target, 5, 0, maxdepth)
            @test depth(model) == maxdepth

            r2 = nfoldCV_tree(df, :target, 3)
            @test mean(r2) > 0.6
        end

        @testset "nfoldCV Forest" begin
            r2 = nfoldCV_forest(df, :target, 2, 10, 3)
            @test mean(r2) > 0.8
        end

    end

end