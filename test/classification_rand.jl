using Base.Test
using DecisionTree
using DataFrames

srand(16)

n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = _int(features * weights);

maxdepth = 3

@testset "nfoldCV - Classification" begin

    @testset "Arrays" begin

        @testset "nfoldCV Tree" begin
            model = build_tree(labels, features, 0, maxdepth)
            @test depth(model) == maxdepth

            accuracy = nfoldCV_tree(labels, features, 0.9, 3)
            @test mean(accuracy) > 0.7
        end

        @testset "nfoldCV Forest" begin
            accuracy = nfoldCV_forest(labels, features, 2, 10, 3)
            @test mean(accuracy) > 0.7
        end

        @testset "nfoldCV Adaboosted Stumps" begin
            accuracy = nfoldCV_stumps(labels, features, 7, 3)
            @test mean(accuracy) > 0.5
        end

    end

    @testset "DataFrame" begin

        df = hcat(DataFrame(features), DataFrame(target = labels))

        @testset "nfoldCV Tree" begin
            model = build_tree(df, :target, 0, maxdepth)
            @test depth(model) == maxdepth

            accuracy = nfoldCV_tree(df, :target, 0.9, 3)
            @test mean(accuracy) > 0.7
        end

        @testset "nfoldCV Forest" begin
            accuracy = nfoldCV_forest(df, :target, 2, 10, 3)
            @test mean(accuracy) > 0.7
        end

        @testset "nfoldCV Adaboosted Stumps" begin
            accuracy = nfoldCV_stumps(df, :target, 7, 3)
            #@test mean(accuracy) > 0.7
        end
        
    end

end