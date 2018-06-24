using DecisionTree
using ScikitLearnBase

if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

function run_tests(list)
    for test in list
        println("TEST: $test \n")
        include(test)
        println("=" ^ 50)
    end
end

classification = "classification/" .* readdir("classification/")
regression     = "regression/"     .* readdir("regression/")
miscellaneous  = "miscellaneous/"  .* readdir("miscellaneous/")

test_suites = [("Classification", classification), ("Regression", regression), ("Miscellaneous", miscellaneous)]

@testset "Test Suites" begin
    for ts in 1:length(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        @testset "$name" begin
            run_tests(list)
        end
    end
end

# remove downloaded .csv files
for f in filter(x -> ismatch(r"\.csv", x), readdir())
    rm(f)
end
