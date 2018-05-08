using Base.Test
using DecisionTree

tests = ["datatest.jl", "classification_rand.jl", "regression_rand.jl", "classification_hetero.jl",
         "misc.jl", "classification_scikitlearn.jl", "regression_scikitlearn.jl"]

println("Running tests...")
for curtest in tests
    println("Test: $curtest")
    include(curtest)
    println("=" ^ 50)
end
