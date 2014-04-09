using Base.Test
using DecisionTree

tests = ["classification_rand.jl", "regression_rand.jl"]

println("Running tests...")
for curtest in tests
    println("Test: $curtest")
    include(curtest)
    println("=" ^ 50)
end
