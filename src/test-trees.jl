include("test-header.jl")

using FileIO

trees_directory = "./results-audio-scan/trees-to-test"
output_directory = "./results-test-trees"

mkpath(trees_directory)
mkpath(output_directory)

tree_files = map(str -> trees_directory * "/" * str, filter!(endswith(".jld"), readdir(trees_directory)))

function load_tree(file_name::String)::Union{DecisionTree.DTree,DecisionTree.DTInternal}
    println("Loading $(file_name)...")
    load(string(file_name))["T"]
end

function load_all_trees(file_names::Vector{String})::Vector{DecisionTree.DTree,DecisionTree.DTInternal}
    arr = []
    for f in file_names
        push!(arr, load_tree(f))
    end
    arr
end

testing_tree = load_tree(tree_files[1])
print_tree(testing_tree)

seeds = [
    "7933197233428195239",
    "1735640862149943821",
    "3245434972983169324",
    "1708661255722239460",
    "1107158645723204584"
]

configs = [
    "30.45.30",
    "30.75.50"
]

for seed in seeds
    for config in configs
        dataset = nothing
        JLD2.@load "./results-audio-scan/datasets/$(seed).1.1.60.($(config)).jld" dataset

        regenerated_tree = update_tree_leaves(testing_tree, dataset)
        print_tree(regenerated_tree)
    end
end