@testset "graph_regression.jl" begin

X, Y, adj = load_data("graph_regression")
model = build_forest(Y, X, adj=adj)


function check_tree(tree, adj)
    tree_features = Vector{Int}()
    tree_max_depth = 0
    # get max depth of the tree and add all included features to a vector
    function walk_tree(node)
        push!(tree_features, node.featid)
        if node.depth > tree_max_depth
            tree_max_depth = node.depth
        end
        if isa(node.left, DecisionTree.Node)
            walk_tree(node.left)
        end
        if isa(node.right, DecisionTree.Node)
            walk_tree(node.right)
        end
    end
    walk_tree(tree)

    # all features which are within tree_max_depth of the feature the first
    # node is split on
    legal_features = findall(!iszero, (adj^tree_max_depth)[tree.featid,:])
    push!(legal_features, tree.featid)
    for i in tree_features
        if i ∉ legal_features
            println(i, "is not a legal feature")
            return false
        end
    end
    return true
end


function check_trees(model, adj)
    for tree in model.trees
        if !check_tree(tree, adj)
            return false
        end
    end
    return true
end


@test check_trees(model, adj)

end
