@testset "graph_regression.jl" begin

X, Y, adj = load_data("graph_regression")
model = build_forest(Y, X, adj=adj)


function check_tree(tree, adj)
    tree_features = Vector{Int}()
    function walk_tree(node)
        push!(tree_features, node.featid)
        if DecisionTree.is_leaf(node.left)
            walk_tree(node.left)
        end
        if DecisionTree.is_leaf(node.right)
            walk_tree(node.right)
        end
    end
    walk_tree(tree)
    tree_max_depth = depth(tree)

    # all features which are within tree_max_depth of the feature the first
    # node is split on
    legal_features = findall(!iszero, (adj^tree_max_depth)[tree.featid,:])
    push!(legal_features, tree.featid)
    for i in tree_features
        if i ∉ legal_features
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
