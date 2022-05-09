"""
Implementation of the `AbstractTrees.jl`-interface 
(see: [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)).

The functions `children` and `printnode` make up the interface traits of `AbstractTrees.jl` 
(see below for details).

The goal of this implementation is to wrap a `DecisionTree` in this abstract layer, 
so that a plot recipe for visualization of the tree can be created that doesn't rely 
on any implementation details of `DecisionTree`. The plot recipe is part of `MLJ.jl` 
where all tree-like models may be visualized using this approach.

For a more detailed explanation of this concept have a look at the follwing article 
in "Towards Data Science": 
["If things are not ready to use"](https://towardsdatascience.com/part-iii-if-things-are-not-ready-to-use-59d2db378bec)
"""


"""
    InfoNode{S, T}
    InfoLeaf{T}

These types are introduced so that additional information currently not present in 
a `DecisionTree`-structure -- namely the feature names and the class labels -- 
can be used for visualization. This additional information is stored in the attribute `info` of 
these types. It is a `NamedTuple`. So it can be used to store arbitraty information, 
apart from the two points mentioned.

In analogy to the type definitions of `DecisionTree`, the generic type `S` is 
the type of the feature values used within a node as a threshold for the splits
between its children and `T` is the type of the classes given (these might be ids or labels).
"""
struct InfoNode{S, T}
    node    :: DecisionTree.Node{S, T}
    info    :: NamedTuple
end

struct InfoLeaf{T}
	leaf    :: DecisionTree.Leaf{T}
    info    :: NamedTuple
end

"""
    wrap(node, info = NamedTuple())
    wrap(leaf, info = NamedTuple())

Add to each `node` (or `leaf`) the additional information `info` 
and wrap both in an `InfoNode`/`InfoLeaf`.

To use a DecisionTree `dc` with the abstraction layer 
provided by the `AbstractTrees`-interface implemented here
and optionally add feature names `feature_names` and/or `class_labels` 
(both: arrays of strings) use the following syntax:

1.  `wdc = wrap(dc)` 
2.  `wdc = wrap(dc, (featurenames = feature_names, classlabels = class_labels))`
3.  `wdc = wrap(dc, (featurenames = feature_names, ))`
4.  `wdc = wrap(dc, (classlabels  = class_labels, ))`

In the first case `dc` gets just wrapped, no information is added. No. 2 adds feature names 
as well as class labels. In the last two cases either of this information is added (Note the 
trailing comma; it's needed to make it a tuple).
"""
wrap(node::DecisionTree.Node, info::NamedTuple = NamedTuple()) = InfoNode(node, info)
wrap(leaf::DecisionTree.Leaf, info::NamedTuple = NamedTuple()) = InfoLeaf(leaf, info)

"""
    children(node::InfoNode)

Return for each `node` given, its children. 
    
In case of a `DecisionTree` there are always exactly two children, because 
the model produces binary trees where all nodes have exactly one left and 
one right child. `children` is used for tree traversal.

The additional information `info` is carried over from `node` to its children.
"""
AbstractTrees.children(node::InfoNode) = (
    wrap(node.node.left, node.info),
    wrap(node.node.right, node.info)
)
AbstractTrees.children(node::InfoLeaf) = ()

"""
    printnode(io::IO, node::InfoNode)
    printnode(io::IO, leaf::InfoLeaf)

Write a printable representation of `node` or `leaf` to output-stream `io`.

If `node.info`/`leaf.info` have a field called
- `featurenames` it is expected to have an array of feature names corresponding 
  to the feature ids used in the `DecsionTree`s nodes. 
  They will be used for printing instead of the ids.
- `classlabels` it is expected to have an array of class labels corresponding 
  to the class ids used in the `DecisionTree`s leaves. 
  They will be used for printing instead of the ids.
  (Note: DecisionTrees created using MLJ use ids in their leaves; 
  otherwise class labels are present)
"""
function AbstractTrees.printnode(io::IO, node::InfoNode)
    if :featurenames ∈ keys(node.info) 
        print(io, node.info.featurenames[node.node.featid], " --> ", node.node.featval)
    else
	    print(io, "Feature: ", node.node.featid, " --> ", node.node.featval)
    end
end

function AbstractTrees.printnode(io::IO, leaf::InfoLeaf)
    dt_leaf = leaf.leaf
    matches     = findall(dt_leaf.values .== dt_leaf.majority)
	match_count = length(matches)
	val_count   = length(dt_leaf.values)
    if :classlabels ∈ keys(leaf.info)
        print(io, leaf.info.classlabels[dt_leaf.majority], " ($match_count/$val_count)")
    else
	    print(io, "Class: ", dt_leaf.majority, " ($match_count/$val_count)")
    end
end