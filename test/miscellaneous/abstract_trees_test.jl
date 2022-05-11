# Test `AbstractTrees`-interface

@testset "abstract_trees_test.jl" begin

# CAVEAT:   These tests rely heavily on the texts generated in `printnode`.
#           After changes in `printnode` the following `*pattern`s might be adapted.

### Some content-checking helpers
# if no feature names or class labels are given, the following keywords must be present
featid_pattern          = "Feature: "               # feature ids are prepended by this text
classid_pattern         = "Class: "                 # `Leaf.majority` is prepended by this text
# if feature names and class labels are given, they can be identified within the tree using these patterns
fname_pattern(fname)    = fname * " <"            # feature names are followed by " <"
clabel_pattern(clabel)  = "─ " * clabel * " ("      # class labels are embedded in "─ " and " ("

# occur all elements of `pool` in the form defined by `fname_/clabel_pattern` in `str_tree`?  
check_occurence(str_tree, pool, pattern) = count(map(elem -> occursin(pattern(elem), str_tree), pool)) == length(pool)

@info("Test base functionality")    
l1 = Leaf(1, [1,1,2])
l2 = Leaf(2, [1,2,2])
l3 = Leaf(3, [3,3,1])
n2 = Node(2, 0.5, l2, l3)
n1 = Node(1, 0.7, l1, n2)
feature_names = ["firstFt", "secondFt"]
class_labels = ["a", "b", "c"]

infotree1 = wrap(n1, (featurenames = feature_names, classlabels = class_labels))
infotree2 = wrap(n1, (featurenames = feature_names,))
infotree3 = wrap(n1, (classlabels = class_labels,))
infotree4 = wrap(n1, (x = feature_names, y = class_labels))
infotree5 = wrap(n1)

@info(" -- Tree with feature names and class labels")
AbstractTrees.print_tree(infotree1)
rep1 = AbstractTrees.repr_tree(infotree1)
@test check_occurence(rep1, feature_names, fname_pattern)
@test check_occurence(rep1, class_labels, clabel_pattern)

@info(" -- Tree with feature names")
AbstractTrees.print_tree(infotree2)
rep2 = AbstractTrees.repr_tree(infotree2)
@test check_occurence(rep2, feature_names, fname_pattern)
@test occursin(classid_pattern, rep2)

@info(" -- Tree with class labels")
AbstractTrees.print_tree(infotree3)
rep3 = AbstractTrees.repr_tree(infotree3)
@test occursin(featid_pattern, rep3)
@test check_occurence(rep3, class_labels, clabel_pattern)

@info(" -- Tree with ids only (nonsense parameters)")
AbstractTrees.print_tree(infotree4)
rep4 = AbstractTrees.repr_tree(infotree4)
@test occursin(featid_pattern, rep4)
@test occursin(classid_pattern, rep4)

@info(" -- Tree with ids only")
AbstractTrees.print_tree(infotree5)
rep5 = AbstractTrees.repr_tree(infotree5)
@test occursin(featid_pattern, rep5)
@test occursin(classid_pattern, rep5)
    
@info("Test `children` with 'adult' decision tree")
@info(" -- Preparing test data")
features, labels = load_data("adult")
feature_names_adult = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
model = build_tree(labels, features)
wrapped_tree = wrap(model, (featurenames = feature_names_adult,))

@info(" -- Test `children`")
function traverse_tree(node::InfoNode)
    l, r = AbstractTrees.children(node)
    @test l.info == node.info
    @test r.info == node.info
    traverse_tree(l)
    traverse_tree(r)
end

traverse_tree(leaf::InfoLeaf) = nothing

traverse_tree(wrapped_tree)
end