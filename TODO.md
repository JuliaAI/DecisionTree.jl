* add more tests
* implement gradient boost and adaboost for regression
* test weights support for both regression and classification trees
* add min_weights_leaf prepruning
* add stricter typing for leaf and node api
    * develop "compact" leaf, which only holds counts of classes
* add new options for user input purity criterions
* optimize vectorization, e.g, changing `nc[:] .= 0` to loops
* add postpruning option with comparison with validation data
* use range [0,1] for purity_thresh in new implementations of `prune_tree` (currently commented out)
* standardize variable names to snake case
* trees should still split if purity change is _equal_ to min_purity_increase
* add benchmarks for other functions
