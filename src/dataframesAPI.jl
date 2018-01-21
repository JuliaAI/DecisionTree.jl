# helper function to check DataFrame input
function _check_dataframe_input(data::DataFrame, target_column::Union{Symbol, AbstractString})
    target = isa(target_column, Symbol) ? target_column : Symbol(target_column)
    colnames = names(data)
    if target in colnames
        features = setdiff(colnames, [target])
        return data[target], data[features]
    else
        error("Specified `target_column` $(target_column) not present in data.")
    end
end

function build_tree(data::DataFrame, target_column::Union{Symbol, AbstractString}, args...; rng=Base.GLOBAL_RNG)
    build_tree(_check_dataframe_input(data, target_column)..., args..., rng=rng)
end

function build_forest(data::DataFrame, target_column::Union{Symbol, AbstractString}, args...; rng=Base.GLOBAL_RNG)
    build_forest(_check_dataframe_input(data, target_column)..., args...; rng=rng)
end

function build_adaboost_stumps(data::DataFrame, target_column::Union{Symbol, AbstractString}, args...; rng=Base.GLOBAL_RNG)
    build_adaboost_stumps(_check_dataframe_input(data, target_column)..., args...; rng=rng)
end