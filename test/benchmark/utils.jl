function build_adaboost(labels, features)
    n_iterations = 10
    model, coeffs = build_adaboost_stumps(labels, features, n_iterations)
    return model
end
function apply_adaboost(model, features)
    n = length(model)
    return apply_adaboost_stumps(model, ones(n), features)
end

function pad(s::String, l::Int=21)
    t = length(s)
    p = max(0, l - t)
    return s * " " ^ p
end

function print_details(results)
    k = keys(results)
    for i in k
        s = "================ " * i * " ================"
        println("\n" * s)
        display(results[i])
#        if typeof(results[i]) <: BenchmarkGroup
#            print_details(results[i])
#        end
    end
end
