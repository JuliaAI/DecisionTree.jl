function build_adaboost(labels, features)
    n_iterations = 10
    model, coeffs = build_adaboost_stumps(labels, features, n_iterations)
    return model
end
function apply_adaboost(model, features)
    n = length(model)
    return apply_adaboost_stumps(model, ones(n), features)
end

function load_digits()
    f = open("data/digits.csv")
    data = readlines(f)[2:end]
    data = [[parse(Float32, i)
        for i in split(row, ",")]
        for row in data]
    data = hcat(data...)
    Y = Int.(data[1, 1:end]) .+ 1
    X = convert(Matrix, transpose(data[2:end, 1:end]))
    return X, Y
end

function load_adult()
    adult = DelimitedFiles.readdlm("data/adult.csv", ',');
    n = floor(Int, size(adult)[1] / 10)
    X = adult[1:n, 1:14];
    Y = adult[1:n, 15];
    return X, Y    
end

function load_iris()
    iris = DelimitedFiles.readdlm("data/iris.csv", ',')
    X = iris[:, 1:4]
    Y = iris[:, 5]
    return X, Y    
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
