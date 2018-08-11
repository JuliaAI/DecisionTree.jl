function load_data(name)
    datasets = ["iris", "adult", "digits"]
    data_path = joinpath(dirname(pathof(DecisionTree)), "..", "test/data/")

    if name == "digits"
        f = open(joinpath(data_path, "digits.csv"))
        data = readlines(f)[2:end]
        data = [[parse(Float32, i)
            for i in split(row, ",")]
            for row in data]
        data = hcat(data...)
        Y = Int.(data[1, 1:end]) .+ 1
        X = convert(Matrix, transpose(data[2:end, 1:end]))
        return X, Y
    end

    if name == "iris"
        iris = DelimitedFiles.readdlm(joinpath(data_path, "iris.csv"), ',')
        X = iris[:, 1:4]
        Y = iris[:, 5]
        return X, Y
    end

    if name == "adult"
        adult = DelimitedFiles.readdlm(joinpath(data_path, "adult.csv"), ',');
        X = adult[:, 1:14];
        Y = adult[:, 15];
        return X, Y
    end

    if !(name in datasets)
        throw("Available datasets are $(join(datasets,", "))")
    end
end
