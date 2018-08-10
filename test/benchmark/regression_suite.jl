function benchmark_regression(build::Function, apply::Function)
    println("\nRunning benchmarks ...")
    ########## benchmarks suite ##########
    suite                       = BenchmarkGroup()
    suite["BUILD"]              = BenchmarkGroup()
    suite["BUILD"]["DIGITS"]    = BenchmarkGroup()
    suite["APPLY"]              = BenchmarkGroup()
    suite["APPLY"]["DIGITS"]    = BenchmarkGroup()
    
    # using DIGITS dataset
    X, Y = load_data("digits")

    m, n = size(X)
    X_Any = Array{Any}(undef, m, n)
    X_Any[:,:] = X
    X_Any           :: Matrix{Any}
    Y = Float64.(Y) :: Vector{Float64}
    model = build(Y, X_Any)
    preds = apply(model, X_Any)
    suite["BUILD"]["DIGITS"][pad("Y::Float64 X::Any")] = @benchmarkable $build($Y, $X_Any)
    suite["APPLY"]["DIGITS"][pad("Y::Float64 X::Any")] = @benchmarkable $apply($model, $X_Any)

    X = Int64.(X)   :: Matrix{Int64}
    Y = Float64.(Y) :: Vector{Float64}
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Float64 X::Int64")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Float64 X::Int64")] = @benchmarkable $apply($model, $X)

    X = Int8.(X)   :: Matrix{Int8}
    Y = Float16.(Y) :: Vector{Float16}
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Float16 X::Int8")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Float16 X::Int8")] = @benchmarkable $apply($model, $X)

    X = Float64.(X) :: Matrix{Float64}
    Y = Float64.(Y) :: Vector{Float64}
    model = build(Y, X)
    preds = apply(model, X)
    suite["BUILD"]["DIGITS"][pad("Y::Float64 X::Float64")] = @benchmarkable $build($Y, $X)
    suite["APPLY"]["DIGITS"][pad("Y::Float64 X::Float64")] = @benchmarkable $apply($model, $X)


    ########## run suite ##########
    tune!(suite)
    results = run(suite, verbose = true)
    return results
end
