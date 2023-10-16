@testset "feature_importance_test.jl" begin
    X = [
        -3 2 2
        -2 -2 -3
        -2 0 2
        -1 -3 1
        -1 -1 0
        2 -3 1
        1 -2 0
        3 -2 3
        1 -1 2
        2 -1 -2
        2 0 1
        5 0 0
        1 0 2
        3 1 -1
        1 4 -3
        5 5 -2
        6 3 -1
        4 2 0
        4 3 2
        5 2 4
    ]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]

    # classifier
    model = build_tree(y, X)
    entropy1 = -(10 * log(10 / 20) + 10 * log(10 / 20)) / 20
    entropy2 = -(5 * log(5 / 15) + 10 * log(10 / 15)) / 20
    entropy3 = -(5 * log(5 / 7) + 2 * log(2 / 7)) / 20
    @test isapprox(
        impurity_importance(model), [entropy1 - entropy2, entropy2 - entropy3, entropy3]
    )
    @test split_importance(model) == [1, 1, 1]

    # prune_tree
    pt = prune_tree(model, 0.7)
    @test isapprox(impurity_importance(pt), [entropy1 - entropy2, entropy2 - entropy3, 0])
    @test split_importance(pt) == [1, 1, 0]

    pt = prune_tree(model, 0.6)
    @test isapprox(impurity_importance(pt), [entropy1 - entropy2, 0, 0])
    @test split_importance(pt) == [1, 0, 0]

    # regressor
    model = build_tree(float.(y), X, 0, -1, 2)
    mse1 = ((1^2 * 10 + 0^2 * 10) / 20 - ((1 * 10 + 0 * 10) / 20)^2)
    mse2 = ((1^2 * 10 + 0^2 * 5) / 15 - ((1 * 10 + 0 * 5) / 15)^2) * 15 / 20
    mse3 = ((1^2 * 2 + 0^2 * 5) / 7 - ((1 * 2 + 0 * 5) / 7)^2) * 7 / 20
    @test isapprox(impurity_importance(model), [mse1 - mse2, mse2 - mse3, mse3])
    @test split_importance(model) == [1, 1, 1]

    # prune_tree
    pt = prune_tree(model, 0.7)
    @test isapprox(impurity_importance(pt), [mse1 - mse2, mse2 - mse3, 0])
    @test split_importance(pt) == [1, 1, 0]

    pt = prune_tree(model, 0.6)
    @test isapprox(impurity_importance(pt), [mse1 - mse2, 0, 0])
    @test split_importance(pt) == [1, 0, 0]

    # Increase samples for testing permutation_importance and ensemble models
    X2 = repeat(X; inner=(50, 1)) .+ rand(StableRNG(1), 1000, 3)
    y2 = repeat(y; inner=50)

    # classifier
    model = build_tree(y2, X2; rng=StableRNG(1))
    p1 = permutation_importance(
        model,
        y2,
        X2,
        (model, y, X) -> accuracy(y, apply_tree(model, X)),
        10;
        rng=StableRNG(1),
    )
    @test similarity(
        impurity_importance(model), [entropy1 - entropy2, entropy2 - entropy3, entropy3]
    ) > 0.9
    @test similarity(split_importance(model), [1, 1, 1]) > 0.9
    @test argmax(p1.mean) == 1
    @test argmin(p1.mean) == 3

    model = build_forest(y2, X2, -1, 100; rng=StableRNG(1))
    i1 = impurity_importance(model)
    s1 = split_importance(model)
    p1 = permutation_importance(
        model,
        y2,
        X2,
        (model, y, X) -> accuracy(y, apply_forest(model, X)),
        10;
        rng=StableRNG(1),
    )
    model = build_forest(y2, X2, -1, 100; rng=StableRNG(100))
    i2 = impurity_importance(model)
    s2 = split_importance(model)
    p2 = permutation_importance(
        model,
        y2,
        X2,
        (model, y, X) -> accuracy(y, apply_forest(model, X)),
        10;
        rng=StableRNG(100),
    )
    @test argmin(p1.mean) == argmin(p2.mean)
    @test (-(sort(p1.mean; rev=true)[1:2]...) - -(sort(p2.mean; rev=true)[1:2]...)) < 0.2
    @test similarity(i1, i2) > 0.9
    @test similarity(s1, s2) > 0.9

    model, coeffs = build_adaboost_stumps(y2, X2, 20; rng=StableRNG(1))
    s1 = split_importance(model)
    p1 = permutation_importance(
        (model, coeffs),
        y2,
        X2,
        (model, y, X) -> accuracy(y, apply_adaboost_stumps(model, X)),
        10;
        rng=StableRNG(1),
    )
    model, coeffs = build_adaboost_stumps(y2, X2, 20; rng=StableRNG(100))
    s2 = split_importance(model)
    p2 = permutation_importance(
        (model, coeffs),
        y2,
        X2,
        (model, y, X) -> accuracy(y, apply_adaboost_stumps(model, X)),
        10;
        rng=StableRNG(100),
    )
    @test argmin(p1.mean) == argmin(p2.mean)
    @test (-(sort(p1.mean; rev=true)[1:2]...) - -(sort(p2.mean; rev=true)[1:2]...)) < 0.1
    @test similarity(s1, s2) > 0.9

    # regressor
    y2 = y2 .+ rand(StableRNG(1), 1000) ./ 100
    model = build_tree(y2, X2, 0, 3, 5, 2, 0.01; rng=StableRNG(1))
    p1 = permutation_importance(
        model, y2, X2, (model, y, X) -> R2(y, apply_tree(model, X)), 10; rng=StableRNG(1)
    )
    @test similarity(impurity_importance(model), [mse1 - mse2, mse2 - mse3, mse3]) > 0.9
    @test similarity(split_importance(model), [1, 1, 1]) > 0.9
    @test argmax(p1.mean) == 1
    @test argmin(p1.mean) == 3

    model = build_forest(y2, X2, 0, 100, 0.7, 3, 5, 2, 0.01; rng=StableRNG(1))
    i1 = impurity_importance(model)
    s1 = split_importance(model)
    p1 = permutation_importance(
        model, y2, X2, (model, y, X) -> R2(y, apply_forest(model, X)), 10; rng=StableRNG(1)
    )
    model = build_forest(y2, X2, 0, 100, 0.7, 3, 5, 2, 0.01; rng=StableRNG(100))
    i2 = impurity_importance(model)
    s2 = split_importance(model)
    p2 = permutation_importance(
        model,
        y2,
        X2,
        (model, y, X) -> R2(y, apply_forest(model, X)),
        10;
        rng=StableRNG(100),
    )
    @test argmax(p1.mean) == argmax(p2.mean)
    @test argmin(p1.mean) == argmin(p2.mean)
    @test similarity(i1, i2) > 0.9
    @test similarity(s1, s2) > 0.9

    # Common datasets
    X, y = load_data("digits")

    # classifier
    model = build_tree(y, X, 0, 3, 1, 2; rng=StableRNG(1))
    # sklearn equivalent: 
    # model = DecisionTreeClassifier(max_depth = 3, criterion = 'entropy', random_state = 1)
    # model.fit(X, y)
    @test isapprox(
        filter(x -> >(x, 0), impurity_importance(model; normalize=true)),
        [0.11896482, 0.15168659, 0.17920925, 0.29679316, 0.11104555, 0.14230064],
        atol=0.0000005,
    )

    # regressor
    model = build_tree(float.(y), X, 0, 3, 1, 2; rng=StableRNG(1))
    # sklearn equivalent: 
    # model = DecisionTreeRegressor(max_depth = 3, random_state = 1)
    # model.fit(X, y)
    @test isapprox(
        filter(x -> >(x, 0), impurity_importance(model; normalize=true)),
        [0.1983883, 0.02315617, 0.09821539, 0.06591425, 0.19884457, 0.14939765, 0.26608367],
        atol=0.0000005,
    )

    X, y = load_data("iris")
    y = String.(y)

    # classifier
    model = build_forest(y, X, -1, 100, 0.7, 2; rng=StableRNG(100))
    # sklearn:
    # model = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 100, criterion = 'entropy')
    # model.fit(X, y)
    f1 = impurity_importance(model)
    @test sum(f1[1:2]) < 0.25 # About 0.1% will fail among different rng
    @test abs(f1[4] - f1[3]) < 0.35 # About 1% will fail among different rng

    model, coeffs = build_adaboost_stumps(y, X, 10; rng=StableRNG(1))
    # sklearn:
    # model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1, criterion = 'entropy'),
    #                           algorithm = 'SAMME', n_estimators = 10, random_state = 1)
    # model.fit(X, y)
    f1 = impurity_importance(model, coeffs)
    @test sum(f1[1:2]) < 0.1            # Very Stable
    @test 0.35 < (f1[3] - f1[4]) < 0.45 # Very Stable
    # regressor
    y2 = repeat([1.0, 2.0, 3.0]; inner=50)
    model = build_forest(y2, X, 0, 100, 0.7, 2; rng=StableRNG(100))
    # sklearn:
    # model = RandomForestRegressor(n_estimators = 100, max_depth = 2, random_state = 100)
    # model.fit(X, y)
    f1 = impurity_importance(model)
    @test sum(f1[1:2]) < 0.1             # Very Stable

    X = X[:, 1:3] # leave only one important feature
    # classifier
    model = build_forest(y, X, -1, 100, 0.7, 2; rng=StableRNG(100))
    # sklearn:
    # model = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 100, criterion = 'entropy')
    # model.fit(X, y)
    f1 = impurity_importance(model)
    @test argmax(f1) == 3

    model, coeffs = build_adaboost_stumps(y, X, 10; rng=StableRNG(1))
    # sklearn:
    # model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 1, criterion = 'entropy'),
    #                           algorithm = 'SAMME', n_estimators = 10, random_state = 1)
    # model.fit(X, y)
    @test 0.85 < split_importance(model, coeffs)[3] < 0.95 # Very Stable

    # regressor
    model = build_forest(y2, X, 0, 100, 0.7, 2; rng=StableRNG(100))
    # sklearn:
    # model = RandomForestRegressor(n_estimators = 100, max_depth = 2, random_state = 100)
    # model.fit(X, y)
    f1 = impurity_importance(model)
    @test sum(f1[1:2]) < 0.1  # Very Stable
end
