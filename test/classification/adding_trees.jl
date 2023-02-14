# determine order of a numerical list, eg, [0.3, 0.1, 0.6, -0.1] -> [3, 2, 4, 1]
function rank(v)
    w = sort(collect(enumerate(v)); by=last)
    return invperm(first.(w))
end

features, labels = load_data("iris")
features = float.(features)
labels = string.(labels)
classes = unique(labels)

@testset "adding models in an ensemble" begin
    n = 40   # n_trees in first step
    Δn = 30  # n_trees to be added

    one_step_model = build_forest(labels, features, 2, n + Δn; rng=srng())

    model1 = build_forest(labels, features, 2, n; rng=srng())
    two_step_model = build_forest(model1, labels, features, 2, Δn; rng=srng())

    @test length(two_step_model) == n + Δn

    # test the models agree on the initial portion of the ensemble:
    @test apply_forest_proba(one_step_model[1:n], features, classes) ≈
        apply_forest_proba(two_step_model[1:n], features, classes)

    # smoke test - predictions are from the classes seen:
    @test issubset(unique(apply_forest(two_step_model, features)), classes)

    # smoke test - one-step and two-step models predict the same feature rankings:
    @test rank(impurity_importance(one_step_model)) ==
        rank(impurity_importance(two_step_model))
end
