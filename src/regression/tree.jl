# The code in this file is a small port from scikit-learn's and numpy's
# library which is distributed under the 3-Clause BSD license.
# The rest of DecisionTree.jl is released under the MIT license.

# written by Poom Chiarawongse <eight1911@gmail.com>

module treeregressor
    include("../util.jl")

    export build_tree

    mutable struct NodeMeta
        l           :: NodeMeta  # right child
        r           :: NodeMeta  # left child
        labels      :: Array{Float64}
        label       :: Float64  # most likely label
        feature     :: Int64    # feature used for splitting
        threshold   :: Any      # threshold value
        is_leaf     :: Bool

        depth       :: Int64
        region      :: UnitRange{Int64} # a slice of the samples used to decide the split of the node
        features    :: Array{Int64} # a list of features not known to be constant

        split_at    :: Int64 # index of samples
        NodeMeta(features, region, depth) = (
            node = new();
            node.depth = depth;
            node.region = region;
            node.features = features;
            node.is_leaf = false;
            node)
    end

    # find an optimal split that satisfy the given constraints
    # (max_depth, min_samples_split, min_purity_increase)
    function _split!(X                   :: Matrix{T}, # the feature array
                     Y                   :: Array{Float64, 1}, # the label array
                     node                :: NodeMeta, # the node to split
                     max_features        :: Int64, # number of features to consider
                     max_depth           :: Int64, # the maximum depth of the resultant tree
                     min_samples_leaf    :: Int64, # the minimum number of samples each leaf needs to have
                     min_samples_split   :: Int64, # the minimum number of samples in needed for a split
                     min_purity_increase :: Float64, # minimum purity needed for a split
                     indX                :: Array{Int64, 1}, # an array of sample indices,
                                                             # we split using samples in indX[node.region]
                     # the two arrays below are given for optimization purposes
                     Xf                  :: Array{T},
                     Yf                  :: Array{Float64},
                     rng                 :: AbstractRNG) where T <: Any

        region = node.region
        n_samples = length(region)
        r_start = region.start - 1

        @simd for i in 1:n_samples
            @inbounds Yf[i] = Y[indX[i + r_start]]
        end

        tssq = 0.0
        tsum = 0.0
        @simd for i in 1:n_samples
            val = Yf[i]
            tssq += val*val
            tsum += val
        end

        node.label =  tsum / n_samples
        if (min_samples_leaf * 2 >  n_samples
         || min_samples_split    >  n_samples
         || max_depth            <= node.depth
         # equvalent to old_purity > -1e-7
         || tsum * node.label    > -1e-7 * n_samples + tssq)
            node.labels = Yf[1:n_samples]
            node.is_leaf = true
            return
        end

        features = node.features
        n_features = length(features)
        best_purity = -Inf
        best_feature = -1
        threshold_lo = Inf32
        threshold_hi = Inf32

        # -- start --
        # is this needed for initialization?
        # i have no idea, but we'll see
        rssq = tssq
        lssq = 0.0
        rsum = tsum
        lsum = 0.0
        # --  end  --

        indf = 1
        # the number of new constants found during this split
        n_constant = 0
        # true if every feature is constant
        unsplittable = true
        # the number of non constant features we will see if
        # only sample n_features used features
        # is a hypergeometric random variable
        total_features = size(X, 2)

        # this is the total number of features that we expect to not
        # be one of the known constant features. since we know exactly
        # what the non constant features are, we can sample at 'non_constants_used'
        # non constant features instead of going through every feature randomly.
        non_constants_used = util.hypergeometric(n_features, total_features-n_features, max_features, rng)
        @inbounds while (unsplittable || indf <= non_constants_used) && indf <= n_features
            feature = let
                indr = rand(rng, indf:n_features)
                features[indf], features[indr] = features[indr], features[indf]
                features[indf]
            end

            rssq = tssq
            lssq = 0.0
            rsum = tsum
            lsum = 0.0

            @simd for i in 1:n_samples
                sub_i = indX[i + r_start]
                Yf[i] = Y[sub_i]
                Xf[i] = X[sub_i, feature]
            end

            # sort Yf and Xf by Xf
            util.q_bi_sort!(Xf, Yf, 1, n_samples)
            nl, nr = 0, n_samples
            # lo and hi are the indices of
            # the least upper bound and the greatest lower bound
            # of the left and right nodes respectively
            lo, hi = 0, 0
            is_constant = true
            while hi < n_samples
                lo = hi + 1
                curr_f = Xf[lo]
                hi = (lo < n_samples && curr_f == Xf[lo+1]
                    ? searchsortedlast(Xf, curr_f, lo, n_samples, Base.Order.Forward)
                    : lo)

                (nl != 0) && (is_constant = false)
                # honor min_samples_leaf
                if nl >= min_samples_leaf && nr >= min_samples_leaf
                    unsplittable = false
                    purity = (rsum * rsum / nr) + (lsum * lsum / nl)
                    if purity > best_purity
                        # will take average at the end, if possible
                        threshold_lo = last_f
                        threshold_hi = curr_f
                        best_purity = purity
                        best_feature = feature
                    end
                end

                let delta = hi - lo + 1
                    nl += delta
                    nr -= delta
                end

                # update, lssq, rssq, lsum, rsum in the direction
                # that would require the smaller number of iterations
                if (hi << 1) < n_samples + lo # i.e., hi - lo < n_samples - hi
                    @simd for i in lo:hi
                        lab  = Yf[i]
                        lsum += lab
                        lssq += lab*lab
                    end
                    rsum = tsum - lsum
                    rssq = tssq - lssq
                else
                    rsum = rssq = 0.0
                    @simd for i in (hi+1):n_samples
                        lab  = Yf[i]
                        rsum += lab
                        rssq += lab*lab
                    end
                    lsum = tsum - rsum
                    lssq = tssq - rssq
                end

                last_f = curr_f
            end

            # keep track of constant features to be used later.
            if is_constant
                n_constant += 1
                features[indf], features[n_constant] = features[n_constant], features[indf]
            end

            indf += 1
        end

        # no splits honor min_samples_leaf
        @inbounds if unsplittable
            node.labels = Yf[1:n_samples]
            node.is_leaf = true
            return
        else
            # new_purity - old_purity < stop.min_purity_increase
            if (best_purity - tsum * node.label < min_purity_increase * n_samples)
                node.labels = Yf[1:n_samples]
                node.is_leaf = true
                return
            end
            bf = Int64(best_feature)
            @simd for i in 1:n_samples
                Xf[i] = X[indX[i + r_start], bf]
            end
            try
                node.threshold = (threshold_lo + threshold_hi) / 2.0
            catch
                node.threshold = threshold_hi
            end
            # split the samples into two parts: ones that are greater than
            # the threshold and ones that are less than or equal to the threshold
            #                                 ---------------------
            # (so we partition at threshold_lo instead of node.threshold)
            node.split_at = util.partition!(indX, Xf, threshold_lo, region)
            node.feature = best_feature
            node.features = features[(n_constant+1):n_features]
        end

    end

    @inline function fork!(node :: NodeMeta)
        ind = node.split_at
        region = node.region
        features = node.features
        # no need to copy because we will copy at the end
        node.l = NodeMeta(features, region[    1:ind], node.depth + 1)
        node.r = NodeMeta(features, region[ind+1:end], node.depth + 1)
    end


    # To do: check that Y actually has
    # meta.n_classes classes
    function check_input(X                   :: Matrix,
                         Y                   :: Array{Float64, 1},
                         max_features        :: Int64,
                         max_depth           :: Int64,
                         min_samples_leaf    :: Int64,
                         min_samples_split   :: Int64,
                         min_purity_increase :: Float64)
        n_samples, n_features = size(X)
        if length(Y) != n_samples
            throw("dimension mismatch between X and Y ($(size(X)) vs $(size(Y))")

        elseif n_features < max_features
            throw("number of features $(n_features) "
                * "is less than the number of "
                * "max features $(max_features)")

        elseif min_samples_leaf < 1
            throw("min_samples_leaf must be a positive integer "
                * "(given $(min_samples_leaf))")

        elseif min_samples_split < 2
            throw("min_samples_split must be at least 2 "
                * "(given $(min_samples_split))")
        end
    end

    function build_tree(X                   :: Matrix{T},
                        Y                   :: Vector,
                        max_features        :: Int64,
                        max_depth           :: Int64,
                        min_samples_leaf    :: Int64,
                        min_samples_split   :: Int64,
                        min_purity_increase :: Float64;
                        rng=Base.GLOBAL_RNG :: AbstractRNG) where T <: Any
        n_samples, n_features = size(X)
        check_input(
            X, Y,
            max_features,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase)
        indX = collect(Int64(1):Int64(n_samples))
        root = NodeMeta(collect(1:n_features), 1:n_samples, 0)
        stack = NodeMeta[ root ]

        Xf  = Array{T}(n_samples)
        Yf  = Array{Float64}(n_samples)
        @inbounds while length(stack) > 0
            node = pop!(stack)
            _split!(
                X,
                Y,
                node,
                max_features,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                indX,
                Xf, Yf,
                rng)
            if !node.is_leaf
                fork!(node)
                push!(stack, node.r)
                push!(stack, node.l)
            end
        end
        return root
    end
end
