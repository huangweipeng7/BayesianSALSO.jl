"""
    SALSO

Optimal point estimates (and comparison metrics) for Bayesian clustering
problems, following Dahl, Johnson & Müller (2022), *Search Algorithms and Loss
Functions for Bayesian Clustering*, JCGS 31(4), 1189–1201.

Given `B` draws of a partition — a `B × n` matrix of integer labels, one row per
draw — `salso` returns the partition minimising the estimated posterior expected
loss, under [`BinderLoss(a)`](@ref BinderLoss) or [`VILoss(a)`](@ref VILoss) with
`a ∈ [0, 2]`, as in R's `salso`:

```julia
using SALSO

# `draws[b, i]` is the label of item i in draw b
draws = [ 1 2 1 3 3 ;   # draw 1
          1 2 2 3 3 ;   # draw 2
          1 2 2 3 1 ]   # draw 3

res = salso(draws; loss = VILoss(1.0), n_runs = 4)
res.partition        # point estimate, labels 1:K
res.expected_loss    # E[loss], on the R `salso` scale
```

Label values are arbitrary integers (`0` means unassigned); non-integer labels
raise an `ArgumentError` rather than being coerced.  `src/salso.jl` holds the
algorithm, `src/interface.jl` the argument checking, `src/metric.jl` the pairwise
comparison metrics.
"""
module SALSO

export SALSOLoss, BinderLoss, VILoss
export salso, SALSOResult
export compute_psm, expected_loss
export binder_loss, binder_loss_n_invariant, ari, vi, nmi, purity, count_cluster

using Clustering: randindex, varinfo

include("salso.jl")
include("interface.jl")
include("metric.jl")

end # module SALSO
