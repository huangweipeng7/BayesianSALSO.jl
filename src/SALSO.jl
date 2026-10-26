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
raise an `ArgumentError` rather than being coerced.  The implementation is split
by responsibility:

- `src/algorithms/search.jl` — shared greedy search and public API
- `src/algorithms/binder.jl` — Binder loss, incremental move score, and search driver
- `src/algorithms/vi.jl` — VI loss, incremental state, and search driver
- `src/support/psm.jl` — pairwise similarity matrix
- `src/support/canonicalization.jl` — cluster-label canonicalisation
- `src/api/interface.jl` — argument checking
- `src/api/metric.jl` — pairwise comparison metrics
"""
module SALSO

export SALSOLoss, BinderLoss, VILoss
export salso, SALSOResult
export compute_psm, expected_loss
export binder_loss, binder_loss_n_invariant, ari, vi, nmi, purity, count_cluster

using Clustering: randindex, varinfo
using Random: default_rng, shuffle!, randperm

include("support/canonicalization.jl")
include("support/psm.jl")
include("algorithms/search.jl")
include("algorithms/binder.jl")
include("algorithms/vi.jl")
include("api/interface.jl")
include("api/metric.jl")

end # module SALSO
