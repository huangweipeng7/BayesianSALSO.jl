# BayesianSALSO

BayesianSALSO implements the search algorithms and loss functions from Dahl,
Johnson, and Müller (2022), *Search Algorithms and Loss Functions for Bayesian
Clustering*, Journal of Computational and Graphical Statistics 31(4), 1189–1201.
Given posterior draws of a partition, it returns the partition that minimises the
estimated posterior expected loss.

The package provides two loss families:

- [`BinderLoss`](@ref), the N-invariant Binder loss.
- [`VILoss`](@ref), the generalized variation-of-information loss.

Both use the cost parameter `a ∈ [0, 2]` from R's `salso`.

## Installation

The package is registered in the Julia General registry:

```julia
using Pkg
Pkg.add("BayesianSALSO")
```

To use the current development version directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/huangweipeng7/BayesianSALSO.jl.git")
```

Julia 1.10 or newer is required.

## Quick start

`draws` is a `B × n` integer matrix: row `b` contains the cluster labels for the
`n` items in posterior draw `b`. The value `0` means that an item is unassigned.

```julia
using BayesianSALSO

draws = [
    1 2 1 3 3
    1 2 2 3 3
    1 2 2 3 1
]

result = salso(draws; loss = VILoss(1.0), n_runs = 16)

result.partition        # labels 1:K
result.n_clusters       # K
result.expected_loss    # estimated posterior expected loss
```

The result also records the number of runs, sweeps, and accepted zealous moves;
see [`SALSOResult`](@ref).

## Choosing a loss

Use [`BinderLoss`](@ref) when pairwise co-clustering decisions are the target and
[`VILoss`](@ref) when a loss based on entropy and mutual information is preferred.
The parameter `a` trades off the two kinds of clustering error, matching the
parameterisation of R's `salso`.

```julia
binder_result = salso(draws; loss = BinderLoss(1.0), n_runs = 16)
vi_result = salso(draws; loss = VILoss(1.0), n_runs = 16)
```

[`expected_loss`](@ref) evaluates the estimated loss of a candidate partition.
[`compute_psm`](@ref) returns the posterior similarity matrix used by the Binder
path.

## Threading and memory

`VILoss` uses a single-threaded incremental search. `BinderLoss` uses the
posterior similarity matrix, whose rows are filled in parallel; start Julia with
`--threads=auto` or `--threads=N` for larger inputs.

The Binder path stores an `n × n` matrix, so its memory use is `O(n²)`. For very
large `n`, the VI path is usually the better choice.

## Comparing partitions

The package exports several comparison metrics:

```julia
estimate = result.partition
reference = [1, 2, 2, 3, 3]

ari(estimate, reference)
vi(estimate, reference)
nmi(estimate, reference)
purity(estimate, reference)
binder_loss(estimate, reference)
```

See the [API](@ref) page for the complete exported interface.
