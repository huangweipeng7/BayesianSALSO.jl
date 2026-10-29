# BayesianSALSO

[![Docs](https://img.shields.io/badge/docs-blue.svg)](https://huangweipeng7.github.io/BayesianSALSO.jl/)

> **Reference paper:** Dahl, D. B., Johnson, D. J., & Müller, P. (2022).
> *Search Algorithms and Loss Functions for Bayesian Clustering.* Journal of
> Computational and Graphical Statistics 31(4), 1189–1201.
> [doi:10.1080/10618600.2022.2069779](https://doi.org/10.1080/10618600.2022.2069779)

This package implements the method of that paper.  Given the cluster label of every
item in each of `B` posterior draws, `salso` returns the partition that minimises
the estimated posterior expected loss.  Two losses are available, each with a cost
parameter `a ∈ [0, 2]` as in R's `salso`: `BinderLoss(a)`, the N-invariant Binder
loss, and `VILoss(a)`, the (generalized) variation of information.
`expected_loss` reports both on the same scale as `salso::binder()` and
`salso::VI()`.

## Install

Once the General registration is merged:

```julia
using Pkg
Pkg.add("BayesianSALSO")
```

Until then, or when working directly from the repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/huangweipeng7/BayesianSALSO.jl.git")
```

Julia 1.10 or newer.

## Use

```julia
using BayesianSALSO

# draws[b, i] is the label of item i in draw b: integers, 0 = in no cluster
draws = [ 1 2 1 3 3 ;   # draw 1
          1 2 2 3 3 ;   # draw 2
          1 2 2 3 1 ]   # draw 3

res = salso(draws; loss = VILoss(1.0), n_runs = 16)

res.partition      # point estimate, labels 1:K
res.n_clusters     # K
res.expected_loss  # E[loss] of that partition
res.n_runs, res.n_scans, res.n_zealous_accepts, res.n_zealous_attempts
```

### Keywords

| keyword | default | |
|---|---|---|
| `loss` | `VILoss(1.0)` | or `BinderLoss(a)` |
| `n_runs` | 16 | independent searches; the best one wins |
| `max_zealous_attempts` | 10 | destroy-and-reallocate attempts per run |
| `max_n_clusters` | 0 | cluster ceiling; 0 uses the largest count seen in a draw, as R's `salso` does |
| `prob_sequential_allocation` | 0.5 | greedy initialiser versus a random start |
| `max_scans` | `typemax(Int)` | sweeps per run |
| `seconds` | `Inf` | soft budget for the whole call, checked between runs |
| `rng` | `default_rng()` | e.g. `MersenneTwister(42)` to repeat a search (identical for the same Julia version and CPU) |

`seconds` is the one option whose effect is not reproducible from `rng`: once the
budget is spent, no further run of the `n_runs` is started and the best one so far is
returned, so a slower or busier machine simply completes fewer runs.  The default
`Inf` keeps a search deterministic.

The VI search is single-threaded.  The Binder search, and the exported
`compute_psm`, use Julia's threads to fill the PSM by row, so start Julia with
`-t auto` or `-t N` for larger inputs.  The row split and exact integer counts keep
the PSM unchanged by the thread count.  The Binder path stores the full PSM, so its
memory use is `O(n²)`; very large `n` is better served by `VILoss`.

Labels must be integers of any width; floats, strings, boolean masks, a single draw
vector or a vector of draw vectors raise an `ArgumentError` saying what to pass
instead.  The search warns when the estimate lands on the cluster ceiling, where
the optimum may lie outside the region searched.

The pairwise metrics `ari`, `vi`, `nmi`, `purity`, `binder_loss`,
`binder_loss_n_invariant`, `compute_psm` and `expected_loss` are exported as well,
for scoring an estimate against a reference partition, plus `count_cluster` for the
number of clusters in a single draw.

## Command line

For a CSV holding one row per observation and one column per draw:

```bash
julia --project=. -t4 ./test/analyze_results.jl \
    --input_file=./data/test.csv \
    --output_file=./test_part.csv
```

It writes the point estimate as a one-column CSV (`salso_partition`) and prints
`K`, the expected loss and the elapsed time.  `--loss` selects `binder` (default)
or `vi`, `--loss_a` sets the cost parameter, and `--n_runs`, `--max_zealous`,
`--k_d` (ceiling, 0 = largest seen), `--seed`, `--header` (skip a header row) and
`--truth_file` (report the ARI) are also accepted.  Every cell must be an integer
label: a file holding probabilities, `1.0`-style floats or an unskipped header is
rejected with a message.

To run the test suite instead:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

MIT licensed; see [LICENSE](LICENSE).
