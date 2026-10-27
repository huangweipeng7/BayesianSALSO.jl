# ─── Shared search and public API ─────────────────────────────────────────────
#
# Both loss drivers run the same greedy stages: sequential allocation, sweetening
# scans, and zealous reallocation.  This file holds the shared sweep, result
# assembly, keyword validation, and the public `salso` entry point.

# ─── Loss interface ───────────────────────────────────────────────────────────

abstract type SALSOLoss end

# ─── Result type ──────────────────────────────────────────────────────────────

"""
    SALSOResult

Result of `salso`: `partition`, `expected_loss`, `n_clusters`, `n_runs`, `n_scans`,
`n_zealous_accepts`, `n_zealous_attempts`."""
SALSOResult = NamedTuple{
    (
        :partition,
        :expected_loss,
        :n_clusters,
        :n_runs,
        :n_scans,
        :n_zealous_accepts,
        :n_zealous_attempts,
    ),
    Tuple{Vector{Int},Float64,Int,Int,Int,Int,Int},
}

Base.show(io::IO, r::SALSOResult) = print(
    io,
    "SALSOResult(K=$(r.n_clusters), " *
    "E[loss]=$(round(r.expected_loss, digits=4)), " *
    "runs=$(r.n_runs), scans=$(r.n_scans), " *
    "zealous=$(r.n_zealous_accepts)/$(r.n_zealous_attempts))",
)

# ─── Shared greedy stages ─────────────────────────────────────────────────────

"""
    _salso_sweeten!(step, state, labels, n, max_scans, rng) -> Int

Sweep the `n` items in random order, applying `step(state, labels, i)` to each,
until a sweep changes nothing or the scan budget runs out.  Returns the number of
sweeps; `step` reports whether it moved the item."""
function _salso_sweeten!(step::F, state, labels, n::Int, max_scans::Int, rng) where {F}
    n_scans = 0
    changed = true
    seen = Set{Vector{Int}}()
    while changed && n_scans < max_scans
        changed = false
        n_scans += 1
        for i in randperm(rng, n)
            changed |= step(state, labels, i)
        end
        # Moves are accepted only when they look improving, so a sweep that
        # reaches a partition seen before can only be cycling on rounding-level
        # deltas (e.g. -2.8e-17); stop rather than sweep forever.
        labels in seen && break
        push!(seen, copy(labels))
    end
    return n_scans
end

# ─── Result assembly ──────────────────────────────────────────────────────────

"""Canonicalise the winning labels and pack them into an `SALSOResult`."""
function _salso_result(best_labels, best_eloss, n_runs, n_scans, n_zaccepts, n_zattempts)
    best_canon = _canonicalize(best_labels)
    return (;
        partition = best_canon,
        expected_loss = best_eloss,
        n_clusters = maximum(best_canon),
        n_runs = n_runs,
        n_scans = n_scans,
        n_zealous_accepts = n_zaccepts,
        n_zealous_attempts = n_zattempts,
    )
end

"""Check that at least one run completed, package the best result, and warn at the cap."""
function _salso_finish(
    best_labels,
    best_eloss,
    actual_n_runs,
    n_runs,
    best_scans,
    best_zaccepts,
    best_zattempts,
    seconds,
    K_max,
)
    actual_n_runs == 0 && error(
        "no BayesianSALSO run completed (n_runs = $(n_runs), seconds = $(seconds)); " *
        "raise the run count or the time budget.",
    )
    res = _salso_result(
        best_labels,
        best_eloss,
        actual_n_runs,
        best_scans,
        best_zaccepts,
        best_zattempts,
    )
    if res.n_clusters ≥ K_max
        @warn "BayesianSALSO estimate hit the cluster cap (n_clusters = $(res.n_clusters), " *
              "K_max = $K_max). The optimum may lie outside the searched " *
              "region; raise max_n_clusters to check."
    end
    return res
end

# ─── Keyword validation ───────────────────────────────────────────────────────

"""
    _check_salso_kwargs!(; n_runs, max_zealous_attempts, max_n_clusters,
                           max_scans, seconds, prob_sequential_allocation)

Check `salso`'s keyword arguments up front, so an impossible value gives a clear
`ArgumentError` instead of a mid-search failure.
"""
function _check_salso_kwargs!(;
    n_runs::Integer,
    max_zealous_attempts::Integer,
    max_n_clusters::Integer,
    max_scans::Integer,
    seconds::Real,
    prob_sequential_allocation::Real,
)
    n_runs > 0 || throw(ArgumentError("n_runs must be > 0, got $n_runs"))
    max_zealous_attempts ≥ 0 ||
        throw(ArgumentError("max_zealous_attempts must be ≥ 0, got $max_zealous_attempts"))
    max_n_clusters ≥ 0 || throw(ArgumentError("max_n_clusters must be ≥ 0, got $max_n_clusters"))
    max_scans > 0 || throw(ArgumentError("max_scans must be > 0, got $max_scans"))
    seconds > 0 || throw(ArgumentError("seconds must be > 0, got $seconds"))
    0 ≤ prob_sequential_allocation ≤ 1 || throw(
        ArgumentError(
            "prob_sequential_allocation must be in [0, 1], got $prob_sequential_allocation",
        ),
    )
    return nothing
end

# ─── Public API ───────────────────────────────────────────────────────────────

"""
    salso(draws::Matrix{Int}; loss = VILoss(), n_runs = 16,
          max_zealous_attempts = 10, max_n_clusters = 0,
          prob_sequential_allocation = 0.5, max_scans = typemax(Int),
          seconds = Inf, rng = default_rng()) -> SALSOResult

SALSO point estimate from a `B × n` matrix of cluster labels: `n_runs` independent
greedy searches, of which the smallest expected `loss` wins.  `max_n_clusters = 0`
uses the largest number of labels seen in a draw, as R's `salso` does;
`prob_sequential_allocation` is the probability of the greedy initialiser versus a
random start; `max_scans` and `max_zealous_attempts` bound each run.  `seconds` is a
soft budget for the whole call, checked between runs: once it is spent, no further
run is started and the best one so far is returned, so a finite value makes the
result depend on the machine (the default `Inf` does not).  Each loss has its own
search driver.

The values are checked up front: `n_runs > 0`, `max_scans > 0`, `seconds > 0`,
`prob_sequential_allocation ∈ [0, 1]`, and non-negative counts."""
function salso(
    draws::Matrix{Int};
    loss::SALSOLoss = VILoss(),
    n_runs::Integer = 16,
    max_zealous_attempts::Integer = 10,
    max_n_clusters::Integer = 0,
    prob_sequential_allocation::Real = 0.5,
    max_scans::Integer = typemax(Int),
    seconds::Real = Inf,
    rng = default_rng(),
)
    B, n = size(draws)

    (B == 0 || n == 0) && throw(
        ArgumentError(
            "draws must be a B × n matrix with at least one draw and one item, got $(size(draws))",
        ),
    )

    _check_salso_kwargs!(;
        n_runs = n_runs,
        max_zealous_attempts = max_zealous_attempts,
        max_n_clusters = max_n_clusters,
        max_scans = max_scans,
        seconds = seconds,
        prob_sequential_allocation = prob_sequential_allocation,
    )
    # the drivers below are typed, and integers of any width (or e.g. `1 // 2`)
    # are accepted above
    n_runs = Int(n_runs)
    max_zealous_attempts = Int(max_zealous_attempts)
    max_n_clusters = Int(max_n_clusters)
    max_scans = Int(max_scans)
    seconds = Float64(seconds)
    prob_sequential_allocation = Float64(prob_sequential_allocation)

    if max_n_clusters ≤ 0
        # The largest number of labels seen in a draw, as in the R reference.
        # The scratch keeps this allocation-free: the buffer is reused for
        # every draw.
        sc = _CanonScratch()
        max_n_clusters = maximum(maximum(_canonicalize!(sc, @view draws[b, :])) for b = 1:B)
    end
    K_max = min(max_n_clusters, n)

    return _salso_search(
        draws,
        loss;
        n_runs = n_runs,
        max_zealous_attempts = max_zealous_attempts,
        K_max = K_max,
        prob_sequential_allocation = prob_sequential_allocation,
        max_scans = max_scans,
        seconds = seconds,
        rng = rng,
    )
end
