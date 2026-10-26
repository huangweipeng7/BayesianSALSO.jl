# ─── Binder loss and search ───────────────────────────────────────────────────
#
# The Binder loss is evaluated from the pairwise similarity matrix.  The search
# keeps the label vector as its state and scores a move with one O(n) pass over
# the item's PSM row.

# ─── Loss and expected loss ───────────────────────────────────────────────────

"""
    BinderLoss(a = 1.0)

N-invariant Binder loss with cost `a ∈ [0, 2]`, as in R's `salso`: `a` is the cost
of separating two items the draws put together, `2 - a` the complementary mistake."""
struct BinderLoss <: SALSOLoss
    a::Float64
    two_minus_a::Float64
    function BinderLoss(a::Float64 = 1.0)
        @assert 0 ≤ a ≤ 2 "Binder a must be in [0, 2], got $a"
        new(Float64(a), 2.0 - Float64(a))
    end
end

"""
    expected_loss(labels, psm, ::BinderLoss) -> Float64

Expected N-invariant Binder loss of `labels` from the PSM, on the same scale as R's
`salso`.  Items labelled 0 are unassigned and skipped."""
function expected_loss(labels::Vector{Int}, psm::Matrix{Float64}, loss::BinderLoss)
    n = length(labels)
    size(psm) == (n, n) || throw(DimensionMismatch("psm is $(size(psm)), but labels has $n items"))
    a = loss.a
    two_minus_a = loss.two_minus_a
    eloss = 0.0

    @inbounds for i = 1:n
        li = labels[i]
        li == 0 && continue                     # unassigned
        # Mask-and-sum rather than branch: the row is contiguous, so this
        # vectorises.  `assigned` keeps unassigned items at zero contribution.
        row_sum = 0.0
        @simd for j = (i+1):n
            lj = labels[j]
            assigned = Float64(lj != 0)
            same = assigned * Float64(lj == li)
            # same pairs pay 2-a for a separated draw, different pairs pay a
            # for a joined draw
            row_sum +=
                same * (two_minus_a - two_minus_a * psm[i, j]) + (assigned - same) * a * psm[i, j]
        end
        eloss += row_sum
    end

    return 2.0 * eloss / (n * n)
end

function expected_loss(labels::Vector{Int}, draws::Matrix{Int}, loss::BinderLoss)
    psm = compute_psm(draws)
    return expected_loss(labels, psm, loss)
end

# ─── Incremental move score ───────────────────────────────────────────────────

"""
    _binder_move_deltas(labels, psm, i, from_k, hi, loss) -> Vector{Float64}

`out[k]` is the change in expected Binder loss from moving item `i` from cluster
`from_k` to each `k` in `1:hi`, from ONE pass over row `i`.  The per-cluster row
sums are shared by every candidate, so a sweetening scan costs O(n) per item
instead of O(n·K) (measured 4.3x at K = 6, 10x at K = 12)."""
function _binder_move_deltas(
    labels::Vector{Int},
    psm::Matrix{Float64},
    i::Int,
    from_k::Int,
    hi::Int,
    loss::BinderLoss,
)
    c = loss.two_minus_a
    n = length(labels)
    bound = max(hi, maximum(labels), 1)      # label-indexed scratch
    acc = zeros(Float64, bound)
    sizes = zeros(Int, bound)
    @inbounds for j = 1:n
        lj = labels[j]
        lj == 0 && continue
        sizes[lj] += 1
        acc[lj] += psm[i, j]
    end
    # item i leaves its own cluster: drop its diagonal contribution and its count
    from_side =
        from_k == 0 ? 0.0 : @inbounds 2.0 * (acc[from_k] - psm[i, i]) - c * (sizes[from_k] - 1)
    out = Vector{Float64}(undef, hi)
    inv = 2.0 / (n * n)
    @inbounds for k = 1:hi
        out[k] = k == from_k ? 0.0 : inv * (from_side + c * sizes[k] - 2.0 * acc[k])
    end
    return out
end

# ─── Initialisation ───────────────────────────────────────────────────────────

"""Initialize labels for the Binder search."""
function _binder_init(
    n::Int,
    K_max::Int,
    psm::Matrix{Float64},
    loss::BinderLoss,
    prob_seq_alloc::Float64,
    rng,
)
    if rand(rng) < prob_seq_alloc
        return _binder_sequential_allocation(n, K_max, psm, loss, rng)
    else
        labels = zeros(Int, n)
        for i = 1:n
            labels[i] = rand(rng, 1:K_max)
        end
        return labels
    end
end

"""
    _binder_best_placement(labels, i, psm, loss, K_max, active_labels, max_label)

Cluster for the not-yet-assigned item `i` in the sequential-allocation initialiser:
the candidate minimising the Binder delta relative to the unassigned state, with
`max_label + 1` (a fresh cluster) as the fallback while fewer than `K_max` clusters
are in use."""
function _binder_best_placement(
    labels::Vector{Int},
    i::Int,
    psm::Matrix{Float64},
    loss::BinderLoss,
    K_max::Int,
    active_labels,
    max_label::Int,
)
    best_k = 1
    best_delta = Inf

    deltas = _binder_move_deltas(labels, psm, i, 0, max_label, loss)
    @inbounds for k in active_labels
        d = deltas[k]
        if d < best_delta
            best_delta = d
            best_k = k
        end
    end

    if length(active_labels) < K_max && 0.0 < best_delta
        best_k = max_label + 1
    end

    return best_k
end

"""Sequential allocation for the Binder search."""
function _binder_sequential_allocation(
    n::Int,
    K_max::Int,
    psm::Matrix{Float64},
    loss::BinderLoss,
    rng,
)
    labels = zeros(Int, n)
    order = randperm(rng, n)

    labels[order[1]] = 1
    active_labels = Set([1])
    max_label = 1

    @inbounds for idx = 2:n
        i = order[idx]
        best_k = _binder_best_placement(labels, i, psm, loss, K_max, active_labels, max_label)
        labels[i] = best_k
        push!(active_labels, best_k)
        max_label = max(max_label, best_k)
    end

    return labels
end

# ─── Greedy search ────────────────────────────────────────────────────────────

"""Immutable inputs of a Binder search (the state itself is the label vector)."""
struct _BinderSearch
    psm::Matrix{Float64}
    loss::BinderLoss
    K_max::Int
end

"""Binder reallocation step: move item `i` to the cluster minimising its loss."""
@inline function _binder_sweeten_step(s::_BinderSearch, labels, i::Int)
    return _binder_move_item!(labels, i, s.psm, s.loss, s.K_max)
end

"""Try to move item `i` to a better cluster (sweetening scan)."""
function _binder_move_item!(labels, i, psm, loss::BinderLoss, K_max)
    from_k = @inbounds labels[i]
    # Candidates: labels 1:K_current (an empty one acts as a fresh cluster) and
    # K_current+1 while the ceiling allows.
    K_current = maximum(labels)
    hi = K_current < K_max ? K_current + 1 : K_current
    deltas = _binder_move_deltas(labels, psm, i, from_k, hi, loss)

    best_k = from_k
    best_delta = 0.0
    @inbounds for k = 1:hi
        if deltas[k] < best_delta
            best_delta = deltas[k]
            best_k = k
        end
    end
    if best_k != from_k
        @inbounds labels[i] = best_k
        return true
    end
    @inbounds labels[i] = from_k
    return false
end

"""
    _binder_best_move!(labels, i, psm, loss, K_max)

Zealous reallocation for item `i`, which arrives unassigned (`labels[i] == 0`):
one pass over row `i` scores every candidate cluster."""
function _binder_best_move!(
    labels::Vector{Int},
    i::Int,
    psm::Matrix{Float64},
    loss::BinderLoss,
    K_max::Int,
)
    best_k = 1
    best_delta = Inf
    K_current = 0

    deltas = _binder_move_deltas(labels, psm, i, 0, max(K_max, maximum(labels)), loss)
    @inbounds for k in unique(labels)
        k <= 0 && continue
        K_current += 1
        d = deltas[k]
        if d < best_delta
            best_delta = d
            best_k = k
        end
    end

    if K_current < K_max && 0.0 < best_delta
        # maximum(labels)+1 (not K_current+1): labels can be non-contiguous
        # here, and the new-cluster candidate must not collide with an
        # existing cluster label
        best_k = maximum(labels) + 1
    end

    @inbounds labels[i] = best_k
    return nothing
end

"""
    _salso_zealous_binder!(labels, search, max_zealous_attempts, max_scans, rng)
        -> (labels, n_accepts, n_attempts, n_scans)

Zealous updates for the Binder driver: the same rule as `_salso_zealous_vi!`, with
the O(n) row pass for the reallocation and `labels` reassigned, so it is returned."""
function _salso_zealous_binder!(
    labels::Vector{Int},
    search::_BinderSearch,
    max_zealous_attempts::Int,
    max_scans::Int,
    rng,
)
    n = length(labels)
    n_zaccepts = 0
    n_zattempts = 0
    n_zscans = 0
    for destroy_k in shuffle!(rng, collect(unique(labels)))
        n_zattempts >= max_zealous_attempts && break
        count(==(destroy_k), labels) <= 1 && continue
        n_zattempts += 1

        eloss_before = expected_loss(labels, search.psm, search.loss)
        old_labels = copy(labels)

        items_to_move = findall(==(destroy_k), labels)
        shuffle!(rng, items_to_move)
        @inbounds for i in items_to_move
            labels[i] = 0
        end

        for i in items_to_move
            _binder_best_move!(labels, i, search.psm, search.loss, search.K_max)
        end

        labels = _canonicalize(labels)

        # Sweeten inside the zealous update before accepting (Alg. 1).
        n_zscans += _salso_sweeten!(_binder_sweeten_step, search, labels, n, max_scans, rng)

        if expected_loss(labels, search.psm, search.loss) < eloss_before
            n_zaccepts += 1
        else
            labels = old_labels
        end
    end
    return labels, n_zaccepts, n_zattempts, n_zscans
end

"""Binder search driver: O(n) per-item moves on the precomputed PSM."""
function _salso_search(
    draws::Matrix{Int},
    loss::BinderLoss;
    n_runs::Int,
    max_zealous_attempts::Int,
    K_max::Int,
    prob_sequential_allocation::Float64,
    max_scans::Int,
    seconds::Float64,
    rng,
)
    n = size(draws, 2)
    psm = compute_psm(draws)
    search = _BinderSearch(psm, loss, K_max)

    best_labels = collect(1:n)
    best_eloss = Inf
    # Diagnostics of the BEST run, mirroring the salso reference implementation
    # (which returns the best candidate's counters, not a sum over all runs).
    best_scans = 0
    best_zaccepts = 0
    best_zattempts = 0
    actual_n_runs = 0

    t_start = time()
    for run = 1:n_runs
        if time() - t_start > seconds
            break
        end
        actual_n_runs += 1

        labels = _binder_init(n, K_max, psm, loss, prob_sequential_allocation, rng)

        n_scans = _salso_sweeten!(_binder_sweeten_step, search, labels, n, max_scans, rng)

        labels, n_zaccepts, n_zattempts, n_zscans =
            _salso_zealous_binder!(labels, search, max_zealous_attempts, max_scans, rng)

        eloss = expected_loss(labels, psm, loss)
        if eloss < best_eloss
            best_eloss = eloss
            best_labels = copy(labels)
            # Reference `nScans` counts sweetening scans for the whole run,
            # including those inside zealous updates.
            best_scans = n_scans + n_zscans
            best_zaccepts = n_zaccepts
            best_zattempts = n_zattempts
        end
    end

    return _salso_finish(
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
end
