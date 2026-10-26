# SALSO (Dahl, Johnson & Müller 2022, JCGS 31(4), 1189-1201): given draws of a
# clustering, find the partition minimising the posterior expected loss.  The
# losses are BinderLoss(a) and VILoss(a) with a ∈ [0, 2]; every run is a greedy
# search: sequential-allocation initialisation, sweetening scans, and zealous
# updates (destroy a cluster, reallocate).

using Random: AbstractRNG, default_rng, shuffle!, randperm
using LinearAlgebra: Symmetric
using Statistics: mean

# ─── Part 1: Pairwise Similarity Matrix (PSM) ───

"""Scale raw pair counts to fractions and force a unit diagonal, in place."""
function _psm_normalize!(psm::Matrix{Float64}, n::Int, B::Int)
    inv_B = 1.0 / B
    @inbounds for i = 1:n
        for j = 1:i
            psm[i, j] *= inv_B
            psm[j, i] = psm[i, j]
        end
        psm[i, i] = 1.0
    end
    return psm
end

# The parallel accumulation splits the ROWS of one shared matrix between the
# workers, so the matrix is allocated once and there is no merge pass (see
# `compute_psm`).
#
# Smallest amount of bookkeeping (draws × items) that is worth handing to
# threads; below it the hand-off costs more than the work itself.
const _PSM_PARALLEL_MIN_WORK = 10_000

# Doubles per 64-byte cache line: the row blocks of the parallel accumulation
# are cut on these boundaries, so that no two workers share a cache line.
const _PSM_CACHE_LINE = 8

"""
Add the co-occurrence counts of draws `b1:b2` to the lower triangle of `psm`.

Items are bucketed by label, so a draw with K_b clusters costs Σ|bucket|²/2 updates
instead of n²/2.  The counts are exact integers, so the result does not depend on
the order of accumulation."""
function _psm_accumulate!(psm::Matrix{Float64}, C_mat::Matrix{Int}, b1::Int, b2::Int, off::Int, K_b::Int)
    n      = size(C_mat, 2)
    counts = zeros(Int, K_b + 1)
    starts = Vector{Int}(undef, K_b + 2)
    cursor = Vector{Int}(undef, K_b + 1)
    order  = Vector{Int}(undef, n)

    @inbounds for b = b1:b2
        _psm_accumulate_draw!(psm, @view(C_mat[b, :]), off, K_b, counts, starts, cursor, order)
    end

    return psm
end

"""
Bucket a draw's labels: `counts[l]` items carry bucket `l`, `starts[l]..starts[l+1]-1`
is that bucket's slice of `order`, and `order` lists the items grouped by label,
increasing within a group.  The scratch vectors are caller-owned, so a call
allocates nothing."""
function _psm_bucket!(
    counts::Vector{Int},
    starts::Vector{Int},
    cursor::Vector{Int},
    order::Vector{Int},
    C::AbstractVector{Int},
    off::Int,
    K_b::Int,
)
    n = length(C)
    @inbounds begin
        fill!(counts, 0)
        for i = 1:n
            counts[C[i]+off] += 1
        end

        s = 1
        for l = 1:K_b
            starts[l] = s
            s += counts[l]
        end
        starts[K_b+1] = s

        # `order` ends up grouped by label, increasing within a group, so
        # order[p] < order[q] for p < q keeps every update in the lower triangle.
        for l = 1:K_b
            cursor[l] = starts[l]
        end
        for i = 1:n
            l = C[i] + off
            order[cursor[l]] = i
            cursor[l] += 1
        end
    end
    return nothing
end

"""
Add ONE draw's co-occurrence counts to the lower triangle of `psm`.  `C` holds
the draw's labels (bucket index `l + off`); the scratch vectors are caller-owned,
so a call allocates nothing."""
function _psm_accumulate_draw!(
    psm::Matrix{Float64},
    C::AbstractVector{Int},
    off::Int,
    K_b::Int,
    counts::Vector{Int},
    starts::Vector{Int},
    cursor::Vector{Int},
    order::Vector{Int},
)
    @inbounds begin
        _psm_bucket!(counts, starts, cursor, order, C, off, K_b)
        for l = 1:K_b
            gl = starts[l]
            gh = starts[l+1] - 1
            for q = (gl+1):gh
                iq = order[q]
                for p = gl:(q-1)
                    psm[iq, order[p]] += 1.0
                end
            end
        end
    end
    return psm
end

"""
Add ONE draw's co-occurrence counts to the rows `r0:r1` of `psm`.

The lower triangle's element `(i, j)` is written only by the worker that owns
row `i` (a row holds the larger of the two items), so workers with disjoint row
blocks never touch the same element."""
function _psm_accumulate_draw_rows!(
    psm::Matrix{Float64},
    C::AbstractVector{Int},
    off::Int,
    K_b::Int,
    counts::Vector{Int},
    starts::Vector{Int},
    cursor::Vector{Int},
    order::Vector{Int},
    r0::Int,
    r1::Int,
)
    @inbounds begin
        _psm_bucket!(counts, starts, cursor, order, C, off, K_b)
        for l = 1:K_b
            gl = starts[l]
            gh = starts[l+1] - 1
            for q = (gl+1):gh
                iq = order[q]
                (iq < r0 || iq > r1) && continue
                for p = gl:(q-1)
                    psm[iq, order[p]] += 1.0
                end
            end
        end
    end
    return psm
end

"""
Add the co-occurrence counts of all draws to the rows `r0:r1` of `psm`.

Every draw is bucketed again here (O(n) per draw), because the buckets are what
turn the O(n²) pair count into Σ|bucket|²/2 updates; the redundant work runs in
parallel and so does not add to the wall time."""
function _psm_accumulate_rows!(
    psm::Matrix{Float64},
    C_mat::Matrix{Int},
    off::Int,
    K_b::Int,
    r0::Int,
    r1::Int,
)
    n = size(C_mat, 2)
    counts = zeros(Int, K_b + 1)
    starts = Vector{Int}(undef, K_b + 2)
    cursor = Vector{Int}(undef, K_b + 1)
    order  = Vector{Int}(undef, n)

    @inbounds for b = 1:size(C_mat, 1)
        _psm_accumulate_draw_rows!(psm, @view(C_mat[b, :]), off, K_b, counts, starts, cursor, order, r0, r1)
    end

    return psm
end

"""
Row blocks `r0:r1` of `n` rows for `nworkers` workers.

The blocks are cut on cache-line boundaries (8 doubles): in a column-major matrix
the rows of a block are 8 bytes apart, so two neighbouring blocks would
otherwise share the cache lines at their boundary and ping-pong them."""
function _psm_row_blocks(n::Int, nworkers::Int)
    blocks = Vector{Tuple{Int,Int}}(undef, nworkers)
    nlines = cld(n, _PSM_CACHE_LINE)          # lines per column
    per = cld(nlines, nworkers)
    @inbounds for w = 1:nworkers
        l0 = (w - 1) * per
        l1 = min(w * per, nlines)
        blocks[w] = (l0 * _PSM_CACHE_LINE + 1, min(l1 * _PSM_CACHE_LINE, n))
    end
    return blocks
end

"""
    compute_psm(C_mat::Matrix{Int}) -> Matrix{Float64}

Pairwise similarity matrix from `B` draws of `n` labels: entry `(i, j)` is the
fraction of draws in which items `i` and `j` share a label, the diagonal is 1.0.

The counts are accumulated in parallel by splitting the ROWS of one shared
matrix between the workers.  Every element of the lower triangle belongs to
exactly one row, so the workers never write the same element and no merge pass
is needed — unlike splitting the draws, which cost one n × n buffer per worker
plus a serial merge, and made the parallel path up to 12x slower than the
serial one."""
function compute_psm(C_mat::Matrix{Int})
    B, n = size(C_mat)
    B == 0 && return zeros(n, n)

    lo, hi = extrema(C_mat)          # labels may be arbitrary Ints (incl. 0)
    off = 1 - lo                     # bucket index of label `l` is `l + off`
    K_b = Int(hi) + off

    nworkers = min(Threads.nthreads(), cld(n, _PSM_CACHE_LINE))
    if nworkers > 1 && B * n >= _PSM_PARALLEL_MIN_WORK
        psm = zeros(n, n)
        blocks = _psm_row_blocks(n, nworkers)
        Threads.@threads for w = 1:nworkers
            r0, r1 = blocks[w]
            _psm_accumulate_rows!(psm, C_mat, off, K_b, r0, r1)
        end
    else
        psm = zeros(n, n)
        _psm_accumulate!(psm, C_mat, 1, B, off, K_b)
    end

    return _psm_normalize!(psm, n, B)
end

# ─── Part 2: Loss Functions ───

abstract type SALSOLoss end

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
    VILoss(a = 1.0)

(Generalized) variation-of-information loss with `a ∈ [0, 2]`, as in R's `salso`;
`VILoss()` is the ordinary expected VI, in bits."""
struct VILoss <: SALSOLoss
    a::Float64
    two_minus_a::Float64
    function VILoss(a::Float64 = 1.0)
        @assert 0 ≤ a ≤ 2 "VI a must be in [0, 2], got $a"
        new(Float64(a), 2.0 - Float64(a))
    end
end

# ─── Part 3: Expected Loss Computation ───

"""
    expected_loss(labels, psm, ::BinderLoss) -> Float64

Expected N-invariant Binder loss of `labels` from the PSM, on the same scale as R's
`salso`.  Items labelled 0 are unassigned and skipped."""
function expected_loss(labels::Vector{Int}, psm::Matrix{Float64}, loss::BinderLoss)
    n = length(labels)
    size(psm) == (n, n) ||
        throw(DimensionMismatch("psm is $(size(psm)), but labels has $n items"))
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
            row_sum += same * (two_minus_a - two_minus_a * psm[i, j]) +
                       (assigned - same) * a * psm[i, j]
        end
        eloss += row_sum
    end

    return 2.0 * eloss / (n * n)
end

"""
    expected_loss(labels, draws, ::VILoss) -> Float64

Expected (generalized) variation-of-information loss from `B × n` draws, in bits and
normalised by `n`, on the same scale as R's `salso::VI()`.  Items labelled 0 are
unassigned: each becomes its own singleton row and contributes nothing."""
function expected_loss(labels::Vector{Int}, draws::Matrix{Int}, loss::VILoss)
    n = length(labels)
    B = size(draws, 1)
    size(draws, 2) == n ||
        throw(DimensionMismatch("draws has $(size(draws, 2)) items, but labels has $n"))

    canon = _canonicalize(labels)
    K = maximum(canon)

    state_counts = zeros(Int, K)
    @inbounds for i = 1:n
        state_counts[canon[i]] += 1
    end

    a = loss.a
    two_minus_a = loss.two_minus_a

    # Candidate-side term: (2 - a)·Σ_c n_c log2 n_c
    est = 0.0
    @inbounds for s = 1:K
        ns = state_counts[s]
        ns > 0 && (est += two_minus_a * ns * log2(ns))
    end

    # Draw-side terms: a·Σ_l n_l log2 n_l − 2·Σ_{c,l} m log2 m, averaged over draws.
    #
    # The draws are folded in one at a time: a draw's confusion table is built
    # in a scratch buffer, added to the running sum and dropped again.  The
    # previous version kept a `K × K_b × B` table and a copy of every
    # canonicalised draw, i.e. O(B·K·K_b + B·n) memory; the scratch is
    # O(K·K_b + n) and the draws are never copied.  The terms are added in the
    # same order as before (per draw: the marginals, then the joint counts),
    # so the result is unchanged bit for bit.
    sc = _CanonScratch()
    Kb_cap = 0
    counts = Int[]                 # K × Kb_cap, row s at (s-1)·Kb_cap + l
    marg = Int[]                   # Kb_cap
    draw_side = 0.0
    @inbounds for b = 1:B
        _canonicalize!(sc, @view draws[b, :])
        ids = sc.out
        Kb = maximum(ids)          # distinct labels in this draw
        if Kb > Kb_cap
            Kb_cap = max(Kb, 2 * Kb_cap, 8)
            resize!(counts, K * Kb_cap)
            resize!(marg, Kb_cap)
        end
        fill!(counts, 0)
        fill!(marg, 0)
        for i = 1:n
            s = canon[i]
            l = ids[i]
            counts[(s-1)*Kb_cap + l] += 1
            marg[l] += 1
        end
        for l = 1:Kb
            nl = marg[l]
            nl > 0 && (draw_side += a * nl * log2(nl))
        end
        for s = 1:K
            base = (s-1) * Kb_cap
            for l = 1:Kb
                m = counts[base + l]
                m > 0 && (draw_side -= 2 * m * log2(m))
            end
        end
    end

    return (est + draw_side / B) / n
end

function expected_loss(labels::Vector{Int}, draws::Matrix{Int}, loss::BinderLoss)
    psm = compute_psm(draws)
    return expected_loss(labels, psm, loss)
end

# ─── Part 4: Efficient Incremental Delta for Binder Loss ───

"""
    _binder_delta(labels, psm, i, from_k, to_k, loss) -> Float64

Change in expected Binder loss from moving item `i` from cluster `from_k` to `to_k`,
on the `expected_loss` scale, in one pass over row `i` of the PSM.  `from_k = 0`
means `i` is unassigned; there the value is defined up to a constant that does not
depend on `to_k`, which is all the search uses it for."""
@inline @fastmath function _binder_delta(
    labels::Vector{Int},
    psm::Matrix{Float64},
    i::Int,
    from_k::Int,
    to_k::Int,
    loss::BinderLoss,
)
    a = loss.a
    two_minus_a = loss.two_minus_a
    delta = 0.0

    if from_k == to_k
        return 0.0
    end

    # Match only the `from` and `to` labels while scanning the row.  A ternary is
    # lazy, so most items (neither label) skip the PSM load entirely; `ifelse`
    # evaluates both sides eagerly and was measured ~5x slower here.
    @inbounds begin
        if from_k == 0
            # i is unassigned: only the `to` side contributes (`from_k == 0`
            # would otherwise match every other unassigned item, i.e. label 0).
            @simd for j = 1:length(labels)
                term = labels[j] == to_k ? two_minus_a - 2.0 * psm[i, j] : 0.0
                delta += term
            end
        else
            @simd for j = 1:(i-1)
                lj = labels[j]
                term =
                    lj == from_k ? 2.0 * psm[i, j] - two_minus_a :
                    (lj == to_k ? two_minus_a - 2.0 * psm[i, j] : 0.0)
                delta += term
            end
            @simd for j = (i+1):length(labels)
                lj = labels[j]
                term =
                    lj == from_k ? 2.0 * psm[i, j] - two_minus_a :
                    (lj == to_k ? two_minus_a - 2.0 * psm[i, j] : 0.0)
                delta += term
            end
        end
    end

    # Same 2/n² normalization as expected_loss.
    n = length(labels)
    return 2.0 * delta / (n * n)
end

"""
    _binder_move_deltas(labels, psm, i, from_k, hi, loss) -> Vector{Float64}

`out[k]` is `_binder_delta(labels, psm, i, from_k, k, loss)` for all `k` in `1:hi`,
from ONE pass over row `i`.  The per-cluster row sums are shared by every candidate,
so a sweetening scan costs O(n) per item instead of O(n·K) (measured 4.3x at K = 6,
10x at K = 12).  The sums accumulate in a different order than `_binder_delta`, so
values agree to rounding rather than bit for bit."""
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
        from_k == 0 ? 0.0 :
        @inbounds 2.0 * (acc[from_k] - psm[i, i]) - c * (sizes[from_k] - 1)
    out = Vector{Float64}(undef, hi)
    inv = 2.0 / (n * n)
    @inbounds for k = 1:hi
        out[k] = k == from_k ? 0.0 : inv * (from_side + c * sizes[k] - 2.0 * acc[k])
    end
    return out
end

# ─── Part 5: Incremental expected-VI state ───
#
# `_VIState` keeps the cluster sizes and the confusion table incrementally, so a
# candidate move costs O(B) instead of rebuilding the K × K_b × B table.  The move
# rule is the reference one (Dahl et al. 2021, Alg. 1).

"""Canonicalize labels to 1..K in order of first appearance, PRESERVING label 0."""
function _canonicalize_keep0(labels::AbstractVector{Int})
    mapping = Dict{Int,Int}()
    out = zeros(Int, length(labels))
    nxt = 0
    @inbounds for (i, lab) in enumerate(labels)
        lab == 0 && continue
        m = get(mapping, lab, 0)
        if m == 0
            nxt += 1
            mapping[lab] = nxt
            m = nxt
        end
        out[i] = m
    end
    return out
end

"""Incremental confusion table for the expected VI loss."""
mutable struct _VIState
    a::Float64             # cost parameter
    B::Int                 # number of draws
    n::Int                 # number of items
    Kb::Int                # number of distinct labels in the draws
    K_max::Int             # cluster ceiling
    D::Matrix{Int}         # D[b, i] = canonical label of item i in draw b
    kern::Vector{Float64}  # kern[j] = (j-1)·log2(j-1)   (j is 1-based ⇒ 0 at j=1)
    delta::Vector{Float64} # delta[j] = kern[j+1] - kern[j]
    conf::Array{Int32,3}   # conf[t, l, b] = #items in cluster t with draw label l
    n_of::Vector{Int32}    # n_of[t] = size of candidate cluster t (t = 0 ⇒ unassigned)
    labels::Vector{Int}    # current candidate partition (0 = unassigned)
    draw_const::Float64    # Σ_b Σ_l n_l log2 n_l  — fixed for the given draws
    scratch_occ::Vector{Int}    # reusable buffer for the occupied-cluster list
    scratch_vals::Vector{Float64} # reusable buffer for surrogate values
end

function _VIState(D::Matrix{Int}, a::Float64, K_max::Int)
    B, n = size(D)
    Kb = maximum(D)
    kmax_count = max(n, Kb) + 1
    kern = zeros(Float64, kmax_count)
    @inbounds for x = 2:(kmax_count-1)
        kern[x+1] = x * log2(x)
    end
    delta = zeros(Float64, kmax_count)
    @inbounds for j = 1:(kmax_count-1)
        delta[j] = kern[j+1] - kern[j]
    end

    draw_const = 0.0
    @inbounds for b = 1:B
        counts = zeros(Int, Kb)
        for i = 1:n
            counts[D[b, i]] += 1
        end
        for l = 1:Kb
            draw_const += kern[counts[l]+1]
        end
    end

    # NOTE: transposing D was tried and measured slower (the companion
    # conf[t, l, b] strides by 1 in b anyway); do not re-add it.
    return _VIState(
        a,
        B,
        n,
        Kb,
        K_max,
        D,
        kern,
        delta,
        zeros(Int32, K_max + 1, Kb, B),
        zeros(Int32, K_max + 1),
        zeros(Int, n),
        draw_const,
        Vector{Int}(undef, K_max),
        Vector{Float64}(undef, K_max),
    )
end

"""
    _vi_is_canonical(s) -> Bool

Whether `s.labels` already uses exactly `1:K` with every item assigned, i.e.
whether `_vi_reset!` would rebuild only the table `conf`/`n_of` already hold."""
function _vi_is_canonical(s::_VIState)
    assigned = 0
    used = 0
    @inbounds for t = 1:s.K_max
        n_t = s.n_of[t]
        n_t > 0 && (used += 1)
        assigned += n_t
    end
    return assigned == s.n && used == maximum(s.labels)
end

"""Install a candidate partition, rebuilding the table from scratch (O(n·B))."""
function _vi_reset!(s::_VIState, labels::AbstractVector{Int})
    s.labels = _canonicalize_keep0(labels)
    fill!(s.n_of, 0)
    fill!(s.conf, 0)
    @inbounds for i = 1:s.n
        t = s.labels[i]
        t == 0 && continue
        s.n_of[t] += 1
        for b = 1:s.B
            s.conf[t, s.D[b, i], b] += 1
        end
    end
    return s
end

"""Move item i to cluster t (t = 0 marks it unassigned).  O(B)."""
@inline function _vi_assign!(s::_VIState, i::Int, t::Int)
    f = @inbounds s.labels[i]
    f == t && return
    if f != 0
        @inbounds s.n_of[f] -= 1
        @inbounds for b = 1:s.B
            s.conf[f, s.D[b, i], b] -= 1
        end
    end
    @inbounds s.labels[i] = t
    if t != 0
        @inbounds s.n_of[t] += 1
        @inbounds for b = 1:s.B
            s.conf[t, s.D[b, i], b] += 1
        end
    end
    return
end

"""One-step change in the expected VI loss for placing item i in each of `cand`."""
function _vi_surrogate(s::_VIState, i::Int, cand::AbstractVector{Int})
    c = s.labels[i]
    vals = Vector{Float64}(undef, length(cand))
    @inbounds for (idx, t) in enumerate(cand)
        off = (t == c) ? 1 : 0
        acc = (2.0 - s.a) * s.B * s.delta[Int(s.n_of[t])-off+1]
        for b = 1:s.B
            acc -= 2.0 * s.delta[Int(s.conf[t, s.D[b, i], b])-off+1]
        end
        vals[idx] = acc
    end
    return vals
end

"""Expected VI loss of the current state, on the same scale as `expected_loss`."""
function _vi_full_loss(s::_VIState)
    term_c = 0.0
    @inbounds for t = 1:s.K_max
        term_c += s.kern[Int(s.n_of[t])+1]
    end
    term_c *= (2.0 - s.a)

    joint = 0.0
    @inbounds for b = 1:s.B, l = 1:s.Kb
        row = 0.0
        @simd for t = 1:s.K_max
            row += s.kern[Int(s.conf[t, l, b])+1]
        end
        joint += row
    end

    return (term_c + (s.a * s.draw_const - 2.0 * joint) / s.B) / s.n
end

function _vi_occupied(s::_VIState)
    out = Int[]
    @inbounds for t = 1:s.K_max
        s.n_of[t] > 0 && push!(out, t)
    end
    return out
end

"""Smallest unused cluster label, or `nothing` when the ceiling is reached."""
@inline function _vi_empty_label(s::_VIState)
    @inbounds for t = 1:s.K_max
        s.n_of[t] == 0 && return t
    end
    return nothing
end

"""Best cluster for item `i` under the reference decision rule: `argmin` over the
occupied clusters in ascending label order, falling back to a fresh empty cluster
when the best surrogate is still positive.  Allocation-free: the candidate list and
the surrogate values live in the state.
"""
function _vi_best_target(s::_VIState, i::Int)
    c = @inbounds s.labels[i]

    # occupied clusters, in ascending label order
    occ = s.scratch_occ
    m = 0
    @inbounds for t = 1:s.K_max
        if s.n_of[t] > 0
            m += 1
            occ[m] = t
        end
    end
    m == 0 && return 1

    # surrogate for each occupied target, into scratch_vals
    vals = s.scratch_vals
    @inbounds for idx = 1:m
        t = occ[idx]
        off = (t == c) ? 1 : 0
        acc = (2.0 - s.a) * s.B * s.delta[Int(s.n_of[t])-off+1]
        for b = 1:s.B
            acc -= 2.0 * s.delta[Int(s.conf[t, s.D[b, i], b])-off+1]
        end
        vals[idx] = acc
    end

    j = 1
    bv = @inbounds vals[1]
    @inbounds for idx = 2:m
        if vals[idx] < bv
            bv = vals[idx]
            j = idx
        end
    end
    best = @inbounds occ[j]
    if bv > 0.0
        e = _vi_empty_label(s)
        e !== nothing && (best = e)
    end
    return best
end

"""Sequential allocation from empty (reference initialisation)."""
function _vi_sequential_allocation!(s::_VIState, rng)
    _vi_reset!(s, zeros(Int, s.n))
    order = randperm(rng, s.n)
    for i in order
        _vi_assign!(s, i, _vi_best_target(s, i))
    end
    return s.labels
end

# ─── Part 6: SALSO Greedy Search ───

# Both drivers run the same three stages (Dahl et al. 2021, Algorithm 1) and
# differ only in the state they carry, so the shared loops live here.

"""Immutable inputs of a Binder search (the state itself is the label vector)."""
struct _BinderSearch
    psm::Matrix{Float64}
    draws::Matrix{Int}
    loss::BinderLoss
    K_max::Int
end

"""Binder reallocation step: move item `i` to the cluster minimising `_eloss`."""
@inline function _binder_sweeten_step(s::_BinderSearch, labels, i::Int)
    return _salso_move_item!(labels, i, s.psm, s.draws, s.loss, s.K_max)
end

"""VI reallocation step: move item `i` to its best target cluster."""
@inline function _vi_sweeten_step(s::_VIState, labels, i::Int)
    t = _vi_best_target(s, i)
    @inbounds t == labels[i] && return false
    _vi_assign!(s, i, t)
    return true
end

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

"""
    _salso_zealous_vi!(state, max_zealous_attempts, max_scans, rng)
        -> (n_accepts, n_attempts, n_scans)

Zealous updates: destroy a cluster, reallocate its items, re-sweeten, and keep the
result only if the expected loss improved."""
function _salso_zealous_vi!(s::_VIState, max_zealous_attempts::Int, max_scans::Int, rng)
    n_zaccepts = 0
    n_zattempts = 0
    n_zscans = 0
    for destroy_k in shuffle!(rng, collect(unique(s.labels)))
        n_zattempts >= max_zealous_attempts && break
        count(==(destroy_k), s.labels) <= 1 && continue
        n_zattempts += 1

        eloss_before = _vi_full_loss(s)
        old_labels = copy(s.labels)

        items_to_move = findall(==(destroy_k), s.labels)
        shuffle!(rng, items_to_move)
        for i in items_to_move
            _vi_assign!(s, i, 0)
        end
        for i in items_to_move
            _vi_assign!(s, i, _vi_best_target(s, i))
        end
        # `conf`/`n_of` are already exact; the rebuild is only for label gaps.
        _vi_is_canonical(s) || _vi_reset!(s, s.labels)

        n_zscans += _salso_sweeten!(_vi_sweeten_step, s, s.labels, s.n, max_scans, rng)

        _vi_full_loss(s) < eloss_before ?
            (n_zaccepts += 1) : _vi_reset!(s, old_labels)   # undo
    end
    return n_zaccepts, n_zattempts, n_zscans
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

        eloss_before = _eloss(labels, search.psm, search.draws, search.loss)
        old_labels = copy(labels)

        items_to_move = findall(labels .== destroy_k)
        shuffle!(rng, items_to_move)
        @inbounds for i in items_to_move
            labels[i] = 0
        end

        for i in items_to_move
            _salso_best_move!(labels, i, search.psm, search.draws, search.loss, search.K_max, rng)
        end

        labels = _canonicalize(labels)

        # Sweeten inside the zealous update before accepting (Alg. 1).
        n_zscans += _salso_sweeten!(_binder_sweeten_step, search, labels, n, max_scans, rng)

        if _eloss(labels, search.psm, search.draws, search.loss) < eloss_before
            n_zaccepts += 1
        else
            labels = old_labels
        end
    end
    return labels, n_zaccepts, n_zattempts, n_zscans
end

"""Canonicalise the winning labels and pack them into an `SALSOResult`."""
function _salso_result(best_labels, best_eloss, n_runs, n_scans, n_zaccepts, n_zattempts)
    best_canon = _canonicalize(best_labels)
    return (;
        partition          = best_canon,
        expected_loss      = best_eloss,
        n_clusters         = maximum(best_canon),
        n_runs             = n_runs,
        n_scans            = n_scans,
        n_zealous_accepts  = n_zaccepts,
        n_zealous_attempts = n_zattempts,
    )
end

"""VI search driver: the incremental confusion table makes one candidate move cost
O(B), against O(K·n·K_b·B) for recomputing `expected_loss(labels, draws, ::VILoss)`.
"""
function _salso_search(
    draws::Matrix{Int},
    loss::VILoss;
    n_runs::Int,
    max_zealous_attempts::Int,
    max_n_clusters::Int,   # accepted for signature parity; the cap warning uses K_max
    K_max::Int,
    prob_sequential_allocation::Float64,
    max_scans::Int,
    seconds::Float64,
    rng,
)
    B, n = size(draws)

    D = Matrix{Int}(undef, B, n)
    sc = _CanonScratch()
    @inbounds for b = 1:B
        D[b, :] = _canonicalize!(sc, @view draws[b, :])
    end
    state = _VIState(D, loss.a, K_max)

    best_labels = collect(1:n)
    best_eloss = Inf
    best_scans = 0
    best_zaccepts = 0
    best_zattempts = 0
    actual_n_runs = 0

    t_start = time()
    for run = 1:n_runs
        time() - t_start > seconds && break
        actual_n_runs += 1

        if rand(rng) < prob_sequential_allocation
            _vi_sequential_allocation!(state, rng)
        else
            _vi_reset!(state, [rand(rng, 1:K_max) for _ = 1:n])
        end

        # Sweetening scans
        n_scans = _salso_sweeten!(_vi_sweeten_step, state, state.labels, n, max_scans, rng)

        # Zealous updates
        n_zaccepts, n_zattempts, n_zscans = _salso_zealous_vi!(state, max_zealous_attempts, max_scans, rng)

        eloss = _vi_full_loss(state)
        if eloss < best_eloss
            best_eloss = eloss
            best_labels = copy(state.labels)
            best_scans = n_scans + n_zscans
            best_zaccepts = n_zaccepts
            best_zattempts = n_zattempts
        end
    end

    actual_n_runs == 0 && error(
        "no SALSO run completed (n_runs = $(n_runs), seconds = $(seconds)); " *
        "raise the run count or the time budget.",
    )
    res = _salso_result(best_labels, best_eloss, actual_n_runs, best_scans, best_zaccepts, best_zattempts)
    if res.n_clusters ≥ K_max
        @warn "SALSO estimate hit the cluster cap (n_clusters = $(res.n_clusters), " *
              "K_max = $K_max). The optimum may lie outside the searched " *
              "region; raise max_n_clusters to check."
    end
    return res
end

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
    max_n_clusters ≥ 0 ||
        throw(ArgumentError("max_n_clusters must be ≥ 0, got $max_n_clusters"))
    max_scans > 0 || throw(ArgumentError("max_scans must be > 0, got $max_scans"))
    seconds > 0 || throw(ArgumentError("seconds must be > 0, got $seconds"))
    0 ≤ prob_sequential_allocation ≤ 1 ||
        throw(ArgumentError(
            "prob_sequential_allocation must be in [0, 1], got $prob_sequential_allocation",
        ))
    return nothing
end

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
        n_runs = n_runs, max_zealous_attempts = max_zealous_attempts,
        max_n_clusters = max_n_clusters, max_scans = max_scans, seconds = seconds,
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
        max_n_clusters = max_n_clusters,
        K_max = K_max,
        prob_sequential_allocation = prob_sequential_allocation,
        max_scans = max_scans,
        seconds = seconds,
        rng = rng,
    )
end

"""Binder search driver: O(n) `_binder_delta` moves on the precomputed PSM."""
function _salso_search(
    draws::Matrix{Int},
    loss::BinderLoss;
    n_runs::Int,
    max_zealous_attempts::Int,
    max_n_clusters::Int,
    K_max::Int,
    prob_sequential_allocation::Float64,
    max_scans::Int,
    seconds::Float64,
    rng,
)
    n = size(draws, 2)
    psm = compute_psm(draws)
    search = _BinderSearch(psm, draws, loss, K_max)

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

        labels = _salso_init(n, K_max, psm, draws, loss, prob_sequential_allocation, rng)

        n_scans = _salso_sweeten!(_binder_sweeten_step, search, labels, n, max_scans, rng)

        labels, n_zaccepts, n_zattempts, n_zscans =
            _salso_zealous_binder!(labels, search, max_zealous_attempts, max_scans, rng)

        eloss = _eloss(labels, psm, draws, loss)
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

    actual_n_runs == 0 && error(
        "no SALSO run completed (n_runs = $(n_runs), seconds = $(seconds)); " *
        "raise the run count or the time budget.",
    )
    res = _salso_result(best_labels, best_eloss, actual_n_runs, best_scans, best_zaccepts, best_zattempts)
    if max_n_clusters > 0 && res.n_clusters ≥ min(max_n_clusters, n)
        @warn "SALSO estimate hit the cluster cap (n_clusters = $(res.n_clusters), " *
              "max_n_clusters = $max_n_clusters). The optimum may lie outside " *
              "the searched region; raise max_n_clusters to check."
    end
    return res
end

# ─── Result type ───

"""
    SALSOResult

Result of `salso`: `partition`, `expected_loss`, `n_clusters`, `n_runs`, `n_scans`,
`n_zealous_accepts`, `n_zealous_attempts`."""
SALSOResult = NamedTuple{
    (:partition, :expected_loss, :n_clusters, :n_runs, :n_scans, :n_zealous_accepts, :n_zealous_attempts),
    Tuple{Vector{Int},Float64,Int,Int,Int,Int,Int},
}

Base.show(io::IO, r::SALSOResult) = print(
    io,
    "SALSOResult(K=$(r.n_clusters), " *
    "E[loss]=$(round(r.expected_loss, digits=4)), " *
    "runs=$(r.n_runs), scans=$(r.n_scans), " *
    "zealous=$(r.n_zealous_accepts)/$(r.n_zealous_attempts))",
)

# ─── Internal helpers ───

# ─── Canonicalisation scratch ─────────────────────────────────────────────────

# `_canonicalize` runs once per draw in the PSM and VI paths, so it must not
# allocate: the scratch holds the label → id map, the output buffer and a
# generation counter, which lets the map be reused without clearing it.
#
# The map is indexed by `label - lo + 1`, where `lo` is the smallest label of
# the call.  The previous version indexed it by the raw label, so a negative
# label read out of bounds (`@inbounds` turned that into silent corruption:
# wrong canonicalisations, and a segfault once the garbage was written back).

const _CANON_DENSE_MAX = 100_000

mutable struct _CanonScratch
    ids::Vector{Int}      # label − lo + 1 → canonical id
    stamps::Vector{Int}   # generation in which `ids` is valid
    gen::Int              # current generation
    dict::Dict{Int,Int}   # fallback for widely spread labels
    out::Vector{Int}      # output buffer
end

_CanonScratch() = _CanonScratch(Int[], Int[], 0, Dict{Int,Int}(), Int[])

"""
    _canonicalize!(sc, labels) -> Vector{Int}

Canonicalize `labels` to 1..K in order of first appearance, writing into the
scratch's buffer.  The buffer is reused between calls, so a caller that keeps
the result must copy it.  Label 0 means "unassigned" and gets a cluster of its
own, exactly as in `_canonicalize`."""
function _canonicalize!(sc::_CanonScratch, labels::AbstractVector{Int})
    n = length(labels)
    length(sc.out) == n || resize!(sc.out, n)
    lo, hi = extrema(labels)
    span = Int128(hi) - Int128(lo) + 1     # Int128: the label range may not fit in Int
    sc.gen += 1
    gen = sc.gen
    nxt = 1
    if span <= _CANON_DENSE_MAX
        # Dense labels: a vector lookup table, reused across calls.  The
        # stamps say which generation an entry belongs to, so the table never
        # has to be cleared.
        if length(sc.ids) < span
            old = length(sc.stamps)
            resize!(sc.ids, span)
            resize!(sc.stamps, span)
            # `resize!` leaves the new space uninitialised, so the stamps of the
            # fresh entries have to be cleared: a stale stamp would make the
            # lookup below believe an id is still current.
            fill!(view(sc.stamps, old+1:span), 0)
        end
        @inbounds for i = 1:n
            lab = labels[i]
            if lab == 0
                # Unassigned label from zealous destruction — assign to a new cluster
                sc.out[i] = nxt
                nxt += 1
                continue
            end
            k = Int(lab - lo) + 1
            if sc.stamps[k] == gen
                sc.out[i] = sc.ids[k]
            else
                sc.stamps[k] = gen
                sc.ids[k] = nxt
                sc.out[i] = nxt
                nxt += 1
            end
        end
    else
        # Sparse labels: a hash map, cleared but not reallocated.
        empty!(sc.dict)
        @inbounds for i = 1:n
            lab = labels[i]
            if lab == 0
                sc.out[i] = nxt
                nxt += 1
                continue
            end
            m = get(sc.dict, lab, 0)
            if m == 0
                m = nxt
                nxt += 1
                sc.dict[lab] = m
            end
            sc.out[i] = m
        end
    end
    return sc.out
end

"""Canonicalize cluster labels to contiguous 1..K."""
function _canonicalize(labels::AbstractVector{Int})
    sc = _CanonScratch()
    return copy(_canonicalize!(sc, labels))
end

"""Compute expected loss for current labels."""
function _eloss(labels::Vector{Int}, psm::Union{Matrix{Float64},Nothing}, draws::Matrix{Int}, loss::SALSOLoss)
    return loss isa BinderLoss ? expected_loss(labels, psm, loss) : expected_loss(labels, draws, loss)
end

"""Initialize labels for SALSO."""
function _salso_init(
    n::Int,
    K_max::Int,
    psm::Union{Matrix{Float64},Nothing},
    draws::Matrix{Int},
    loss::SALSOLoss,
    prob_seq_alloc::Float64,
    rng,
)
    if rand(rng) < prob_seq_alloc
        return _sequential_allocation(n, K_max, psm, draws, loss, rng)
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
the candidate minimising `_binder_delta` relative to the unassigned state, with
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

function _sequential_allocation(n::Int, K_max::Int, psm::Matrix{Float64}, draws::Matrix{Int}, loss::BinderLoss, rng)
    labels = zeros(Int, n)
    order = randperm(rng, n)

    labels[order[1]] = 1
    active_labels = Set([1])
    max_label = 1

    for idx = 2:n
        i = order[idx]
        best_k = _binder_best_placement(labels, i, psm, loss, K_max, active_labels, max_label)
        labels[i] = best_k
        push!(active_labels, best_k)
        max_label = max(max_label, best_k)
    end

    return labels
end

"""Sequential allocation: place items one at a time into the best cluster."""
function _sequential_allocation(
    n::Int,
    K_max::Int,
    psm::Union{Matrix{Float64},Nothing},
    draws::Matrix{Int},
    loss::SALSOLoss,
    rng,
)

    labels = zeros(Int, n)
    order = randperm(rng, n)

    labels[order[1]] = 1
    active_labels = Set([1])  # track which cluster labels are in use
    max_label = 1

    @inbounds for idx = 2:n
        i = order[idx]
        best_k = 1
        best_eloss = Inf

        for k in active_labels
            labels[i] = k
            eloss = _eloss(labels, psm, draws, loss)
            if eloss < best_eloss
                best_eloss = eloss
                best_k = k
            end
        end

        if length(active_labels) < K_max
            labels[i] = max_label + 1
            eloss = _eloss(labels, psm, draws, loss)
            if eloss < best_eloss
                best_eloss = eloss
                best_k = max_label + 1
            end
        end

        labels[i] = best_k
        push!(active_labels, best_k)
        max_label = max(max_label, best_k)
    end

    return labels
end

"""Try to move item i to a better cluster (sweetening scan, Binder fast path)."""
function _salso_move_item!(labels, i, psm, draws, loss::BinderLoss, K_max)
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

"""Try to move item i to a better cluster (sweetening scan, VI full recomputation)."""
function _salso_move_item!(labels, i, psm, draws, loss::VILoss, K_max)
    from_k = labels[i]
    best_k = from_k

    current_eloss = _eloss(labels, psm, draws, loss)
    best_eloss = current_eloss
    old_label = labels[i]

    @inbounds for k = 1:maximum(labels)
        k == from_k && continue
        labels[i] = k
        eloss = _eloss(labels, psm, draws, loss)
        if eloss < best_eloss
            best_eloss = eloss
            best_k = k
        end
    end

    K_current = maximum(labels)
    if K_current < K_max
        @inbounds labels[i] = K_current + 1
        eloss = _eloss(labels, psm, draws, loss)
        if eloss < best_eloss
            best_eloss = eloss
            best_k = K_current + 1
        end
    end

    @inbounds begin
        labels[i] = old_label  # restore before branching

        if best_k != from_k
            labels[i] = best_k
            return true
        else
            labels[i] = from_k
            return false
        end
    end
end

"""
    _salso_best_move!(labels, i, psm, draws, loss::BinderLoss, K_max, rng)

Binder fast path of the zealous reallocation for item `i`, which arrives unassigned
(`labels[i] == 0`): one pass over row `i` scores every candidate cluster."""
function _salso_best_move!(
    labels::Vector{Int},
    i::Int,
    psm::Matrix{Float64},
    draws::Matrix{Int},
    loss::BinderLoss,
    K_max::Int,
    rng,
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

"""Place item i into the best cluster label (for zealous reallocation)."""
function _salso_best_move!(labels, i, psm, draws, loss, K_max, rng)
    best_k = 1
    best_eloss = Inf

    candidates = unique(labels)
    candidates = filter(x -> x > 0, candidates)
    K_current = length(candidates)

    for k in candidates
        @inbounds labels[i] = k
        eloss = _eloss(labels, psm, draws, loss)
        if eloss < best_eloss
            best_eloss = eloss
            best_k = k
        end
    end

    if K_current < K_max
        # maximum(labels)+1 (not K_current+1): labels can be non-contiguous
        # here, and the new-cluster candidate must not collide with an
        # existing cluster label
        new_k = maximum(labels) + 1
        @inbounds labels[i] = new_k
        eloss = _eloss(labels, psm, draws, loss)
        if eloss < best_eloss
            best_eloss = eloss
            best_k = new_k
        end
    end

    @inbounds labels[i] = best_k
end
