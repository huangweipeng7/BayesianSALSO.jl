# ─── VI loss and search ───────────────────────────────────────────────────────
#
# The expected VI loss is evaluated from the draws with a reusable confusion
# buffer.  The search keeps cluster sizes and confusion counts incrementally, so
# a candidate move costs O(B).

# ─── Loss and expected loss ───────────────────────────────────────────────────

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
            counts[(s-1)*Kb_cap+l] += 1
            marg[l] += 1
        end
        for l = 1:Kb
            nl = marg[l]
            nl > 0 && (draw_side += a * nl * log2(nl))
        end
        for s = 1:K
            base = (s - 1) * Kb_cap
            for l = 1:Kb
                m = counts[base+l]
                m > 0 && (draw_side -= 2 * m * log2(m))
            end
        end
    end

    return (est + draw_side / B) / n
end

# ─── Incremental state ────────────────────────────────────────────────────────

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
    conf::Array{Int,3}     # conf[t, l, b] = #items in cluster t with draw label l
    n_of::Vector{Int}      # n_of[t] = size of candidate cluster t (t = 0 ⇒ unassigned)
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
        zeros(Int, K_max + 1, Kb, B),
        zeros(Int, K_max + 1),
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
    @inbounds for t = 1:(s.K_max)
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
    @inbounds for i = 1:(s.n)
        t = s.labels[i]
        t == 0 && continue
        s.n_of[t] += 1
        for b = 1:(s.B)
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
        @inbounds for b = 1:(s.B)
            s.conf[f, s.D[b, i], b] -= 1
        end
    end
    @inbounds s.labels[i] = t
    if t != 0
        @inbounds s.n_of[t] += 1
        @inbounds for b = 1:(s.B)
            s.conf[t, s.D[b, i], b] += 1
        end
    end
    return
end

"""Expected VI loss of the current state, on the same scale as `expected_loss`."""
function _vi_full_loss(s::_VIState)
    term_c = 0.0
    @inbounds for t = 1:(s.K_max)
        term_c += s.kern[Int(s.n_of[t])+1]
    end
    term_c *= (2.0 - s.a)

    joint = 0.0
    @inbounds for b = 1:(s.B), l = 1:(s.Kb)
        row = 0.0
        @simd for t = 1:(s.K_max)
            row += s.kern[Int(s.conf[t, l, b])+1]
        end
        joint += row
    end

    return (term_c + (s.a * s.draw_const - 2.0 * joint) / s.B) / s.n
end

"""Smallest unused cluster label, or `nothing` when the ceiling is reached."""
@inline function _vi_empty_label(s::_VIState)
    @inbounds for t = 1:(s.K_max)
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
    @inbounds for t = 1:(s.K_max)
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
        for b = 1:(s.B)
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

# ─── Greedy search ────────────────────────────────────────────────────────────

"""VI reallocation step: move item `i` to its best target cluster."""
@inline function _vi_sweeten_step(s::_VIState, labels, i::Int)
    t = _vi_best_target(s, i)
    @inbounds t == labels[i] && return false
    _vi_assign!(s, i, t)
    return true
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

        _vi_full_loss(s) < eloss_before ? (n_zaccepts += 1) : _vi_reset!(s, old_labels)   # undo
    end
    return n_zaccepts, n_zattempts, n_zscans
end

"""VI search driver: the incremental confusion table makes one candidate move cost
O(B), against O(K·n·K_b·B) for recomputing `expected_loss(labels, draws, ::VILoss)`.
"""
function _salso_search(
    draws::Matrix{Int},
    loss::VILoss;
    n_runs::Int,
    max_zealous_attempts::Int,
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
        n_zaccepts, n_zattempts, n_zscans =
            _salso_zealous_vi!(state, max_zealous_attempts, max_scans, rng)

        eloss = _vi_full_loss(state)
        if eloss < best_eloss
            best_eloss = eloss
            best_labels = copy(state.labels)
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
