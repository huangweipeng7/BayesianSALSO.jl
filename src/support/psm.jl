# ─── Pairwise similarity matrix ───────────────────────────────────────────────
#
# The PSM is accumulated into one shared matrix by splitting its rows across
# workers.  Exact integer counts keep the result independent of thread scheduling.

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
Add the co-occurrence counts of all draws to the lower triangle of `psm`.

Items are bucketed by label, so a draw with K_b clusters costs Σ|bucket|²/2 updates
instead of n²/2.  The counts are exact integers, so the result does not depend on
the order of accumulation."""
function _psm_accumulate!(psm::Matrix{Float64}, C_mat::Matrix{Int}, off::Int, K_b::Int)
    n = size(C_mat, 2)
    counts = zeros(Int, K_b + 1)
    starts = Vector{Int}(undef, K_b + 2)
    cursor = Vector{Int}(undef, K_b + 1)
    order = Vector{Int}(undef, n)

    @inbounds for b in axes(C_mat, 1)
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
turn the O(n²) pair count into Σ|bucket|²/2 updates.  This repeats the bucketing
for each worker, but the pair updates dominate for the inputs this path targets."""
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
    order = Vector{Int}(undef, n)

    @inbounds for b = 1:size(C_mat, 1)
        _psm_accumulate_draw_rows!(
            psm,
            @view(C_mat[b, :]),
            off,
            K_b,
            counts,
            starts,
            cursor,
            order,
            r0,
            r1,
        )
    end

    return psm
end

"""
Row blocks `r0:r1` of `n` rows for `nworkers` workers.

The blocks are cut on cache-line boundaries (8 doubles): in a column-major matrix
the rows of a block are 8 bytes apart, so two neighbouring blocks would
otherwise share the cache lines at their boundary and ping-pong them."""
function _psm_row_blocks(n::Int, nworkers::Int)
    nlines = cld(n, _PSM_CACHE_LINE)          # lines per column
    1 <= nworkers <= nlines ||
        throw(ArgumentError("need 1 ≤ nworkers ≤ $nlines cache-line blocks, got $nworkers"))

    blocks = Vector{Tuple{Int,Int}}(undef, nworkers)
    per, extra = divrem(nlines, nworkers)
    first_line = 0
    @inbounds for w = 1:nworkers
        width = per + (w <= extra)
        l0 = first_line
        l1 = l0 + width
        first_line = l1
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
        _psm_accumulate!(psm, C_mat, off, K_b)
    end

    return _psm_normalize!(psm, n, B)
end
