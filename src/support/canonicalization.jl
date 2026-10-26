# ─── Cluster-label canonicalisation ───────────────────────────────────────────
#
# Canonical labels are 1..K in order of first appearance; label 0 is unassigned.

# `_canonicalize!` runs once per draw in the VI path and the capacity scan, so
# it must not allocate: the scratch holds the label → id map, the output buffer
# and a generation counter, which lets the map be reused without clearing it.
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
            fill!(view(sc.stamps, (old+1):span), 0)
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
