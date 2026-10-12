# ─── Clustering comparison metrics ───────────────────────────────────────────
#
# Pairwise comparison of two partitions: Binder losses, adjusted Rand index,
# variation of information, normalized mutual information, purity.  The losses
# SALSO minimises live in `salso.jl`; `BinderLoss` is the N-invariant form with
# one parameter, while `binder_loss` here is the classical two-weight one.
#
# Everything comes from the contingency table of the two partitions, O(n + K₁K₂)
# rather than O(n²), and the labels may be any integers.

# Above this many cells a dense table would be large (4e6 Int64 = 32 MB), which
# only happens for two very fine partitions; then a `Dict` is used instead.
const _CONTINGENCY_MAX = 4_000_000

"""
    _confusion(C₁, C₂) -> (counts₁, counts₂, cells)

What the metrics below need from two partitions: the size of each cluster, and
the non-empty contingency cells as `(row, column, count)`.  A dense table is used
while it stays small (the usual case, K₁, K₂ ≪ n), else a `Dict`; labels may be
any integers, including negative ones and gaps.
"""
function _confusion(C₁::Vector{Int}, C₂::Vector{Int})
    isempty(C₁) && return Dict{Int,Int}(), Dict{Int,Int}(), Tuple{Int,Int,Int}[]
    lo1, hi1 = extrema(C₁)
    lo2, hi2 = extrema(C₂)
    k1 = hi1 - lo1 + 1
    k2 = hi2 - lo2 + 1

    if k1 * k2 <= _CONTINGENCY_MAX
        ct = zeros(Int, k1, k2)
        @inbounds for i = eachindex(C₁, C₂)
            ct[C₁[i]-lo1+1, C₂[i]-lo2+1] += 1
        end
        # in the dense case the cell keys are table positions, so the marginals
        # are plain vectors: no hashing anywhere in the common path
        counts1 = vec(sum(ct, dims = 2))
        counts2 = vec(sum(ct, dims = 1))
        cells = Tuple{Int,Int,Int}[]
        for i = 1:k1, j = 1:k2
            ct[i, j] == 0 && continue
            push!(cells, (i, j, ct[i, j]))
        end
        return counts1, counts2, cells
    end

    counts1 = Dict{Int,Int}()
    counts2 = Dict{Int,Int}()
    ct = Dict{Tuple{Int,Int},Int}()
    @inbounds for i = eachindex(C₁, C₂)
        key = (C₁[i], C₂[i])
        ct[key] = get(ct, key, 0) + 1
    end
    cells = Tuple{Int,Int,Int}[]
    for ((a, b), m) in ct
        push!(cells, (a, b, m))
        counts1[a] = get(counts1, a, 0) + m
        counts2[b] = get(counts2, b, 0) + m
    end
    return counts1, counts2, cells
end

"""Cluster sizes of a partition, from `_confusion`'s marginals."""
_sizes(counts::Vector{Int}) = counts
_sizes(counts::Dict{Int,Int}) = values(counts)

"""Number of item pairs a partition keeps together, `Σ_c C(n_c, 2)`."""
function _same_pairs(counts)
    total = 0
    for c in _sizes(counts)
        total += c * (c - 1) ÷ 2
    end
    return total
end

"""
    binder_loss(C₁, C₂; a = 1.0, b = 1.0) -> Float64

Binder loss between two partitions: `a` for every pair of items that `C₁` puts
together and `C₂` separates (one kind of error), `b` for the pairs that `C₁`
separates and `C₂` puts together (the other kind).  Lower is better.
"""
function binder_loss(
    C₁::Vector{Int}, C₂::Vector{Int}; a::Real = 1.0, b::Real = 1.0)::Float64
    counts1, counts2, cells = _confusion(C₁, C₂)
    both = 0
    for (_, _, m) in cells
        both += m * (m - 1) ÷ 2
    end
    return float(a) * (_same_pairs(counts1) - both) +
           float(b) * (_same_pairs(counts2) - both)
end

"""
    binder_loss_n_invariant(c1, c2; a = 1.0, b = 1.0) -> Float64

`a·Σ_S (|S|/n)² + b·Σ_T (|T|/n)² − (a+b)·Σ_{S,T} (|S∩T|/n)²` over the clusters `S`
and `T` of the two partitions.  Unlike [`binder_loss`](@ref) it does not grow with
`n` for a fixed degree of disagreement.
"""
function binder_loss_n_invariant(
    c1::Vector{Int}, c2::Vector{Int}; a::Float64 = 1.0, b::Float64 = 1.0)::Float64
    n = length(c1)
    counts1, counts2, cells = _confusion(c1, c2)
    inv_n = 1.0 / n
    term1 = 0.0
    for c in _sizes(counts1)
        term1 += (c * inv_n)^2
    end
    term2 = 0.0
    for c in _sizes(counts2)
        term2 += (c * inv_n)^2
    end
    joint = 0.0
    for (_, _, m) in cells
        joint += (m * inv_n)^2
    end
    return a * term1 + b * term2 - (a + b) * joint
end

"""
    ari(C₁, C₂) -> Float64

Adjusted Rand index (Hubert & Arabie, 1985) between two label vectors, via
`Clustering.randindex`.  `1.0` is perfect agreement and `≈ 0` is chance level;
the label values themselves are irrelevant, only the partition matters.
"""
@inline ari(C₁::Vector{Int}, C₂::Vector{Int}) = randindex(C₁, C₂) |> first

"""
    vi(C₁, C₂) -> Float64

`H(C₁) + H(C₂) - 2 I(C₁, C₂)` (Meilă, 2003), in nats and normalised by the number
of items, via `Clustering.varinfo`; `0.0` is perfect agreement.  The objective SALSO
minimises is [`VILoss`](@ref).
"""
@inline vi(C₁::Vector{Int}, C₂::Vector{Int}) = varinfo(C₁, C₂)

"""
    nmi(C₁, C₂) -> Float64

`I / √(H(C₁)·H(C₂))` with entropies in nats, in `[0, 1]`: `1.0` for identical
partitions, `0.0` for independent ones or a partition without entropy (`K = 1`).
"""
function nmi(C₁::Vector{Int}, C₂::Vector{Int})::Float64
    n = length(C₁)
    counts1, counts2, cells = _confusion(C₁, C₂)
    inv_n = 1.0 / n
    H_X = 0.0
    for c in _sizes(counts1)
        c == 0 && continue          # empty label values in the dense table
        p = c * inv_n
        H_X -= p * log(p)
    end
    H_Y = 0.0
    for c in _sizes(counts2)
        c == 0 && continue
        p = c * inv_n
        H_Y -= p * log(p)
    end
    MI = 0.0
    for (a, b, m) in cells
        p = m * inv_n
        MI += p * log(p / (counts1[a] * inv_n * counts2[b] * inv_n))
    end
    den = sqrt(H_X * H_Y)
    return den > 1e-12 ? MI / den : 0.0
end

"""
    purity(C₁, C₂) -> Float64

Fraction of items in a `C₂` cluster that belong to its most frequent `C₁` cluster,
averaged over clusters: `(1/n)·Σ_j max_i |Cᵢ ∩ C₂ⱼ|`.
"""
function purity(C₁::Vector{Int}, C₂::Vector{Int})::Float64
    n = length(C₁)
    n == length(C₂) || error("Partitions must have the same length")
    n > 0 || return 1.0
    _, _, cells = _confusion(C₁, C₂)
    best = Dict{Int,Int}()          # column label -> largest cell count in it
    for (_, b, m) in cells
        m > get(best, b, 0) && (best[b] = m)
    end
    total = 0
    for m in values(best)
        total += m
    end
    return total / n
end

"""Number of non-empty clusters in one draw (labels `≤ 0` count as unassigned)."""
@inline count_cluster(draw::AbstractVector{<:Integer}) =
    length(unique(filter(>(0), _as_int_labels(draw))))
