# Argument checking around `salso.jl`, which is otherwise untouched: labels must
# be integers (any width), and the mistakes users make - a float or boolean label
# array, a single draw instead of a B × n matrix, a vector of draws - get a
# message instead of a MethodError or a silent `Int.()` conversion.

"""Error out with an actionable message about non-integer cluster labels."""
function _label_type_error(labels)
    kind =
        labels isa AbstractMatrix ? "matrix" :
        labels isa AbstractVector ? "vector" : "array"
    throw(
        ArgumentError(
            "cluster labels must be integers, but this $kind has element type " *
            "$(eltype(labels)).  Use integer labels: whole numbers stored as " *
            "floats can be converted deliberately with `Int.(round.(labels))`, " *
            "and codes with a dictionary.  Booleans are rejected on purpose — " *
            "`false` would become the label 0, which means *unassigned*.",
        ),
    )
end

"""Accept integer labels of any width, converting them to `Int`."""
_as_int_labels(labels::AbstractVector{Int}) = labels
_as_int_labels(labels::AbstractVector{Bool}) = _label_type_error(labels)
_as_int_labels(labels::AbstractVector{<:Integer}) = Int.(labels)
_as_int_labels(labels) = _label_type_error(labels)

"""Accept integer draw matrices of any width, converting them to `Matrix{Int}`."""
_as_int_draws(draws::Matrix{Int}) = draws
_as_int_draws(draws::AbstractMatrix{Bool}) = _label_type_error(draws)
_as_int_draws(draws::AbstractMatrix{<:Integer}) = Matrix{Int}(draws)
_as_int_draws(draws::AbstractMatrix) = _label_type_error(draws)

# ─── salso ───────────────────────────────────────────────────────────────────

"""
    salso(draws::AbstractMatrix; loss = VILoss(), kwargs...)

Accept any integer label matrix (any integer width, any array wrapper) and hand
off to the algorithm in `salso.jl`: `invoke` calls that method directly, so a
non-integer label type produces one clear error instead of a dispatch loop.  A
`loss` that is not a loss object is caught by the algorithm's own keyword
signature, which reports the type it expected.
"""
function salso(draws::AbstractMatrix; loss = VILoss(), kwargs...)
    return invoke(salso, Tuple{Matrix{Int}}, _as_int_draws(draws); loss = loss, kwargs...)
end

"""
    salso(draws::AbstractVector; kwargs...)

Catch the two vector-shaped mistakes: a `Vector{Vector{Int}}` of draws (needs
stacking into a matrix) and a single label vector (not enough draws to define a
posterior).
"""
function salso(draws::AbstractVector; kwargs...)
    if !isempty(draws) && first(draws) isa AbstractVector
        throw(
            ArgumentError(
                "salso expects a `B × n` matrix of labels (one row per draw), but " *
                "got a vector of $(length(draws)) label vectors.  Stack the draws " *
                "into a matrix — `reduce(vcat, [reshape(d, 1, :) for d in draws])`.",
            ),
        )
    end
    if eltype(draws) <: Integer && !(eltype(draws) === Bool)
        throw(
            ArgumentError(
                "salso needs the posterior draws as a `B × n` matrix (one row per " *
                "draw): a single label vector has no posterior to minimise over " *
                "(its own loss is trivially zero).  Pass the matrix of draws.",
            ),
        )
    end
    return _label_type_error(draws)
end

# ─── expected_loss / compute_psm ─────────────────────────────────────────────
# Same treatment as `salso`: validate the labels, accept any integer width.

expected_loss(labels::AbstractVector, psm::Matrix{Float64}, loss::BinderLoss) =
    expected_loss(_as_int_labels(labels), psm, loss)

expected_loss(labels::AbstractVector, draws::Matrix{Int}, loss::VILoss) =
    expected_loss(_as_int_labels(labels), draws, loss)

expected_loss(labels::AbstractVector, draws::AbstractMatrix{<:Integer}, loss::VILoss) =
    expected_loss(labels, _as_int_draws(draws), loss)

compute_psm(draws::AbstractMatrix{<:Integer}) = compute_psm(_as_int_draws(draws))

compute_psm(draws::AbstractMatrix) = _label_type_error(draws)

# ─── loss constructors ───────────────────────────────────────────────────────

# `BinderLoss(1)` / `VILoss(0)` are natural things to write; the algorithm's
# constructors take `Float64`, so widen them rather than making the caller guess.
BinderLoss(a::Integer) = BinderLoss(Float64(a))
VILoss(a::Integer) = VILoss(Float64(a))

# ─── pairwise metrics ────────────────────────────────────────────────────────
#
# The metrics in `metric.jl` are typed on `Vector{Int}`; accept any integer
# container, and fail the same way as everywhere else on non-integer labels.

"""Validate label vectors and pack them as `Vector{Int}` for the metrics."""
function _metric_labels(labels::AbstractVector)
    labels = _as_int_labels(labels)     # validates, and converts once when needed
    return labels isa Vector{Int} ? labels : collect(Int, labels)
end

ari(C₁::AbstractVector, C₂::AbstractVector) = ari(_metric_labels(C₁), _metric_labels(C₂))
vi(C₁::AbstractVector, C₂::AbstractVector) = vi(_metric_labels(C₁), _metric_labels(C₂))
nmi(C₁::AbstractVector, C₂::AbstractVector) = nmi(_metric_labels(C₁), _metric_labels(C₂))
purity(C₁::AbstractVector, C₂::AbstractVector) =
    purity(_metric_labels(C₁), _metric_labels(C₂))
binder_loss(C₁::AbstractVector, C₂::AbstractVector; kwargs...) =
    binder_loss(_metric_labels(C₁), _metric_labels(C₂); kwargs...)
binder_loss_n_invariant(C₁::AbstractVector, C₂::AbstractVector; kwargs...) =
    binder_loss_n_invariant(_metric_labels(C₁), _metric_labels(C₂); kwargs...)
