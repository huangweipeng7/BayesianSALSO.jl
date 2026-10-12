# Correctness and regression tests for the SALSO search.
#
#   julia --project=. -e 'using Pkg; Pkg.test()'
#
# The loss functions are checked against independent transcriptions of their
# definitions (pairwise Binder for the PSM path, counts/combinatorial Binder for
# the same quantity via a different route, entropy + mutual information for the
# generalized VI), the incremental states are checked against the direct
# formulas, and the search is checked against brute-force enumeration of every
# partition on small instances.  A few golden values pin down the random
# trajectory of the search, so an accidental change in the order in which the
# algorithm draws from its RNG fails the suite.

using Test
using LinearAlgebra: diag
using Random
using SALSO
using SALSO:
    _canonicalize, _binder_delta, _VIState, _vi_reset!, _vi_assign!, _vi_full_loss,
    _vi_best_target, _binder_move_deltas, expected_loss

# ═══════════════════════════════════════════════════════════════════════════════
# Independent reference implementations
# ═══════════════════════════════════════════════════════════════════════════════

"""Count how often each label occurs."""
countmap(labels) =
    foldl((d, l) -> (d[l] = get(d, l, 0) + 1; d), labels; init = Dict{Int,Int}())

"""PSM by visiting every pair of every draw (the obvious O(B·n²) way)."""
function naive_psm(draws::Matrix{Int})
    B, n = size(draws)
    psm = ones(Float64, n, n)
    for i = 1:n, j = 1:n
        i == j && continue
        same = 0
        for b = 1:B
            draws[b, i] == draws[b, j] && (same += 1)
        end
        psm[i, j] = same / B
    end
    return psm
end

"""
Counts-based combinatorial Binder loss (Dahl, Johnson & Müller 2021) on the
N-invariant scale: with `C(x, 2) = x(x-1)/2` counting pairs, the loss is

    (2/n²)·[ a·E Σ_l C(n_l, 2) + (2-a)·Σ_c C(n_c, 2) - 2·E Σ_{cl} C(m_cl, 2) ]

where `n_c` are the estimate's cluster sizes, `n_l` a draw's cluster sizes and
`m_cl` the confusion counts.  This is the same number `expected_loss` obtains
from the PSM, reached without ever building the PSM.
"""
function counts_binder(labels::Vector{Int}, draws::Matrix{Int}, a::Float64)
    B, n = size(draws)
    est = _canonicalize(labels)
    K = maximum(est)
    Kb = maximum(maximum(_canonicalize(draws[b, :])) for b = 1:B)

    n_c = zeros(Int, K)
    for s in est
        n_c[s] += 1
    end
    pairs(x) = x * (x - 1) / 2
    P_est = sum(pairs, n_c)

    P_draw = 0.0        # E Σ_l C(n_l, 2)
    P_joint = 0.0       # E Σ_cl C(m_cl, 2)
    for b = 1:B
        d = _canonicalize(draws[b, :])
        n_l = zeros(Int, Kb)
        m = zeros(Int, K, Kb)
        for i = 1:n
            n_l[d[i]] += 1
            m[est[i], d[i]] += 1
        end
        P_draw += sum(pairs, n_l) / B
        P_joint += sum(pairs, m) / B
    end
    return 2 * (a * P_draw + (2 - a) * P_est - 2 * P_joint) / n^2
end

"""Entropy (bits) of a label vector, from its cluster sizes."""
function entropy_bits(labels::Vector{Int})
    n = length(labels)
    p = [c / n for c in values(countmap(labels))]
    return -sum(x -> x * log2(x), p)
end

"""Mutual information (bits) of a contingency table given as counts."""
function mutual_info_bits(conf::Matrix{Int})
    n = sum(conf)
    row = vec(sum(conf, dims = 2)) ./ n
    col = vec(sum(conf, dims = 1)) ./ n
    mi = 0.0
    for i in axes(conf, 1), j in axes(conf, 2)
        conf[i, j] == 0 && continue
        p = conf[i, j] / n
        mi += p * log2(p / (row[i] * col[j]))
    end
    return mi
end

"""
Generalized VI loss `a·H(π̂) + (2-a)·E[H(ρ)] - 2·E[I(π̂,ρ)]` in bits (Dahl,
Johnson & Müller 2021, Eq. (12)), with every entropy and mutual information
recomputed from scratch — the closed form `expected_loss(::VILoss)` is supposed
to evaluate exactly this.
"""
function entropy_vi(labels::Vector{Int}, draws::Matrix{Int}, a::Float64)
    B, n = size(draws)
    est = _canonicalize(labels)
    K = maximum(est)
    Kb = maximum(maximum(_canonicalize(draws[b, :])) for b = 1:B)

    E_H = 0.0
    E_I = 0.0
    for b = 1:B
        d = _canonicalize(draws[b, :])
        E_H += entropy_bits(d) / B
        conf = zeros(Int, K, Kb)
        for i = 1:n
            conf[est[i], d[i]] += 1
        end
        E_I += mutual_info_bits(conf) / B
    end
    return a * entropy_bits(est) + (2 - a) * E_H - 2 * E_I
end

"""
The pre-rewrite `_binder_delta`: one branch per item, skipping item `i` and every
unassigned item.  Kept as the reference for the unassigned case, where the
branchless form once treated label 0 as a cluster.
"""
function delta_reference(labels, psm, i, from_k, to_k, loss::BinderLoss)
    c = loss.two_minus_a
    delta = 0.0
    from_k == to_k && return 0.0
    @inbounds for j = 1:length(labels)
        j == i && continue
        lj = labels[j]
        lj == 0 && continue
        p = psm[i, j]
        if lj == from_k
            delta += 2.0 * p - c
        elseif lj == to_k
            delta += c - 2.0 * p
        end
    end
    n = length(labels)
    return 2.0 * delta / (n * n)
end

"""Every partition of `n` items, as label vectors with labels `1:K`."""
function all_partitions(n::Int)
    out = Vector{Vector{Int}}()
    lab = zeros(Int, n)
    function rec(i::Int, mx::Int)
        if i > n
            push!(out, copy(lab))
            return
        end
        for v = 1:(mx+1)
            lab[i] = v
            rec(i + 1, max(mx, v))
        end
    end
    rec(1, 0)
    return out
end

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pairwise similarity matrix
# ═══════════════════════════════════════════════════════════════════════════════

@testset "pairwise similarity matrix" begin
    rng = MersenneTwister(1)
    for (B, n, K) in [(1, 5, 2), (7, 12, 3), (40, 60, 4), (3, 200, 5)]
        draws = rand(rng, 1:K, B, n)
        psm = compute_psm(draws)
        @test psm ≈ naive_psm(draws)
        @test all(diag(psm) .== 1.0)
        @test psm == transpose(psm)
        # arbitrary (non-contiguous, negative) labels must not matter
        @test compute_psm(draws .* 7 .- 11) == psm
    end
    # label values are arbitrary, but an all-in-one-cluster draw is all ones
    draws = rand(MersenneTwister(2), 1:4, 15, 25)
    @test compute_psm(ones(Int, 5, 25)) == ones(Float64, 25, 25)
    @test compute_psm(zeros(Int, 0, 25)) == zeros(Float64, 25, 25)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Expected losses
# ═══════════════════════════════════════════════════════════════════════════════

@testset "expected Binder loss" begin
    rng = MersenneTwister(3)
    for (B, n, K) in [(5, 10, 3), (200, 40, 4)], a in (0.0, 0.5, 1.0, 1.7, 2.0)
        draws = rand(rng, 1:K, B, n)
        labels = rand(rng, 1:K, n)
        psm = compute_psm(draws)
        got = expected_loss(labels, psm, BinderLoss(a))
        # the PSM path, the counts path and the raw pairwise sum agree
        @test got ≈ counts_binder(labels, draws, a) atol = 1e-12
        pairwise = 2 * sum(
            labels[i] == labels[j] ? (2 - a) * (1 - psm[i, j]) : a * psm[i, j] for
            i = 1:n for j = (i+1):n
        ) / n^2
        @test got ≈ pairwise atol = 1e-12
        @test expected_loss(labels, draws, BinderLoss(a)) ≈ got atol = 1e-12
    end
    # a = 0 / a = 2 extremes: with a = 0 only clustering mistakes are paid for
    # (the complement 2 - a = 2), with a = 2 only separating ones
    draws = [1 1 2 2; 1 2 2 1]
    psm = compute_psm(draws)
    @test expected_loss([1, 2, 3, 4], psm, BinderLoss(0.0)) ≈ 0.0 atol = 1e-14
    @test expected_loss([1, 1, 1, 1], psm, BinderLoss(2.0)) ≈ 0.0 atol = 1e-14
    # items labelled 0 are unassigned and skipped: only the pair (1, 2) counts
    # here, and its expected cost is 2·(2 - a)·(1 - psm[1, 2]) / n²
    @test expected_loss([1, 1, 0, 0], psm, BinderLoss(1.0)) ≈
          2 * (2 - 1.0) * (1 - psm[1, 2]) / 4^2 atol = 1e-14
    @test BinderLoss(1.0).two_minus_a == 1.0
    @test_throws AssertionError BinderLoss(2.5)
end

@testset "expected VI loss" begin
    rng = MersenneTwister(4)
    for (B, n, K) in [(5, 10, 3), (120, 50, 5)], a in (0.0, 0.5, 1.0, 2.0)
        draws = rand(rng, 1:K, B, n)
        labels = rand(rng, 1:K, n)
        @test expected_loss(labels, draws, VILoss(a)) ≈ entropy_vi(labels, draws, a) atol =
            1e-12
    end
    # identical partitions cost nothing
    draws = repeat([1 1 2 2 3], 9, 1)
    @test expected_loss([2, 2, 1, 1, 3], draws, VILoss(1.0)) ≈ 0.0 atol = 1e-14
    @test VILoss(0.5).two_minus_a == 1.5
    @test_throws AssertionError VILoss(-0.1)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Incremental state and deltas
# ═══════════════════════════════════════════════════════════════════════════════

@testset "incremental VI state" begin
    rng = MersenneTwister(5)
    for trial = 1:20
        n = rand(rng, 6:30)
        B = rand(rng, 3:15)
        Kmax = rand(rng, 1:n)
        draws = rand(rng, 1:rand(rng, 1:4), B, n)
        a = rand(rng) * 2
        loss = VILoss(a)
        D = Matrix{Int}(undef, B, n)
        for b = 1:B
            D[b, :] = _canonicalize(draws[b, :])
        end
        s = _VIState(D, a, Kmax)
        _vi_reset!(s, rand(rng, 1:Kmax, n))
        # the state's incrementally maintained loss matches the closed form
        @test _vi_full_loss(s) ≈ expected_loss(s.labels, draws, loss) atol = 1e-12
        for _ = 1:5
            _vi_assign!(s, rand(rng, 1:n), rand(rng, 0:Kmax))
            @test _vi_full_loss(s) ≈ expected_loss(s.labels, draws, loss) atol = 1e-12
        end
        # _vi_best_target picks the target attaining the minimum full loss over
        # the occupied clusters and one fresh one (all empty labels are alike)
        for i = 1:n
            target = _vi_best_target(s, i)
            @test 1 <= target <= Kmax
            keep = s.labels[i]
            best = Inf
            for t in unique([s.labels; collect(1:Kmax)])
                t == 0 && continue
                s.labels[i] = t
                best = min(best, expected_loss(s.labels, draws, loss))
            end
            s.labels[i] = target
            @test expected_loss(s.labels, draws, loss) <= best + 1e-10
            s.labels[i] = keep
        end
    end
end

@testset "O(n) Binder delta" begin
    rng = MersenneTwister(6)
    for trial = 1:20
        n = rand(rng, 5:40)
        B = rand(rng, 2:12)
        draws = rand(rng, 1:rand(rng, 1:4), B, n)
        psm = compute_psm(draws)
        a = rand(rng) * 2
        loss = BinderLoss(a)
        labels = rand(rng, 1:3, n)
        i = rand(rng, 1:n)
        from_k = labels[i]
        for to_k in [1:4; maximum(labels)+1]
            to_k == from_k && continue
            before = expected_loss(labels, psm, loss)
            moved = copy(labels)
            moved[i] = to_k
            after = expected_loss(moved, psm, loss)
            @test _binder_delta(labels, psm, i, from_k, to_k, loss) ≈ after - before atol =
                1e-12
        end
        @test _binder_delta(labels, psm, i, from_k, from_k, loss) == 0.0
        # one row pass must give every candidate the same delta as the single
        # call (up to summation order), for the assigned case ...
        hi = maximum(labels) + 1
        deltas = _binder_move_deltas(labels, psm, i, from_k, hi, loss)
        for k = 1:hi
            @test deltas[k] ≈ _binder_delta(labels, psm, i, from_k, k, loss) atol = 1e-12
        end
        # ... and for the unassigned case, where item i is one of several items
        # labelled 0: those must not be treated as a cluster (a regression that
        # the candidate-difference test below cannot see, since it cancels)
        if n >= 6
            unassigned = labels
            for j in randperm(rng, n)[1:3]
                unassigned[j] = 0
            end
            i2 = findfirst(==(0), unassigned)
            hi2 = maximum(unassigned) + 1
            deltas0 = _binder_move_deltas(unassigned, psm, i2, 0, hi2, loss)
            reference = [
                delta_reference(unassigned, psm, i2, 0, k, loss) for k = 1:hi2
            ]
            @test deltas0 ≈ reference atol = 1e-12
        end
    end
end


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Search
# ═══════════════════════════════════════════════════════════════════════════════

"""A small, noisy draw matrix used by the brute-force and compatibility tests."""
function small_instance(seed::Int, n::Int, B::Int)
    rng = MersenneTwister(seed)
    draws = rand(rng, 1:3, B, n)
    return [rand(rng) < 0.35 ? rand(rng, 1:3) : c for c in draws]
end

@testset "search finds the global optimum" begin
    draws = small_instance(7, 8, 400)
    parts = all_partitions(8)
    # With `max_n_clusters = 0` the search ceiling is the largest number of
    # clusters seen in a draw, exactly as in the R reference implementation, so
    # the brute-force comparison has to respect that ceiling.
    cap = maximum(maximum(_canonicalize(draws[b, :])) for b = 1:size(draws, 1))
    for loss in (VILoss(1.0), BinderLoss(1.0), BinderLoss(0.5), VILoss(0.5))
        best = minimum(expected_loss(p, draws, loss) for p in parts if maximum(p) <= cap)
        res = salso(
            draws;
            loss = loss,
            n_runs = 16,
            max_zealous_attempts = 50,
            rng = MersenneTwister(11),
        )
        @test res.expected_loss ≈ best atol = 1e-10
        @test res.expected_loss ≈ expected_loss(res.partition, draws, loss) atol = 1e-12
        @test res.n_clusters == maximum(res.partition)
        @test res.partition == _canonicalize(res.partition)
    end
    # Given room to grow, the search reaches the unrestricted optimum as well.
    for loss in (VILoss(1.0), BinderLoss(1.0))
        best = minimum(expected_loss(p, draws, loss) for p in parts)
        res = salso(
            draws;
            loss = loss,
            n_runs = 32,
            max_zealous_attempts = 100,
            max_n_clusters = 8,
            rng = MersenneTwister(11),
        )
        @test res.expected_loss ≈ best atol = 1e-10
    end
end

@testset "result bookkeeping" begin
    draws = rand(MersenneTwister(8), 1:4, 60, 30)
    res = salso(draws; loss = BinderLoss(1.0), n_runs = 3, rng = MersenneTwister(9))
    @test res.n_runs == 3
    @test 0 <= res.n_zealous_accepts <= res.n_zealous_attempts
    @test res.n_scans >= 1
    @test res.expected_loss ≈
          expected_loss(res.partition, compute_psm(draws), BinderLoss(1.0)) atol = 1e-12
    @test occursin("SALSOResult", sprint(show, res))
    res1 = salso(
        draws;
        loss = VILoss(1.0),
        n_runs = 1,
        seconds = 100.0,
        rng = MersenneTwister(9),
    )
    @test res1.n_runs == 1
    # a search that never starts must not return a garbage result
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0), n_runs = 0)
    @test_throws ArgumentError salso(
        draws;
        loss = VILoss(1.0),
        n_runs = 2,
        seconds = -1.0,
    )
end

@testset "boundary items" begin
    # the Binder delta splits the row into the two ranges around item i, so the
    # first and last item exercise the empty-range paths
    draws = rand(MersenneTwister(19), 1:3, 50, 9)
    psm = compute_psm(draws)
    labels = _canonicalize(draws[1, :])
    loss = BinderLoss(1.0)
    for i in (1, length(labels))
        from_k = labels[i]
        to_k = from_k == 3 ? 2 : 3
        before = expected_loss(labels, psm, loss)
        moved = copy(labels)
        moved[i] = to_k
        @test _binder_delta(labels, psm, i, from_k, to_k, loss) ≈
              expected_loss(moved, psm, loss) - before atol = 1e-12
        deltas = _binder_move_deltas(labels, psm, i, from_k, 4, loss)
        for k = 1:4
            @test deltas[k] ≈ _binder_delta(labels, psm, i, from_k, k, loss) atol = 1e-12
        end
    end
end

@testset "degenerate instance terminates" begin
    # data/test.csv: four items, three draws.  With BinderLoss the moves are tied
    # to within floating-point rounding, so a pair of them can each look
    # improving (-2.8e-17) and the sweetening cycle forever.  The scan count must
    # stay small even when the budget is large.
    tiny = permutedims([1 2 1; 2 4 2; 2 3 2; 2 3 1])
    res = salso(tiny; loss = BinderLoss(1.0), n_runs = 4, max_scans = 500,
        rng = MersenneTwister(3))
    @test res.n_scans < 50
    @test isfinite(res.expected_loss)
    res_vi = salso(tiny; loss = VILoss(1.0), n_runs = 4, max_scans = 500,
        rng = MersenneTwister(3))
    @test res_vi.n_scans < 50
end

@testset "input types" begin
    draws = rand(MersenneTwister(10), 1:4, 40, 20)
    # a fresh RNG per call, so the runs sample the search identically
    run(d) = salso(d; loss = VILoss(1.0), n_runs = 4, rng = MersenneTwister(12))
    frommatrix = run(draws)
    @test run(Int32.(draws)) == frommatrix      # non-Int label type
    @test run(view(draws, :, :)) == frommatrix  # non-Matrix wrapper
    @test run(Matrix{Int}(draws)) == frommatrix # explicit copy
    @test compute_psm(view(draws, :, :)) == compute_psm(draws)
end

@testset "golden values (random trajectory)" begin
    # Recorded when the search was moved onto the incremental VI state; they pin
    # the order in which the driver draws from its RNG.
    golden = (
        (B = 60, n = 30, runs = 4, loss = 1.926121209744790, K = 1),
        (B = 200, n = 80, runs = 6, loss = 1.970905594100702, K = 1),
    )
    for g in golden
        draws = rand(MersenneTwister(5), 1:4, g.B, g.n)
        res = salso(draws; loss = VILoss(1.0), n_runs = g.runs, rng = MersenneTwister(77))
        @test res.expected_loss ≈ g.loss atol = 1e-9
        @test res.n_clusters == g.K
    end
end

@testset "cluster ceiling" begin
    # two well separated groups of three items
    draws = repeat([1 1 1 2 2 2; 1 1 2 2 2 2; 1 1 1 2 2 2], 20, 1)
    res1 = salso(
        draws;
        loss = VILoss(1.0),
        n_runs = 4,
        max_n_clusters = 1,
        rng = MersenneTwister(13),
    )
    @test res1.n_clusters == 1
    @test res1.partition == ones(Int, 6)
    res2 = salso(draws; loss = VILoss(1.0), n_runs = 4, rng = MersenneTwister(13))
    @test res2.n_clusters == 2
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Argument checking
# ═══════════════════════════════════════════════════════════════════════════════

@testset "non-integer labels are rejected" begin
    draws = rand(MersenneTwister(15), 1:4, 30, 10)
    # a boolean mask would become the label 0, which means *unassigned*
    @test_throws ArgumentError salso(Bool.(draws .== 1); n_runs = 2)
    # floats hide a real mistake: Int. would truncate or invent clusters
    @test_throws ArgumentError salso(Float64.(draws); n_runs = 2)
    @test_throws ArgumentError compute_psm(Float64.(draws))
    # codes (strings, symbols, categorical) are not labels either
    @test_throws ArgumentError salso([:a :b ; :b :a])
    # also on the metric side, where the labels are plain vectors
    @test_throws ArgumentError ari([1.0, 2.0, 2.0], [1, 2, 2])
    # the message has to say what to do, not just that it failed
    msg = try
        salso(Float64.(draws); n_runs = 2)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("must be integers", msg)
    @test occursin("Int.(round.", msg)
    @test_throws ArgumentError expected_loss([1.0, 2.0, 2.0], draws, VILoss(1.0))
end

@testset "wrong shapes are rejected with a hint" begin
    draws = rand(MersenneTwister(16), 1:4, 30, 10)
    @test_throws ArgumentError salso(draws[1, :]; n_runs = 2)                 # one draw
    @test_throws ArgumentError salso([draws[b, :] for b = 1:3]; n_runs = 2)   # vector of draws
    # a string or `nothing` is not a SALSOLoss, so dispatch turns it into a TypeError
    @test_throws TypeError salso(draws; loss = "vi", n_runs = 2)
    @test_throws TypeError salso(draws; loss = nothing, n_runs = 2)
    # keyword arithmetic and bounds: invalid numbers are clear errors
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0), n_runs = -1)
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0), max_scans = -1)
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0),
        prob_sequential_allocation = 1.5)
    # a zero budget is refused too, with the same error as a negative one
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0), max_scans = 0)
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0), n_runs = 2, seconds = 0.0)
    @test_throws ArgumentError salso(draws; loss = VILoss(1.0),
        prob_sequential_allocation = NaN)
    @test_throws MethodError salso(draws; nonsense = 1)                       # unknown option
    # empty inputs are refused with a clear message
    @test_throws ArgumentError salso(zeros(Int, 3, 0); loss = VILoss(1.0), n_runs = 2)
    @test_throws ArgumentError salso(zeros(Int, 0, 3); loss = VILoss(1.0), n_runs = 2)
    # but a whole number is not a mistake: `seconds = 10` and narrow integers work
    @test salso(draws; loss = VILoss(1.0), n_runs = 2, seconds = 10) isa SALSOResult
    @test salso(draws; loss = VILoss(1.0), n_runs = Int32(2), max_scans = Int16(50)) isa
          SALSOResult
    @test salso(draws; loss = VILoss(1.0), n_runs = 2, prob_sequential_allocation = 1 // 2) isa
          SALSOResult
end

@testset "integers of any width are accepted" begin
    draws = rand(MersenneTwister(17), 1:4, 40, 20)
    cfg = (loss = BinderLoss(1), n_runs = 4, rng = MersenneTwister(18))  # Int cost, too
    reference = salso(draws; cfg...)
    @test salso(Int16.(draws); loss = BinderLoss(1.0), n_runs = 4, rng = MersenneTwister(18)) ==
          reference
    @test expected_loss(Int8.(draws[1, :]), draws, VILoss(1.0)) isa Float64
    @test ari(Int8[1, 2, 1, 2], Int16[1, 2, 1, 2]) == 1.0
    # a narrow label vector is converted once, not once for validation and again
    # for the metrics themselves
    narrow = Int32.(mod1.(1:500, 7))
    pack() = SALSO._metric_labels(narrow)
    pack()
    @test (@allocated pack()) < 6000      # one Vector{Int} (4000 B), not two
    @test VILoss(1).a == 1.0
    @test BinderLoss(1).two_minus_a == 1.0
    @test VILoss(1 // 2).a == 0.5         # any Real, not just Float64/Int
    @test BinderLoss(0.5f0).a == 0.5
end

@testset "metrics" begin
    truth = [1, 1, 2, 2, 3, 3]
    # degenerate inputs
    @test binder_loss(Int[], Int[]) == 0.0
    @test binder_loss_n_invariant(Int[], Int[]) == 0.0
    @test binder_loss([1], [1]) == 0.0
    @test purity(Int[], Int[]) == 1.0
    # ari and vi go through Clustering, which cannot take empty vectors
    @test ari(Int[], Int[]) == 1.0
    @test vi(Int[], Int[]) == 0.0
    @test_throws DimensionMismatch ari([1, 2], [1])
    @test_throws DimensionMismatch vi([1, 2], [1])
    # labels need not be 1..K, or even positive
    c1 = [-3, -3, 7, 7]
    c2 = [5, 5, -1, -1]
    @test purity(c1, c2) == 1.0
    @test binder_loss(c1, c2) == 0.0
    @test binder_loss_n_invariant(c1, c2) ≈ 0.0 atol = 1e-12
    @test isapprox(nmi(c1, c2), 1.0)
    @test ari(truth, truth) == 1.0
    @test ari(truth, [2, 2, 3, 3, 1, 1]) == 1.0             # label names are arbitrary
    @test 0.0 <= ari(truth, [1, 1, 1, 2, 2, 2]) < 1.0
    @test vi(truth, truth) == 0.0
    @test vi(truth, [1, 1, 1, 1, 1, 1]) > 0
    @test nmi(truth, truth) ≈ 1.0
    @test purity(truth, truth) == 1.0
    @test purity(truth, [1, 2, 1, 2, 3, 3]) == 4 / 6
    @test binder_loss([1, 1, 2], [1, 2, 2]; a = 1.0, b = 1.0) == 2.0
    @test binder_loss_n_invariant([1, 1, 2], [1, 2, 2]; a = 1.0, b = 1.0) ≈ 4 / 9
    # exported, and it validates labels like everything else
    @test count_cluster([1, 2, 2, 0]) == 2
    @test count_cluster(Int8[1, 1, 2]) == 2
    @test_throws ArgumentError count_cluster([1.0, 2.0])
    # a loss needs the labels and the partition it scores to match
    @test_throws DimensionMismatch expected_loss([1, 2], ones(3, 3), BinderLoss(1.0))
    @test_throws DimensionMismatch expected_loss([1, 2], ones(Int, 4, 5), VILoss(1.0))
end
