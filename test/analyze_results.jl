#!/usr/bin/env julia
# BayesianSALSO point estimate from a CSV of cluster assignments.
#
# The input CSV has one row per observation and one column per MCMC draw:
#
#   sample1_mcmc1,sample1_mcmc2,...
#   sample2_mcmc1,sample2_mcmc2,...
#
# The point estimate is written as a one-column CSV (`salso_partition`), with
# the labels of `salso`'s winning partition, and a summary is printed.
#
# Usage:
#   julia --project=. -t4 test/analyze_results.jl \
#       --input_file=data/test.csv --output_file=test_part.csv
#
#   julia --project=. -t4 test/analyze_results.jl --input_file=draws.csv --output_file=part.csv \
#       --loss=vi --loss_a=0.5 --n_runs=16 --max_zealous=50 --seed=42
#
# Options (all of the form `--option=value` or `--option value`):
#
#   --input_file   CSV of labels, rows = observations, columns = MCMC draws
#   --output_file  CSV to write, holding the point estimate
#   --loss         `binder` (default) or `vi`
#   --loss_a       cost parameter a ∈ [0, 2] of the loss (default 1.0, the
#                  symmetric case; 2 - a is the cost of the other mistake)
#   --n_runs       number of independent searches (default 16)
#   --max_zealous  zealous-update attempts per search (default 10)
#   --k_d          cluster ceiling, 0 = no ceiling (default 0 = largest seen)
#   --seed         RNG seed (default 1)
#   --header       `true` to skip the first row of the input (default false)
#   --truth_file   optional CSV of true labels, reported as an ARI

using DelimitedFiles
using Printf
using Random
using BayesianSALSO

const OPTIONS = Dict(
    "input_file" => "",
    "output_file" => "",
    "loss" => "binder",
    "loss_a" => "1.0",
    "n_runs" => "16",
    "max_zealous" => "10",
    "k_d" => "0",
    "seed" => "1",
    "header" => "false",
    "truth_file" => "",
)

"""Parse `--key=value` / `--key value` options into a copy of `OPTIONS`."""
function parse_cli(args)
    opts = copy(OPTIONS)
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("unexpected argument `$arg` (options start with --)")
        body = arg[3:end]
        if occursin('=', body)
            key, value = split(body, '=', limit = 2)
        else
            key = body
            i += 1
            i <= length(args) || error("option --$key needs a value")
            value = args[i]
        end
        haskey(opts, key) ||
            error("unknown option --$key; known options: " * join(sort!(collect(keys(opts))), ", "))
        opts[key] = value
        i += 1
    end
    return opts
end

"""
    read_labels(path, skip_header) -> Matrix{Int}

Read a CSV whose cells must all be integers, optionally skipping a header row.
Anything else — `1.0`, `1.5`, strings, a header that was not skipped — is the
user's mistake and is reported as such instead of being coerced.
"""
function read_labels(path::AbstractString, skip_header::Bool)
    isfile(path) || error("file not found: $path")
    try
        return readdlm(path, ',', Int; skipstart = skip_header ? 1 : 0)
    catch err
        error(
            "could not read `$path` as integer cluster labels: " *
            sprint(showerror, err) *
            "\nEvery cell must be an integer label.  Typical causes: the file holds " *
            "probabilities or other non-integer values, the labels were written as " *
            "floats (`1.0` instead of `1`), or it has a header row — pass " *
            "`--header=true` if so.",
        )
    end
end

function main(args)
    opts = parse_cli(args)
    isempty(opts["input_file"]) && error("--input_file is required")
    isempty(opts["output_file"]) && error("--output_file is required")

    loss_name = lowercase(opts["loss"])
    loss_name in ("binder", "vi") || error("--loss must be binder or vi, got `$(opts["loss"])`")
    a = parse(Float64, opts["loss_a"])
    loss = loss_name == "binder" ? BinderLoss(a) : VILoss(a)

    # the file is n × B (rows are observations, columns are draws); BayesianSALSO wants
    # the B × n transpose, one row per draw
    labels_by_item =
        read_labels(opts["input_file"], lowercase(opts["header"]) in ("true", "yes", "1"))
    n, n_draws = size(labels_by_item)
    draws = permutedims(labels_by_item)
    @printf("input    %s: n = %d items, %d draws\n", opts["input_file"], n, n_draws)
    @printf(
        "salso    loss = %s (a = %s), n_runs = %d, max_zealous = %d, k_d = %s\n",
        loss_name,
        opts["loss_a"],
        parse(Int, opts["n_runs"]),
        parse(Int, opts["max_zealous"]),
        opts["k_d"] == "0" ? "auto" : opts["k_d"],
    )
    flush(stdout)

    t0 = time()
    res = salso(
        draws;
        loss = loss,
        n_runs = parse(Int, opts["n_runs"]),
        max_zealous_attempts = parse(Int, opts["max_zealous"]),
        max_n_clusters = parse(Int, opts["k_d"]),
        rng = MersenneTwister(parse(Int, opts["seed"])),
    )
    elapsed = time() - t0
    @printf(
        "result   K = %d, E[loss] = %.4f, scans = %d, zealous = %d/%d, %.2f s\n",
        res.n_clusters,
        res.expected_loss,
        res.n_scans,
        res.n_zealous_accepts,
        res.n_zealous_attempts,
        elapsed,
    )

    if !isempty(opts["truth_file"])
        truth = vec(read_labels(opts["truth_file"], false))
        @printf("quality  ARI vs %s = %.4f\n", opts["truth_file"], ari(truth, res.partition))
    end

    open(opts["output_file"], "w") do io
        println(io, "salso_partition")
        for label in res.partition
            println(io, label)
        end
    end
    @printf("wrote    %s\n", opts["output_file"])
    return nothing
end

# run only when executed as a script (`PROGRAM_FILE` is empty when the file is
# included, and the parenthesised macro is needed: `@__FILE__ && …` does not
# parse as a comparison on Julia 1.13)
if PROGRAM_FILE != "" && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
