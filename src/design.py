import argparse
from collections.abc import Callable

import numpy as np

from .features import AMINO_ACIDS


def single_mutants(sequences: list[str]) -> list[str]:
    mutants = set()

    for sequence in sequences:
        for position, original in enumerate(sequence):
            for residue in AMINO_ACIDS:
                if residue != original:
                    mutants.add(sequence[:position] + residue + sequence[position + 1:])

    return sorted(mutants)


def beam_search(
    score: Callable[[list[str]], np.ndarray],
    *,
    starts: int,
    beam_width: int,
    query_budget: int,
    random_seed: int,
) -> tuple[list[dict], dict]:
    if min(starts, beam_width, query_budget) < 1 or starts > query_budget:
        raise ValueError("Positive search counts and starts <= query_budget are required")

    rng = np.random.default_rng(random_seed)
    initial = sorted({
        "".join(row)
        for row in rng.choice(list(AMINO_ACIDS), size = (starts, 10))
    })
    initial_scores = np.asarray(score(initial))
    if initial_scores.shape != (len(initial),) or not np.isfinite(initial_scores).all():
        raise ValueError("Invalid initial scores")

    archive     = dict(zip(initial, map(float, initial_scores), strict = True))
    rounds      = 0
    stop_reason = "query_budget"

    while True:
        beam       = sorted(archive, key = lambda s: (-archive[s], s))[:beam_width]
        candidates = [s for s in single_mutants(beam) if s not in archive]
        if not candidates:
            stop_reason = "no_unvisited_neighbors"
            break
        if len(archive) + len(candidates) > query_budget:
            break

        predictions = np.asarray(score(candidates))
        if predictions.shape != (len(candidates),) or not np.isfinite(predictions).all():
            raise ValueError("Invalid candidate scores")
        archive.update(zip(candidates, map(float, predictions), strict = True))
        rounds += 1

        new_beam = sorted(archive, key = lambda s: (-archive[s], s))[:beam_width]
        if new_beam == beam:
            stop_reason = "beam_converged"
            break

    ordered = sorted(archive, key = lambda s: (-archive[s], s))
    rows    = [{"sequence": s, "score": archive[s]} for s in ordered]
    audit = {
        "unique_queries": len(archive),
        "rounds": rounds,
        "stop_reason": stop_reason,
        "initial_sequences": initial,
    }
    return rows, audit


def design_objective(args: argparse.Namespace) -> tuple[list[str], np.ndarray]:
    if args.objective == "affinity":
        if args.off_target or args.second_target:
            raise ValueError("affinity uses --target alone")
        return [args.target], np.asarray([1.])

    if args.objective == "margin":
        if not args.off_target or args.second_target:
            raise ValueError("margin requires --target and --off-target")
        names, coefficients = [args.target, args.off_target], [1., -1.]
    else:
        if not args.second_target or args.off_target:
            raise ValueError("mean requires --target and --second-target")
        names, coefficients = [args.target, args.second_target], [.5, .5]

    if names[0] == names[1]:
        raise ValueError("Specify two distinct PDZ identifiers")
    return names, np.asarray(coefficients)
