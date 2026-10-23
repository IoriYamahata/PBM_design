import argparse

from src.design import beam_search, design_objective
from src.features import read_fasta
from src.inference import add_shared_arguments, load_config, make_predictor, save_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Design canonical ten-residue PBMs")
    add_shared_arguments(parser)
    parser.add_argument("--objective", choices = ["affinity", "margin", "mean"], required = True)
    parser.add_argument("--target", required = True, help = "PDZ identifier in the input FASTA")
    parser.add_argument("--off-target")
    parser.add_argument("--second-target")

    for option in ["starts", "beam-width", "query-budget", "top-k", "random-seed"]:
        parser.add_argument(f"--{option}", type = int)
    args = parser.parse_args()

    config = load_config(args.config)
    settings = {
        key: getattr(args, key) if getattr(args, key) is not None else value
        for key, value in config["search"].items()
    }
    if settings["top_k"] < 1:
        raise ValueError("--top-k must be positive")

    names, coefficients = design_objective(args)
    pdzs = read_fasta(args.pdz)
    if not set(names).issubset(pdzs):
        raise ValueError("Every target identifier must appear in the input FASTA")
    selected = {name: pdzs[name] for name in names}

    predictor, metadata = make_predictor(args, selected, config)
    candidates, audit = beam_search(
        lambda sequences: predictor.score_direction(sequences, coefficients),
        starts       = settings["starts"],
        beam_width   = settings["beam_width"],
        query_budget = settings["query_budget"],
        random_seed  = settings["random_seed"],
    )
    candidates  = candidates[:settings["top_k"]]
    predictions = predictor.predict([row["sequence"] for row in candidates])

    rows = []
    for rank, (candidate, profile) in enumerate(zip(candidates, predictions, strict = True), start = 1):
        row = {
            "rank": rank,
            "sequence": candidate["sequence"],
            "objective_score": candidate["score"],
        }
        row.update({
            f"predicted_pkd:{name}": float(value)
            for name, value in zip(names, profile, strict = True)
        })
        rows.append(row)

    metadata.update(
        objective           = args.objective,
        coefficients        = coefficients.tolist(),
        search              = settings,
        search_audit        = audit,
        returned_candidates = len(rows),
    )
    save_output(args.output, "candidates.csv", rows, metadata)
    print(f"Saved {len(rows)} candidates ({audit['unique_queries']} unique evaluations) to {args.output}")
