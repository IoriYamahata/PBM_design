import argparse
from pathlib import Path

from src.features import read_fasta
from src.inference import add_shared_arguments, load_config, make_predictor, save_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Predict pKd for all input PDZ–PBM pairs")
    add_shared_arguments(parser)
    parser.add_argument(
        "--pbm",
        type     = Path,
        required = True,
        help     = "Ten-residue PBM sequences in FASTA format",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pdzs   = read_fasta(args.pdz)
    pbms   = read_fasta(args.pbm)
    if any(len(sequence) != 10 for sequence in pbms.values()):
        raise ValueError("PBMs must be exactly ten canonical amino acids, N to C")

    predictor, metadata = make_predictor(args, pdzs, config)
    scores = predictor.predict(list(pbms.values()))

    rows = [
        {
            "pbm_id": name,
            "sequence": sequence,
            "pdz_id": pdz,
            "predicted_pkd": float(scores[i, j]),
        }
        for i, (name, sequence) in enumerate(pbms.items())
        for j, pdz in enumerate(pdzs)
    ]
    save_output(args.output, "predictions.csv", rows, metadata)
    print(f"Saved {len(rows)} pair predictions to {args.output}")
