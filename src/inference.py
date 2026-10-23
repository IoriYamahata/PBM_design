import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .features import encode_pbm, load_blosum, prepare_pdz_features
from .model import DualEncoder


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


class Predictor:
    def __init__(
        self,
        config: dict,
        root: Path,
        pdz_features: np.ndarray,
        pdz_ids: list[str],
        device: torch.device,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        self.pdz_ids    = pdz_ids
        self.device     = device
        self.batch_size = batch_size

        blosum_path = root / config["reference"]["blosum62"]
        self.blosum = load_blosum(blosum_path)

        self.models: list[DualEncoder] = []
        self.anchors: list[torch.Tensor] = []
        features = torch.as_tensor(pdz_features, dtype = torch.float32, device = device)

        for relative in config["weights"]:
            path  = root / relative
            model = DualEncoder().to(device)
            model.load_state_dict(
                torch.load(path, map_location = device, weights_only = True),
                strict = True,
            )
            model.eval().requires_grad_(False)

            with torch.inference_mode():
                anchors = model.encode_pdz(features)

            self.models.append(model)
            self.anchors.append(anchors)

    @torch.inference_mode()
    def predict_per_model(self, sequences: list[str]) -> np.ndarray:
        encoded = encode_pbm(sequences, self.blosum)
        chunks  = []

        for start in range(0, len(sequences), self.batch_size):
            features = torch.from_numpy(encoded[start:start + self.batch_size]).to(self.device)
            predictions = [
                (
                    model.global_bias
                    + model.affinity_scale * (model.encode_pbm(features) @ anchors.T)
                ).cpu().numpy()
                for model, anchors in zip(self.models, self.anchors, strict = True)
            ]
            chunks.append(np.stack(predictions))

        result = np.concatenate(chunks, axis = 1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite predictions")

        return result

    def predict(self, sequences: list[str]) -> np.ndarray:
        return self.predict_per_model(sequences).mean(axis = 0)

    @torch.inference_mode()
    def score_direction(self, sequences: list[str], coefficients: np.ndarray) -> np.ndarray:
        encoded = encode_pbm(sequences, self.blosum)
        directions, norms = [], []

        for anchors in self.anchors:
            vector = coefficients @ anchors.cpu().double().numpy()
            norm   = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm < 1e-8:
                raise ValueError("These anchors do not define a unique design direction")

            directions.append(vector / norm)
            norms.append(norm)

        chunks = []
        for start in range(0, len(sequences), self.batch_size):
            features = torch.from_numpy(encoded[start:start + self.batch_size]).to(self.device)
            scores   = []

            for model, direction, norm in zip(self.models, directions, norms, strict = True):
                latent = model.encode_pbm(features).cpu().double().numpy()
                values = (latent @ direction) * float(model.affinity_scale) * norm
                values += float(model.global_bias) * float(coefficients.sum())
                scores.append(values)

            chunks.append(np.stack(scores).mean(axis = 0))

        return np.concatenate(chunks)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pdz",
        type     = Path,
        required = True,
        help     = "PDZ domain sequences in FASTA format",
    )
    parser.add_argument(
        "--output",
        type     = Path,
        required = True,
        help     = "New output directory; never overwritten",
    )
    parser.add_argument("--config", type = Path, default = ROOT / "config.yaml")
    parser.add_argument("--cache-dir", type = Path, default = ROOT / ".cache")
    parser.add_argument("--device", choices = ["cpu", "cuda"], default = "cpu")
    parser.add_argument("--mafft", help = "MAFFT executable, version 7.505")
    parser.add_argument("--offline", action = "store_true", help = "Use existing local Hugging Face files only")
    parser.add_argument("--batch-size", type = int)
    parser.add_argument("--threads", type = int, default = 4, help = "PyTorch CPU threads")


def make_predictor(args: argparse.Namespace, records: dict[str, str], config: dict) -> tuple[Predictor, dict]:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    root = args.config.resolve().parent
    features, positions = prepare_pdz_features(
        records,
        config,
        root,
        device,
        args.cache_dir,
        offline = args.offline,
        mafft   = args.mafft,
    )
    predictor = Predictor(
        config,
        root,
        features,
        list(records),
        device,
        args.batch_size if args.batch_size is not None else config["inference"]["batch_size"],
    )

    metadata = {
        "config": config,
        "pdz_sequences": records,
        "site_positions_1based_gap_zero": (positions + 1).tolist(),
        "device": str(device),
        "torch_version": torch.__version__,
        "batch_size": predictor.batch_size,
        "aggregation": "arithmetic mean of three calibrated scalar model predictions",
    }
    return predictor, metadata


def save_output(directory: Path, filename: str, rows: list[dict], metadata: dict) -> None:
    directory.mkdir(parents = True, exist_ok = False)
    with (directory / filename).open("w", newline = "", encoding = "utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (directory / "run.json").write_text(json.dumps(metadata, indent = 2) + "\n", encoding = "utf-8")
