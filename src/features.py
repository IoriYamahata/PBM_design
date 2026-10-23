import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path: Path, *, aligned: bool = False) -> dict[str, str]:
    records: dict[str, str] = {}
    identifier = None

    for line in path.read_text(encoding = "utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            header = line[1:].split()
            if not header or header[0] in records:
                raise ValueError(f"Missing or duplicate FASTA identifier in {path}")
            identifier = header[0]
            records[identifier] = ""
        elif identifier is None:
            raise ValueError(f"Sequence before FASTA header in {path}")
        else:
            records[identifier] += line.upper()

    allowed = set(AMINO_ACIDS + ("-" if aligned else ""))
    if not records or any(not s or set(s) - allowed for s in records.values()):
        raise ValueError(f"Expected nonempty canonical amino-acid sequences in {path}")
    return records


def load_blosum(path: Path) -> np.ndarray:
    lines = [
        line.split()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    alphabet = lines[0]
    rows     = {line[0]: np.asarray(line[1:], dtype = np.float32) for line in lines[1:]}
    columns  = [alphabet.index(residue) for residue in AMINO_ACIDS]

    return np.stack([rows[residue][columns] for residue in AMINO_ACIDS])


def encode_pbm(sequences: list[str], matrix: np.ndarray) -> np.ndarray:
    alphabet = set(AMINO_ACIDS)
    if not sequences or any(len(sequence) != 10 or set(sequence) - alphabet for sequence in sequences):
        raise ValueError("PBMs must be exactly ten canonical amino acids, N to C")

    lookup  = {residue: i for i, residue in enumerate(AMINO_ACIDS)}
    indices = [[lookup[residue] for residue in sequence] for sequence in sequences]

    return matrix[indices].astype(np.float32, copy = False)


def check_mafft(executable: str, version: str) -> None:
    result = subprocess.run(
        [executable, "--version"],
        capture_output = True,
        text           = True,
        check          = True,
        timeout        = 30,
    )
    reported = result.stdout + result.stderr
    if not re.search(r"(?<!\d)" + re.escape(version) + r"(?!\d)", reported):
        raise ValueError(f"MAFFT {version} is required")


def align_sequence(sequence: str, reference_path: Path, columns: list[int], executable: str) -> np.ndarray:
    reference = read_fasta(reference_path, aligned = True)

    with tempfile.TemporaryDirectory(prefix = "pdz-alignment-") as directory:
        query = Path(directory) / "query.fasta"
        query.write_text(f">QUERY\n{sequence}\n", encoding = "utf-8")

        command = [
            executable,
            "--thread", "1",
            "--add", str(query),
            "--keeplength",
            "--mapout",
            str(reference_path.resolve()),
        ]
        result = subprocess.run(command, capture_output = True, text = True, timeout = 120)
        if result.returncode:
            details = result.stderr.strip().splitlines()
            reason  = details[-1] if details else f"exit code {result.returncode}"
            raise RuntimeError(f"MAFFT failed: {reason}")

        aligned_path = Path(directory) / "aligned.fasta"
        aligned_path.write_text(result.stdout, encoding = "utf-8")
        aligned = read_fasta(aligned_path, aligned = True)
        if {name: seq for name, seq in aligned.items() if name != "QUERY"} != reference:
            raise ValueError("MAFFT changed the fixed reference alignment")

        mapping: dict[int, int] = {}
        for line in Path(str(query) + ".map").read_text().splitlines():
            if not line or line.startswith((">", "#")):
                continue

            residue, position, column = [field.strip() for field in line.split(",")]
            index = int(position) - 1
            if not 0 <= index < len(sequence) or sequence[index] != residue.upper():
                raise ValueError("MAFFT residue map disagrees with the input sequence")
            if column == "-":
                continue
            mapping[int(column)] = index

    positions = np.asarray([mapping.get(column, -1) for column in columns], dtype = np.int32)
    if np.all(positions == -1):
        raise ValueError("No binding sites mapped; check the PDZ domain input sequence")
    return positions


def extract_site_features(embeddings: np.ndarray, positions: np.ndarray) -> np.ndarray:
    features = np.zeros((17, 1152), dtype = np.float32)
    occupied = positions >= 0
    features[occupied] = embeddings[positions[occupied]]
    features /= np.maximum(np.linalg.norm(features, axis = 1, keepdims = True), 1e-12)

    return features.reshape(-1)


def embed_pdz(
    sequences: list[str],
    settings: dict,
    device: torch.device,
    *,
    offline: bool,
) -> list[np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    common    = {"revision": settings["revision"], "local_files_only": offline}
    tokenizer = AutoTokenizer.from_pretrained(settings["model_id"], **common)
    model = AutoModel.from_pretrained(
        settings["model_id"],
        dtype = getattr(torch, settings["dtype"]),
        **common,
    ).to(device).eval().requires_grad_(False)

    features = []
    with torch.inference_mode():
        for start in range(0, len(sequences), 16):
            batch = sequences[start:start + 16]
            tokens = tokenizer(
                batch,
                add_special_tokens         = True,
                padding                    = True,
                return_special_tokens_mask = True,
                return_tensors             = "pt",
            )
            mask = tokens["attention_mask"].bool() & ~tokens.pop("special_tokens_mask").bool()
            if mask.sum(dim = 1).tolist() != list(map(len, batch)):
                raise ValueError("ESMC tokenizer did not preserve one token per residue")

            hidden = model(**{key: value.to(device) for key, value in tokens.items()}).last_hidden_state
            for row, keep in zip(hidden, mask, strict = True):
                features.append(row[keep.to(device)].float().cpu().numpy())

    return features


def prepare_pdz_features(
    records: dict[str, str],
    config: dict,
    root: Path,
    device: torch.device,
    cache_dir: Path,
    *,
    offline: bool = False,
    mafft: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference  = config["reference"]
    alignment  = root / reference["alignment"]
    sites      = json.loads((root / reference["binding_sites"]).read_text())
    executable = mafft or config["mafft"]["executable"]
    check_mafft(executable, config["mafft"]["version"])

    sequences = list(records.values())
    cache_inputs = {
        "sequences": sequences,
        "columns": sites["columns_1based"],
        "alignment": alignment.read_text(),
        "esmc": config["esmc"],
        "mafft": config["mafft"]["version"],
        "torch": torch.__version__,
        "device": str(device),
    }
    metadata   = json.dumps(cache_inputs, sort_keys = True)
    cache_path = cache_dir / "pdz_features.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle = False) as cache:
            if str(cache["metadata"]) == metadata:
                return cache["features"], cache["positions"]

    positions = np.stack([
        align_sequence(seq, alignment, sites["columns_1based"], executable)
        for seq in sequences
    ])
    embeddings = embed_pdz(sequences, config["esmc"], device, offline = offline)
    features = np.stack([
        extract_site_features(vector, index)
        for vector, index in zip(embeddings, positions, strict = True)
    ])

    cache_dir.mkdir(parents = True, exist_ok = True)
    with tempfile.TemporaryDirectory(prefix = "cache-write-", dir = cache_dir) as temporary:
        staged = Path(temporary) / "features.npz"
        np.savez_compressed(
            staged,
            features  = features,
            positions = positions,
            metadata  = np.asarray(metadata),
        )
        staged.replace(cache_path)

    return features, positions
