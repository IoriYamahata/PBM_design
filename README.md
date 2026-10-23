# PDZ–PBM prediction and design

Predict binding affinity and design 10-residue PBM peptides with a dual encoder.

## Setup

Requires Python 3.12+, Git, and MAFFT 7.505. ESMC weights download on first use.

```bash
pip install -r requirements.txt
```

## Usage

Use FASTA files containing PDZ domain sequences and standard 10-residue PBMs.
Replace `PDZ_A` and `PDZ_B` with PDZ FASTA identifiers.

```bash
python predict.py --pdz pdz.fasta --pbm pbm.fasta --output outputs/prediction
python design.py --pdz pdz.fasta --target PDZ_A --objective affinity --output outputs/affinity
python design.py --pdz pdz.fasta --target PDZ_A --off-target PDZ_B --objective margin --output outputs/margin
python design.py --pdz pdz.fasta --target PDZ_A --second-target PDZ_B --objective mean --output outputs/mean
```

Design objectives maximize target pKd, target-minus-non-target pKd, or mean pKd
for two targets. Results are saved as CSV. Settings: `config.yaml`; options: `--help`.

Unpublished draft. Licensing and reference redistribution review pending.
