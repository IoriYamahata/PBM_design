import torch
from torch import nn
from torch.nn import functional as F


class PDZEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_normalization = nn.LayerNorm(17 * 1152)
        self.dropout               = nn.Dropout(0.1)
        self.projection            = nn.Linear(17 * 1152, 128)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.feature_normalization(features)
        return self.projection(self.dropout(normalized))


class PBMEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_normalization = nn.LayerNorm(20)
        self.residue_projection    = nn.Linear(20, 16)
        self.position_embedding    = nn.Embedding(10, 16)

        self.transformer = nn.TransformerEncoderLayer(
            d_model         = 16,
            nhead           = 4,
            dim_feedforward = 64,
            dropout         = 0.1,
            activation      = "relu",
            batch_first     = True,
            norm_first      = False,
        )
        self.projection = nn.Linear(10 * 16, 128)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        positions  = torch.arange(10, device = features.device).unsqueeze(0)
        residues   = self.residue_projection(self.feature_normalization(features))
        contextual = self.transformer(residues + self.position_embedding(positions))

        return self.projection(contextual.flatten(start_dim = 1))


class DualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pdz_encoder        = PDZEncoder()
        self.pbm_encoder        = PBMEncoder()
        self.raw_affinity_scale = nn.Parameter(torch.zeros(()))
        self.global_bias        = nn.Parameter(torch.zeros(()))

    @property
    def affinity_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_affinity_scale)

    def encode_pdz(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pdz_encoder(features), p = 2, dim = 1)

    def encode_pbm(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pbm_encoder(features), p = 2, dim = 1)

    def forward(self, pbm_features: torch.Tensor, pdz_features: torch.Tensor) -> torch.Tensor:
        similarity = self.encode_pbm(pbm_features) @ self.encode_pdz(pdz_features).T
        return self.global_bias + self.affinity_scale * similarity
