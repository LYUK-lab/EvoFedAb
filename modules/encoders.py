# References
#
# The DuaDeep encoder is implemented with reference to DuaDeep-SeqAffinity https://arxiv.org/abs/2512.22007
#
# The MINT encoder is implemented with referene to https://github.com/VarunUllanat/mint.
#
# The AntiBinder encoder is implemented with reference to https://github.com/fdu-wangfeilab/AntiBinder.

from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch import nn
from utils.esm import load_or_download_esm2


class ESMTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
                or self.tokenizer.cls_token
                or self.tokenizer.mask_token
            )
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

    def __call__(
        self, sequences: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        seqs = [seq or "" for seq in sequences]
        inputs = self.tokenizer(
            seqs, padding=True, return_tensors="pt", add_special_tokens=True
        )
        inputs.pop("token_type_ids", None)
        if "input_ids" not in inputs:
            raise ValueError(
                "Tokenizer output must include input_ids."
            )
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = inputs["input_ids"].ne(self.pad_id)
        return (inputs, inputs["attention_mask"])


class _SimpleConv1D(nn.Module):

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv5 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_t = self.drop(self.act(self.conv5(x_t)))
        out = x_t.transpose(1, 2)
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return out


class DuaDeepEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        inner_dim: int | None = None,
        nhead: int = 4,
        dropout: float = 0.1,
        transformer_layers: int = 1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        if inner_dim is None:
            inner_dim = min(int(hidden_dim), 128)
        inner_dim = int(inner_dim)
        if inner_dim <= 0:
            raise ValueError("inner_dim must be positive.")
        if inner_dim % int(nhead) != 0:
            inner_dim = max(int(nhead), inner_dim // int(nhead) * int(nhead))
        self._hidden_dim = int(hidden_dim)
        self._inner_dim = int(inner_dim)
        self.down = (
            nn.Linear(input_dim, self._inner_dim)
            if input_dim != self._inner_dim
            else nn.Identity()
        )
        transformer_layers = int(transformer_layers)
        if transformer_layers <= 0:
            raise ValueError("transformer_layers must be >= 1.")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._inner_dim,
            nhead=nhead,
            dim_feedforward=max(1, int(self._inner_dim * 1.25)),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.cnn = _SimpleConv1D(self._inner_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(self._inner_dim * 2, self._hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse_ln = nn.LayerNorm(self._hidden_dim)

    @staticmethod
    def _infer_pad_mask(x: torch.Tensor) -> torch.Tensor | None:
        if x.numel() == 0:
            return None
        return x.abs().sum(dim=-1).eq(0)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1).to(x.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def _encode_stream(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.down(x)
        t = self.transformer(x, src_key_padding_mask=mask)
        if mask is not None:
            t = t.masked_fill(mask.unsqueeze(-1), 0.0)
        c = self.cnn(x, mask=mask)
        fused = self.fuse_ln(self.fuse(torch.cat([t, c], dim=-1)))
        if mask is not None:
            fused = fused.masked_fill(mask.unsqueeze(-1), 0.0)
        return fused

    def forward(
        self, ab_repr: torch.Tensor, ag_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ab_mask = self._infer_pad_mask(ab_repr)
        ag_mask = self._infer_pad_mask(ag_repr)
        ab_out = self._encode_stream(ab_repr, ab_mask)
        ag_out = self._encode_stream(ag_repr, ag_mask)
        ab_pool = self._masked_mean(ab_out, ab_mask)
        ag_pool = self._masked_mean(ag_out, ag_mask)
        return (ab_out, ag_out, ab_pool, ag_pool)


class MintEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        inner_dim: int | None = None,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        if inner_dim is None:
            inner_dim = min(int(hidden_dim), 128)
        inner_dim = int(inner_dim)
        if inner_dim <= 0:
            raise ValueError("inner_dim must be positive.")
        if inner_dim % int(nhead) != 0:
            inner_dim = max(int(nhead), inner_dim // int(nhead) * int(nhead))
        self._hidden_dim = int(hidden_dim)
        self._inner_dim = int(inner_dim)
        self.down = (
            nn.Linear(input_dim, self._inner_dim)
            if input_dim != self._inner_dim
            else nn.Identity()
        )
        self.up = (
            nn.Linear(self._inner_dim, self._hidden_dim)
            if self._inner_dim != self._hidden_dim
            else nn.Identity()
        )
        self.self_ln = nn.LayerNorm(self._inner_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self._inner_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_ln = nn.LayerNorm(self._inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self._inner_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fuse_norm = nn.LayerNorm(self._inner_dim)

    @staticmethod
    def _infer_pad_mask(x: torch.Tensor) -> torch.Tensor | None:
        if x.numel() == 0:
            return None
        return x.abs().sum(dim=-1).eq(0)

    def _self_block(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        h = self.self_ln(x)
        attn_out, _ = self.self_attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        return x + self.dropout(attn_out)

    def _cross_block(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.cross_ln(query)
        attn_out, _ = self.cross_attn(
            h,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout(attn_out)

    def forward(
        self, ab_repr: torch.Tensor, ag_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ab_mask = self._infer_pad_mask(ab_repr)
        ag_mask = self._infer_pad_mask(ag_repr)
        ab_repr = self.down(ab_repr)
        ag_repr = self.down(ag_repr)
        ab_self = self._self_block(ab_repr, ab_mask)
        ag_self = self._self_block(ag_repr, ag_mask)
        ab_cross = self._cross_block(ab_self, ag_self, ag_mask)
        ag_cross = self._cross_block(ag_self, ab_self, ab_mask)
        ab_out = self.fuse_norm(ab_self + ab_cross)
        ag_out = self.fuse_norm(ag_self + ag_cross)
        ab_out = self.up(ab_out)
        ag_out = self.up(ag_out)
        ab_pool = ab_out.mean(dim=1)
        ag_pool = ag_out.mean(dim=1)
        return (ab_out, ag_out, ab_pool, ag_pool)


class AntibinderEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        inner_dim: int | None = None,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        if inner_dim is None:
            inner_dim = min(int(hidden_dim), 128)
        inner_dim = int(inner_dim)
        if inner_dim <= 0:
            raise ValueError("inner_dim must be positive.")
        if inner_dim % int(nhead) != 0:
            inner_dim = max(int(nhead), inner_dim // int(nhead) * int(nhead))
        self._hidden_dim = int(hidden_dim)
        self._inner_dim = int(inner_dim)
        self.down = (
            nn.Linear(input_dim, self._inner_dim)
            if input_dim != self._inner_dim
            else nn.Identity()
        )
        self.up = (
            nn.Linear(self._inner_dim, self._hidden_dim)
            if self._inner_dim != self._hidden_dim
            else nn.Identity()
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self._inner_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.ln_ab = nn.LayerNorm(self._inner_dim)
        self.ln_ag = nn.LayerNorm(self._inner_dim)
        self.out_ab = nn.Linear(self._inner_dim, self._inner_dim)
        self.out_ag = nn.Linear(self._inner_dim, self._inner_dim)

    @staticmethod
    def _infer_pad_mask(x: torch.Tensor) -> torch.Tensor | None:
        if x.numel() == 0:
            return None
        return x.abs().sum(dim=-1).eq(0)

    def forward(
        self, ab_repr: torch.Tensor, ag_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ab_mask = self._infer_pad_mask(ab_repr)
        ag_mask = self._infer_pad_mask(ag_repr)
        ab_repr = self.down(ab_repr)
        ag_repr = self.down(ag_repr)
        ab_attn, _ = self.cross_attn(
            ab_repr, ag_repr, ag_repr, key_padding_mask=ag_mask, need_weights=False
        )
        ag_attn, _ = self.cross_attn(
            ag_repr, ab_repr, ab_repr, key_padding_mask=ab_mask, need_weights=False
        )
        ab_out = self.out_ab(self.ln_ab(self.drop(ab_attn)))
        ag_out = self.out_ag(self.ln_ag(self.drop(ag_attn)))
        ab_out = self.up(ab_out)
        ag_out = self.up(ag_out)
        ab_pool = ab_out.mean(dim=1)
        ag_pool = ag_out.mean(dim=1)
        return (ab_out, ag_out, ab_pool, ag_pool)


class MLPEncoder(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(x)
        pooled = out.mean(dim=1)
        return (out, pooled)


class FrozenESMEncoder(nn.Module):

    def __init__(self, esm_path: str):
        super().__init__()
        self.encoder, tokenizer = load_or_download_esm2(esm_path)
        self.tokenizer = ESMTokenizer(tokenizer)
        self.hidden_dim = int(self.encoder.config.hidden_size)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def encode(self, sequences: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        self.encoder.eval()
        inputs, mask = self.tokenizer(sequences)
        device = next(self.encoder.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        mask = mask.to(device).unsqueeze(-1)
        tokens = self.encoder(**inputs).last_hidden_state * mask
        pooled = tokens.sum(1) / mask.sum(1).clamp_min(1)
        return (tokens, pooled)

    def forward(self, sequences: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(sequences)
