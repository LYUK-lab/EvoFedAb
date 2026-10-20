from __future__ import annotations
from typing import Optional, Sequence, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from modules import GaussianAffinityHead, build_feature_space_adaptor


def _normalize_seq_list(seqs: Sequence[str] | str) -> list[str]:
    if isinstance(seqs, str):
        return [seqs]
    if torch.is_tensor(seqs):
        raise TypeError("no-esm baseline expects raw sequences, not embedding tensors.")
    if isinstance(seqs, (list, tuple)) and seqs and torch.is_tensor(seqs[0]):
        raise TypeError("no-esm baseline expects raw sequences, not embedding tensors.")
    return [str(s) for s in seqs]


def _resolve_adaptor_config(model_cfg: dict) -> tuple[str, dict]:
    head_type = str(model_cfg.get("head_type", "")).strip().lower()
    if not head_type:
        raise ValueError("model.head_type is required.")
    peft_impl = model_cfg.get("peft_implementation")
    if not isinstance(peft_impl, dict):
        raise ValueError("model.peft_implementation must be a dict.")
    space = "feature"
    space_key = f"{space}_space"
    space_cfg = peft_impl.get(space_key)
    if not isinstance(space_cfg, dict):
        raise ValueError(f"model.peft_implementation.{space_key} must be a dict.")
    adaptor_cfg = space_cfg.get("feature_space_adaptor")
    if not isinstance(adaptor_cfg, dict):
        raise ValueError(
            f"model.peft_implementation.{space_key}.feature_space_adaptor must be a dict."
        )
    impl_cfg = adaptor_cfg.get(head_type)
    if not isinstance(impl_cfg, dict):
        raise ValueError(
            f"model.peft_implementation.{space_key}.feature_space_adaptor.{head_type} is required and must be a dict for --feature-space-adaptor {head_type!r}."
        )
    return (head_type, impl_cfg)


class SimpleProteinTokenizer:

    def __init__(self):
        tokens = [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
            "<null_1>",
            "<mask>",
        ]
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]
        self.cls_id = self.vocab["<cls>"]
        self.eos_id = self.vocab["<eos>"]
        self.mask_id = self.vocab["<mask>"]
        self.vocab_size = len(self.vocab)

    def encode(self, seq: str, add_special_tokens: bool = True) -> list[int]:
        seq = (seq or "").upper()
        ids = [self.vocab.get(ch, self.unk_id) for ch in seq]
        if add_special_tokens:
            ids = [self.cls_id, *ids, self.eos_id]
        return ids

    def __call__(
        self, sequences: Sequence[str], add_special_tokens: bool = True
    ) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        ids_list = [
            self.encode(seq, add_special_tokens=add_special_tokens) for seq in sequences
        ]
        max_len = max((len(ids) for ids in ids_list), default=0)
        input_ids = torch.full((len(ids_list), max_len), self.pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            if ids:
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask = input_ids.ne(self.pad_id).to(torch.long)
        return (
            {"input_ids": input_ids, "attention_mask": attention_mask},
            attention_mask,
        )


class BaseAffinityModel(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        encoder_type: str = "mlp",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        super().__init__()
        if not embedding_dim:
            raise ValueError("embedding_dim must be set for ESM embeddings.")
        base_dim = int(embedding_dim)
        if hidden_dim is None:
            hidden_dim = base_dim
        elif hidden_dim != base_dim:
            raise ValueError("hidden_dim must match embedding_dim to avoid adapters.")
        self.seq_encoder, self._pair_encoder = build_feature_space_adaptor(
            encoder_type=encoder_type,
            input_dim=base_dim,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            transformer_layers=int(transformer_layers),
            transformer_heads=int(transformer_heads),
        )
        fused_dim = int(hidden_dim) * 2
        self.head = GaussianAffinityHead(
            fused_dim, int(hidden_dim), dropout=float(dropout)
        )

    def _prepare_token_repr(self, embeddings) -> torch.Tensor:
        if torch.is_tensor(embeddings):
            tok_repr = embeddings
        elif (
            isinstance(embeddings, (list, tuple))
            and embeddings
            and torch.is_tensor(embeddings[0])
        ):
            if len(embeddings) == 1:
                tok_repr = embeddings[0].unsqueeze(0)
            else:
                tok_repr = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        else:
            raise TypeError(
                "Expected precomputed ESM embeddings (Tensor or list of Tensors)."
            )
        device = next(self.seq_encoder.parameters()).device
        return tok_repr.to(device)

    def _encode_sequence(self, sequences) -> torch.Tensor:
        tok_repr = self._prepare_token_repr(sequences)
        _, pooled = self.seq_encoder(tok_repr)
        return pooled

    def _encode_pair(self, ab_seqs, ag_seqs) -> tuple[torch.Tensor, torch.Tensor]:
        ab_repr = self._prepare_token_repr(ab_seqs)
        ag_repr = self._prepare_token_repr(ag_seqs)
        _, _, ab_vec, ag_vec = self.seq_encoder(ab_repr, ag_repr)
        return (ab_vec, ag_vec)

    def forward(
        self,
        ab_seqs: Sequence[torch.Tensor] | torch.Tensor,
        ag_seqs: Optional[Sequence[torch.Tensor] | torch.Tensor] = None,
    ) -> dict:
        if ag_seqs is None:
            if torch.is_tensor(ab_seqs):
                ag_seqs = torch.zeros_like(ab_seqs)
            elif (
                isinstance(ab_seqs, (list, tuple))
                and ab_seqs
                and torch.is_tensor(ab_seqs[0])
            ):
                ag_seqs = [torch.zeros_like(x) for x in ab_seqs]
            else:
                raise TypeError(
                    "Expected precomputed ESM embeddings for ab_seqs/ag_seqs."
                )
        if self._pair_encoder:
            ab_vec, ag_vec = self._encode_pair(ab_seqs, ag_seqs)
        else:
            ab_vec = self._encode_sequence(ab_seqs)
            ag_vec = self._encode_sequence(ag_seqs)
        return self.head(ab_vec, ag_vec)


class NoESMAffinityModel(nn.Module):

    def __init__(self, model_cfg: dict):
        super().__init__()
        model_type, impl_cfg = _resolve_adaptor_config(model_cfg)
        embedding_dim = (
            impl_cfg.get("embedding_dim") or model_cfg.get("embedding_dim") or 320
        )
        hidden_dim = impl_cfg.get("hidden_dim")
        if hidden_dim is None:
            hidden_dim = embedding_dim
        dropout = float(impl_cfg.get("dropout", 0.1))
        transformer_layers_default = 2
        self.tokenizer = SimpleProteinTokenizer()
        self.embedding = nn.Embedding(
            self.tokenizer.vocab_size,
            int(embedding_dim),
            padding_idx=self.tokenizer.pad_id,
        )
        self.seq_encoder, self._pair_encoder = build_feature_space_adaptor(
            encoder_type=str(model_type),
            input_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            dropout=dropout,
            transformer_layers=int(
                impl_cfg.get("transformer_layers", transformer_layers_default)
            ),
            transformer_heads=int(impl_cfg.get("transformer_heads", 4)),
        )
        fused_dim = int(hidden_dim) * 2
        self.head = GaussianAffinityHead(fused_dim, int(hidden_dim), dropout=dropout)

    def _encode_sequence(self, sequences: Sequence[str]) -> torch.Tensor:
        inputs, _ = self.tokenizer(sequences)
        device = next(self.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        token_repr = self.embedding(input_ids)
        _, pooled = self.seq_encoder(token_repr)
        return pooled

    def _encode_pair(
        self, ab_seqs: Sequence[str], ag_seqs: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_ab, _ = self.tokenizer(ab_seqs)
        inputs_ag, _ = self.tokenizer(ag_seqs)
        device = next(self.parameters()).device
        ab_ids = inputs_ab["input_ids"].to(device)
        ag_ids = inputs_ag["input_ids"].to(device)
        ab_repr = self.embedding(ab_ids)
        ag_repr = self.embedding(ag_ids)
        _, _, ab_vec, ag_vec = self.seq_encoder(ab_repr, ag_repr)
        return (ab_vec, ag_vec)

    def forward(
        self, ab_seqs: Sequence[str] | str, ag_seqs: Sequence[str] | str | None = None
    ) -> dict:
        ab_list = _normalize_seq_list(ab_seqs)
        if ag_seqs is None:
            ag_list = ["" for _ in range(len(ab_list))]
        else:
            ag_list = _normalize_seq_list(ag_seqs)
            if len(ag_list) != len(ab_list):
                raise ValueError("ab_seqs and ag_seqs must have the same batch size.")
        if self._pair_encoder:
            ab_vec, ag_vec = self._encode_pair(ab_list, ag_list)
        else:
            ab_vec = self._encode_sequence(ab_list)
            ag_vec = self._encode_sequence(ag_list)
        return self.head(ab_vec, ag_vec)
