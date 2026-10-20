# References
# The ESM-2 integration refers to Evolutionary Scale Modeling (ESM).
# https://github.com/facebookresearch/esm

from pathlib import Path
from typing import Tuple
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

__all__ = ["load_or_download_esm2", "load_esm_tokenizer"]
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "pretrained"


def _looks_like_hf_model_dir(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        or (path / "pytorch_model.bin").is_file()
        or (path / "model.safetensors").is_file()
    )


def _resolve_model_source(model_path: str | None) -> tuple[str, Path]:
    if not model_path:
        raise ValueError("esm_path is required; no default ESM model is configured.")
    path = Path(model_path).expanduser()
    if path.is_dir():
        if _looks_like_hf_model_dir(path):
            return (str(path), path)
        raise ValueError(
            f"esm_path directory {path} is missing config.json or model weights."
        )
    if path.is_file():
        if path.suffix == ".pt":
            raise ValueError(
                "Legacy .pt ESM checkpoints are not supported. Use a Hugging Face model id or cache dir."
            )
        if path.name == "config.json":
            return (str(path.parent), path.parent)
        if path.suffix in {".bin", ".safetensors"}:
            return (str(path.parent), path.parent)
        raise ValueError(
            f"Unsupported esm_path file: {path}. Use a Hugging Face model id or cache dir."
        )
    return (model_path, DEFAULT_CACHE_DIR)


def validate_esm_source(model_path: str) -> None:
    model_id, cache_dir = _resolve_model_source(model_path)
    path = Path(model_id)
    if not path.exists() and (not path.name.lower().startswith("esm")):
        raise ValueError("Only ESM models are supported; non-ESM PLMs are unsupported.")
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
    )
    if config.model_type != "esm":
        raise ValueError(f"Only ESM models are supported, got {config.model_type!r}.")


def load_or_download_esm2(
    model_path: str | None = None,
) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    model_id, cache_dir = _resolve_model_source(model_path)
    validate_esm_source(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=str(cache_dir)
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token or tokenizer.cls_token or tokenizer.mask_token
        )
    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_id, cache_dir=str(cache_dir)
    )
    encoder = mlm_model.base_model
    return (encoder, tokenizer)


def load_esm_tokenizer(model_path: str | None = None) -> PreTrainedTokenizerBase:
    model_id, cache_dir = _resolve_model_source(model_path)
    validate_esm_source(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token or tokenizer.cls_token or tokenizer.mask_token
        )
    return tokenizer
