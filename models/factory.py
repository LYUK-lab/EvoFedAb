from models.affinity_base import NoESMAffinityModel
from models.peft.feature_space import (
    AntibinderAffinityModel,
    DuaDeepAffinityModel,
    MLPAffinityModel,
    MintAffinityModel,
)

ENCODERS = ("mlp", "duadeep", "mint", "antibinder")


def build_model(model_cfg: dict, embedding_dim: int | None = None):
    if str(model_cfg.get("esm_path", "")).strip().lower() == "no-esm":
        return NoESMAffinityModel(model_cfg)
    encoder = str(model_cfg["head_type"]).lower()
    classes = {
        "mlp": MLPAffinityModel,
        "duadeep": DuaDeepAffinityModel,
        "mint": MintAffinityModel,
        "antibinder": AntibinderAffinityModel,
    }
    if encoder not in classes:
        raise ValueError(f"Unsupported encoder {encoder!r}; choose from {ENCODERS}.")
    if not embedding_dim:
        raise ValueError("Missing embedding_dim; prepare ESM embeddings first.")
    params = model_cfg["peft_implementation"]["feature_space"]["feature_space_adaptor"][
        encoder
    ]
    return classes[encoder](
        embedding_dim=int(embedding_dim),
        hidden_dim=params.get("hidden_dim"),
        dropout=float(params["dropout"]),
        transformer_layers=int(params.get("transformer_layers", 1)),
        transformer_heads=int(params.get("transformer_heads", 4)),
    )
