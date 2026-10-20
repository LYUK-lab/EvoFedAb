from torch import nn
from modules.encoders import AntibinderEncoder, DuaDeepEncoder, MLPEncoder, MintEncoder


def build_feature_space_adaptor(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    transformer_layers: int,
    transformer_heads: int,
) -> tuple[nn.Module, bool]:
    if encoder_type == "mlp":
        return (MLPEncoder(input_dim, hidden_dim, dropout), False)
    pair_encoders = {
        "mint": MintEncoder,
        "antibinder": AntibinderEncoder,
        "duadeep": DuaDeepEncoder,
    }
    if encoder_type not in pair_encoders:
        raise ValueError(f"Unsupported encoder: {encoder_type}.")
    kwargs = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        nhead=transformer_heads,
    )
    if encoder_type == "duadeep":
        kwargs["transformer_layers"] = transformer_layers
    return (pair_encoders[encoder_type](**kwargs), True)
