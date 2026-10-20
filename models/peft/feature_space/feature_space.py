from models.affinity_base import BaseAffinityModel


class MintAffinityModel(BaseAffinityModel):

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "mint"
        super().__init__(**kwargs)


class AntibinderAffinityModel(BaseAffinityModel):

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "antibinder"
        super().__init__(**kwargs)


class MLPAffinityModel(BaseAffinityModel):

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "mlp"
        super().__init__(**kwargs)


class DuaDeepAffinityModel(BaseAffinityModel):

    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "duadeep"
        super().__init__(**kwargs)
