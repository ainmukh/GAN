from dataclasses import dataclass

@dataclass
class Arguments:
    latent_dim: int = 16
    lambda_sty: int = 1
    lambda_ds: int = 1
    lambda_cyc: int = 1
    lambda_reg: int = 1
