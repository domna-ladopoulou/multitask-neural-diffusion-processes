from dataclasses import dataclass


@dataclass(frozen=True)
class Synthetic:
    input_dim: int = 1
    diffusion_steps: int = 50
    learning_rate: float = 1e-4
    batch_size: int = 8
    maximum_training_steps: int = 10_000
    task_embedding_dimensions: tuple = (4, 8, 16)
    context_sizes: tuple = (0, 5, 10, 25)
    prediction_draws: int = 100
    conditioning: str = 'RePaint'


@dataclass(frozen=True)
class SchoolsMain:
    input_dim: int = 1
    training_schools: int = 30
    validation_schools: int = 5
    test_schools: int = 6
    context_size: int = 20
    diffusion_steps: int = 50
    denoiser_layers: int = 4
    denoiser_hidden_dim: int = 64
    denoiser_attention_heads: int = 4
    task_embedding_dim: int = 16
    prediction_draws: int = 100


@dataclass(frozen=True)
class SchoolsLOSO:
    input_dim: int = 1
    training_schools_per_fold: int = 5
    test_schools_per_fold: int = 1
    folds: int = 6
    context_size: int = 20
    diffusion_steps: int = 50
    denoiser_layers: int = 4
    denoiser_hidden_dim: int = 64
    denoiser_attention_heads: int = 4
    task_embedding_dim: int = 4


@dataclass(frozen=True)
class NeuralProcessBaseline:
    input_dim: int = 1  # school data: standLRT -> normexam
    hidden_dim: int = 128
    latent_dim: int = 32
    anp_attention_heads: int = 8
    training_steps: int = 12_000
    learning_rate: float = 3e-4
    batch_size: int = 32
    kl_weight: float = 0.01
