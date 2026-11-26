from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticPaperConfig:
    input_dim: int = 1
    training_tasks: int = 20
    held_out_tasks: int = 5
    diffusion_steps: int = 50
    diffusion_schedule: str = 'cosine'
    task_embedding_dimensions: tuple[int, ...] = (4, 8, 16)
    context_sizes: tuple[int, ...] = (0, 5, 10, 25)
    batch_size: int = 8
    learning_rate: float = 1e-4
    maximum_training_steps: int = 10_000
    predictive_samples: int = 100
    conditioning: str = 'RePaint'
    training_observations_per_function: int = 50
    evaluation_points_per_held_out_task: int = 100
