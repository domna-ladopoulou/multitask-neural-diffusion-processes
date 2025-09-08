from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 500


@dataclass
class OptimizerConfig:
    num_warmup_epochs: int = 20
    num_decay_epochs: int = 200
    init_lr: float = 2e-5
    peak_lr: float = 1e-3
    end_lr: float = 1e-5
    ema_rate: float = 0.995  # 0.999


@dataclass
class NetworkConfig:
    n_layers: int = 4
    hidden_dim: int = 64
    num_heads: int = 8


@dataclass
class EvalConfig:
    batch_size: int = 1
    num_function_samples: int = 90  # 3 context sizes x 10
    n_samples_per_function: int = 20 #100 # 200
    float64: bool = False

    # if true, the context is used both in the encoder and then in repaint
    # if false, the context is used only in the encoder and there is NO REPAINT loop
    n_conditional_sample_with_repaint: bool = True # False # True

    @property
    def __steps_per_epoch(self) -> int:
        return self.num_function_samples // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.__steps_per_epoch

# new task encoder ocnfig
@dataclass
class TaskEncoderConfig:
    n_layers: int = 4 # 3
    hidden_dim: int = 64 # 16
    # num_heads: int = 8 # TODO: implement self attention
    embed_dim: int = 8 # 4 # 2


@dataclass
class Config:
    seed: int = 42
    test_turbine_id: int = 1
    dataset: str =  "wind_data_5d_test_turbine_1"
    dataset_name: str = "wind_data_5d_test_turbine_1"
    data_dir: str = "data"
    input_dim: int = 5
    batch_size: int = 32
    num_epochs: int = 250
    n_function_samples_per_epoch: int = 100 * 5
    loss_type: str = "l1"
    eval: EvalConfig = EvalConfig()
    network: NetworkConfig = NetworkConfig()
    schedule: DiffusionConfig = DiffusionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    task_encoder: TaskEncoderConfig = TaskEncoderConfig()

    restore: str = ""

    @property
    def steps_per_epoch(self) -> int:
        return self.n_function_samples_per_epoch // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.num_epochs


