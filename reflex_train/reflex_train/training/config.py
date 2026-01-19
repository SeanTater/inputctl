from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from reflex_train.data.intent import INTENTS


class TrainConfig(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
        env_prefix="REFLEX_TRAIN_",
    )

    data_dir: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    val_split: float = 0.1
    context_frames: int = 3
    action_horizon: int = 2
    goal_intent: str = "INFER"
    intent_weight: float = 0.25
    seed: int = 1337
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    key_threshold: float = 0.5
    require_intent_labels: bool = False

    # IQL settings (always enabled)
    gamma: float = 0.99  # Discount factor
    value_weight: float = 0.5  # Weight for value loss
    advantage_clip: float = 10.0  # Max advantage weight to prevent instability
    iql_expectile: float = 0.7  # Expectile for value regression
    iql_adv_temperature: float = 1.0  # Temperature for exp(advantage)

    # Inverse dynamics (always enabled)
    inv_dyn_weight: float = 0.2  # Weight for inverse dynamics loss
    inv_dyn_use_action_horizon: bool = False

    @field_validator("goal_intent")
    @classmethod
    def validate_goal_intent(cls, value):
        if value == "INFER":
            return value
        if value not in INTENTS:
            raise ValueError(f"goal_intent must be one of {INTENTS} or 'INFER'")
        return value
