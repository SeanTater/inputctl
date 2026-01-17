from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="REFLEX_TRAIN_",
    )

    data_dir: str
    labeler: str = "supertux"
    intent_horizon: int = 10
    sprite_scale: float = 0.5
    sprite_threshold: float = 0.85
    sprite_proximity: float = 96.0
    base_dir: str = "/usr/share/games/supertux2/images"
    overwrite: bool = False
