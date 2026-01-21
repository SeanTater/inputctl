from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="REFLEX_TRAIN_",
    )

    data_dir: str
    sprite_threshold: float = 0.85
    base_dir: str = "/usr/share/games/supertux2/images"
    overwrite: bool = False

    # Event/episode detection for RL
    detect_events: bool = True
    death_threshold: float = 0.75
    event_stride: int = 1
    attack_threshold: float = 0.8
    win_key: str = "KEY_BACKSLASH"
    win_key_min_presses: int = 1
    win_key_window_s: float = 2.0
    win_key_cooldown_s: float = 30.0
    respawn_gap: int = 30
    blank_frame_mean_threshold: float = 5.0
    blank_frame_std_threshold: float = 3.0
    attack_min_gap: int = 5
    death_min_gap: int = 30

    # Rewards
    gamma: float = 0.99
    death_reward: float = -1.0
    win_reward: float = 1.0
    attack_reward: float = 0.1  # Bonus for attacking/killing enemies
    survival_bonus: float = 0.0
