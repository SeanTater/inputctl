from pydantic_settings import BaseSettings, SettingsConfigDict


class LabelingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="REFLEX_TRAIN_",
    )

    data_dir: str
    sprite_scale: float = 0.5
    sprite_threshold: float = 0.85
    base_dir: str = "/usr/share/games/supertux2/images"
    overwrite: bool = False

    # Event/episode detection for RL
    detect_events: bool = True
    death_threshold: float = 0.75
    win_proximity_px: float = 96.0
    sparkle_threshold: float = 0.9
    win_min_frames: int = 3
    event_stride: int = 1
    attack_threshold: float = 0.8
    win_llm_gate: bool = False
    win_llm_sample_stride: int = 30
    win_llm_prompt: str = (
        "Is this the SuperTux level-complete or win screen? Reply YES or NO."
    )
    win_llm_timeout_s: float = 30.0
    win_llm_model: str = "qwen3-vl:4b"
    win_llm_url: str = "http://localhost:11434/api/generate"
    respawn_gap: int = 30
    blank_frame_mean_threshold: float = 5.0
    blank_frame_std_threshold: float = 3.0
    attack_min_gap: int = 5
    death_min_gap: int = 30
    win_min_gap: int = 30

    # Rewards
    gamma: float = 0.99
    death_reward: float = -1.0
    win_reward: float = 1.0
    attack_reward: float = 0.1  # Bonus for attacking/killing enemies
    survival_bonus: float = 0.0
