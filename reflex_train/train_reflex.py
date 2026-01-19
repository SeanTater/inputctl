import sys
from pathlib import Path
from reflex_train.training.config import TrainConfig
from reflex_train.training.train import train


def main():
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        # Default to configs/default.toml in the reflex_train directory
        config_path = Path(__file__).parent / "configs" / "default.toml"

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    cfg = TrainConfig.from_toml(config_path)
    train(cfg)


if __name__ == "__main__":
    main()
