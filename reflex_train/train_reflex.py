from reflex_train.training.config import TrainConfig
from reflex_train.training.train import train


def main():
    cfg = TrainConfig()
    train(cfg)


if __name__ == "__main__":
    main()
