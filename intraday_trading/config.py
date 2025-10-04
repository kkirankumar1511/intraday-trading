from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    ticker: str = "AAPL"
    interval: str = "15m"
    history_days: int = 60
    lookback: int = 200
    horizon: int = 10
    test_size: float = 0.2
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    model_dir: Path = Path("models")
    scaler_path: Path = Path("models") / "feature_scaler.pkl"
    model_path: Path = Path("models") / "bilstm_model.pt"

    def ensure_directories(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
