"""离线 digits 分类：DataLoader、MLP、早停、checkpoint 与独立测试。"""

import argparse
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_model():
    return nn.Sequential(
        nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 10)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    x, y = load_digits(return_X_y=True)
    xdev, xtest, ydev, ytest = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    xtrain, xval, ytrain, yval = train_test_split(
        xdev, ydev, test_size=0.2, stratify=ydev, random_state=42
    )
    scaler = StandardScaler().fit(xtrain)

    def dataset(a, b):
        return TensorDataset(
            torch.tensor(scaler.transform(a), dtype=torch.float32),
            torch.tensor(b, dtype=torch.long),
        )

    loader = DataLoader(
        dataset(xtrain, ytrain),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )
    val_x, val_y = dataset(xval, yval).tensors
    device = torch.device(args.device)
    model = make_model().to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            count += len(xb)
        model.eval()
        with torch.inference_mode():
            val_loss = criterion(model(val_x), val_y).item()
        history.append(
            {"epoch": epoch + 1, "train_loss": loss_sum / count, "val_loss": val_loss}
        )
        if val_loss < best - 1e-5:
            best = val_loss
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best_state)
    model.eval()
    test_x, test_y = dataset(xtest, ytest).tensors
    with torch.inference_mode():
        logits = model(test_x.to(device)).cpu()
        accuracy = (logits.argmax(1) == test_y).float().mean().item()
    out = Path("outputs/mlp")
    out.mkdir(parents=True, exist_ok=True)
    # CPU tensors + weights_only-compatible primitives; no custom pickled classes.
    checkpoint = {
        "model": {k: v.cpu() for k, v in best_state.items()},
        "mean": torch.tensor(scaler.mean_),
        "scale": torch.tensor(scaler.scale_),
    }
    torch.save(checkpoint, out / "checkpoint.pt")
    loaded = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
    restored = make_model()
    restored.load_state_dict(loaded["model"])
    restored.eval()
    # Rebuild preprocessing from saved parameters, starting with raw test input.
    restored_test_x = (
        (torch.tensor(xtest, dtype=torch.float64) - loaded["mean"]) / loaded["scale"]
    ).float()
    with torch.inference_mode():
        restored_logits = restored(restored_test_x)
    assert torch.allclose(logits, restored_logits, atol=1e-4, rtol=1e-4)
    report = {
        "test_accuracy": accuracy,
        "best_val_loss": best,
        "epochs_ran": len(history),
        "train_rows": len(xtrain),
        "validation_rows": len(xval),
        "test_rows": len(xtest),
        "device": str(device),
        "torch": torch.__version__,
        "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "history"})


if __name__ == "__main__":
    main()
