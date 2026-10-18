"""GRU 预测已知窗口之后的一个点；按原始时间划分，隔离边界窗口。"""

import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    series = np.sin(np.arange(600) / 10) + rng.normal(0, 0.05, 600)
    split = 450
    window = 20
    mean = series[:split].mean()
    std = series[:split].std()
    z = (series - mean) / std

    def windows(start, end):
        indices = range(start + window, end)
        return (
            torch.tensor(
                np.array([z[i - window : i] for i in indices]), dtype=torch.float32
            ).unsqueeze(-1),
            torch.tensor(z[start + window : end], dtype=torch.float32),
        )

    xt, yt = windows(0, split)
    xv, yv = windows(split, len(z))
    loader = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True)
    model = Forecaster()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        pred = model(xv)
        print("original-scale RMSE", torch.mean((pred - yv) ** 2).sqrt().item() * std)
        print(
            "last-value baseline RMSE",
            torch.mean((xv[:, -1, 0] - yv) ** 2).sqrt().item() * std,
        )
    # shape=(batch,time,features)。这里只评估一步滚动预测，过去真实值逐时可用。


if __name__ == "__main__":
    main()
