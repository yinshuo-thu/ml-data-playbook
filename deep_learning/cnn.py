"""离线 8x8 digits：CNN、NCHW 形状、CrossEntropyLoss。独立留出集。"""

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    x, y = load_digits(return_X_y=True)
    x = torch.tensor(x.reshape(-1, 1, 8, 8) / 16, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    a, b = train_test_split(
        list(range(len(y))), test_size=0.2, stratify=y, random_state=42
    )
    loader = DataLoader(TensorDataset(x[a], y[a]), batch_size=64, shuffle=True)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 10),
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        print("holdout accuracy", (model(x[b]).argmax(1) == y[b]).float().mean().item())
    # 输入 (N,C,H,W)，输出 (N,10) 未经 softmax 的 logits，标签为 long (N,)。
    # 若据此反复选 epochs，需再留测试集；完整早停方案参见 mlp.py。


if __name__ == "__main__":
    main()
