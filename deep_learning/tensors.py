"""PyTorch tensor、autograd、形状、设备；CPU 也可运行。"""

import torch


def main():
    torch.manual_seed(42)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss = (x * x).mean()
    loss.backward()
    print("loss", loss.item(), "gradient", x.grad)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("available device", device, "shape", x.shape)
    print("detached numpy", x.detach().cpu().numpy())


if __name__ == "__main__":
    main()
