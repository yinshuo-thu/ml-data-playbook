# 安装与环境

在仓库根目录执行命令。不同项目使用独立虚拟环境，避免依赖冲突。

系统自带 `python3` 可能是 3.9；先检查版本。若版本不足，使用已安装的 `python3.11` / `python3.12` / `python3.13` 创建环境，或按下文用 uv。

## 标准安装

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows：`py -3 -m venv .venv`，PowerShell 激活 `.venv\Scripts\Activate.ps1`。
不激活也可直接用 `.venv/bin/python`（Windows 为 `.venv\Scripts\python.exe`）。
`-e` 是可编辑安装（editable install）：修改 `playbook/` 后不必重新安装。

可选功能：

```bash
python -m pip install -e '.[dl]'        # PyTorch
python -m pip install -e '.[io]'        # Parquet / Excel
python -m pip install -e '.[notebook]'  # JupyterLab
python -m pip install -e '.[kaggle]'    # Kaggle CLI
python -m pip install -e '.[dl,io,notebook]'
```

CUDA 的 PyTorch 包请按 [官方安装选择器](https://pytorch.org/get-started/locally/) 选择与你机器匹配的命令。
不要根据示例硬编码 CUDA 版本。Apple 芯片可试 `--device mps`，本库默认 CPU。

## 使用 uv（可选）

若已安装 uv：

```bash
uv venv --python 3.12
uv pip install -e '.[dl,io]'
source .venv/bin/activate
```

## Notebook 与版本

```bash
python -m ipykernel install --user --name ml-data-playbook --display-name 'ML Data Playbook'
jupyter lab
python -m pip freeze > outputs/environment.txt
```

先创建 outputs 目录再导出。Notebook 中选择对应 kernel。
`requirements-tested.txt` 是本次验证环境的版本记录，不是跨平台保证；常规安装优先使用 pyproject 中的兼容范围。

## 更新代码

```bash
git pull --ff-only
python -m pip install -e .
python -m scripts.smoke --suite core
```

不要把另一个项目的 requirements 全部装进同一个环境。
