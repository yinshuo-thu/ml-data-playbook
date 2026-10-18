# 辅助命令

- `python -m scripts.make_demo_data`：在 `data/demo/` 创建回归与分类演示 CSV。
- `python -m scripts.smoke --suite core`：单独进程运行核心离线示例，错误不会被吞掉。
- `python -m scripts.smoke --suite deep`：运行 PyTorch 示例，需安装 `.[dl]`。
- `python -m unittest discover -s tests -v`：检查脏数据隔离、时序特征、分组划分、模型重载和文档链接。

日志保存在 `outputs/smoke_logs/`，汇总为 `outputs/smoke_core.json` / `outputs/smoke_deep.json`。
真实 Kaggle 数据不作为 CI 必需输入，运行方式见 [案例](../cases/house_prices/README.md)。
