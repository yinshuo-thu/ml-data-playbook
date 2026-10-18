# 官方参考与数据来源

本仓库代码围绕下列官方 API 编写；示例是本仓库实现，链接用于继续查参数和边界行为。
稳定版文档会更新，验证版本见 [VALIDATION](VALIDATION.md)。

- [Pandas 用户手册](https://pandas.pydata.org/docs/user_guide/index.html)：索引、缺失、分组、连接、窗口。
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)：数组与广播。
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)：组合预处理与模型。
- [scikit-learn 常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html)：数据泄漏。
- [scikit-learn 模型选择](https://scikit-learn.org/stable/model_selection.html)：验证、搜索、指标。
- [scikit-learn 模型持久化](https://scikit-learn.org/stable/model_persistence.html)：兼容与加载格式。
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)：数据加载与训练。
- [PyTorch 安装](https://pytorch.org/get-started/locally/)：平台与加速器安装选择。
- [Matplotlib 示例](https://matplotlib.org/stable/gallery/index.html)：图表参数。
- [SciPy 统计](https://docs.scipy.org/doc/scipy/reference/stats.html)：统计检验。
- [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)：真实比赛、数据字典、提交要求。

合成演示数据由 `playbook/data.py` 固定随机种子生成；digits 数据由 scikit-learn 提供。
Kaggle 原始数据需用户从比赛页获取并遵守相应规则，不随本仓库上传。
