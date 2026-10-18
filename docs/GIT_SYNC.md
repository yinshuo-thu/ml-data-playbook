# GitHub 同步与日常维护

本地仓库目录：`ml-data-playbook/`。远端使用账号 `yinshuo-thu` 下同名独立仓库。

## 首次创建远端（若尚未创建）

在 [GitHub 新建页面](https://github.com/new) 创建 `ml-data-playbook`，建议 Private。
已有本地提交时，不额外勾选初始化 README / License / .gitignore，避免出现两段独立历史。
然后在仓库根目录：

```bash
git remote -v
git push -u origin main
```

本地 origin 使用 `ssh://git@ssh.github.com:443/yinshuo-thu/ml-data-playbook.git`。
这是 GitHub 官方支持的 SSH 443 入口。SSH 用于认证 Git 推送；创建远端仓库仍需网页或 GitHub API。
如果遇到服务器指纹提示，应与 [GitHub 公布的指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints) 核对。

## 日常更新

```bash
git pull --ff-only
# 修改示例和文档
python -m scripts.smoke --suite core
python -m unittest discover -s tests -v
git status
git add README.md docs data_analysis machine_learning deep_learning
# 根据实际修改选择文件，不必每次添加以上全部目录
git commit -m 'Add a reusable analysis recipe'
git push
```

冲突时先处理冲突，不使用强制推送覆盖远端。
`data/`、`outputs/`、模型、虚拟环境和 .env 已被忽略；提交前仍要检查 `git diff --cached`。
