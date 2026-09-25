# GitHub 新仓库与项目管理教程

本项目建议把“代码与小型配置”放在 GitHub，把模型权重、checkpoint、数据集、completion、judge cache 和完整运行结果放在服务器大盘、Hugging Face Hub 或对象存储。仓库已经提供 `.gitignore` 防止常见的大文件和密钥被误提交。

如果已克隆本仓库，先执行 `git remote -v` 查看现有远端，无需再次创建仓库或重命名远端。提交前的部署检查入口为：

```bash
python -m pip install -r requirements-test.txt
make test
DATASET_NAME=example/UltraChat-200k make dry-run
git diff --check
```

测试和 dry-run 不加载模型。远端 CI 使用同一套 CPU 测试；GPU/NCCL/实际训练验收需单独记录，不能用 CI 通过代替。

## 1. 首次上传前检查

在项目根目录执行：

```bash
git status --short
git diff --check
git diff --stat
git grep -nE '(sk-[A-Za-z0-9_-]{16,}|api[_-]?key[[:space:]]*=[[:space:]]*[^$])' -- . ':!docs/*' || true
```

确认列表中没有以下内容：

- `.env`、API key、Hugging Face token、W&B key；
- `checkpoint-*`、`*.safetensors`、`pytorch_model*.bin`；
- `grpo_runs/`、completion、reward data、judge cache；
- 原始数据集或个人路径中的隐私信息。

如果密钥曾经进入 Git 历史，仅删除当前文件不够：应立即吊销并重建密钥，再清理历史。首轮最稳妥的是先把 GitHub 仓库设为 private。

## 2. 在 GitHub 网页创建空仓库

1. 登录 GitHub，打开 `https://github.com/new`。
2. Repository name 设为 `grpo`。
3. 选择 Private（准备开源时再改 Public）。
4. 不要勾选初始化 README、`.gitignore` 或 License；本地项目已经包含这些文件，远端初始化会额外制造一次无关历史。
5. 点击 Create repository，复制页面给出的 SSH 地址，例如：

```text
git@github.com:YOUR_NAME/grpo.git
```

## 3. 配置 Git 身份和 SSH

只需在这台服务器上配置一次：

```bash
git config --global user.name "你的 GitHub 用户名或姓名"
git config --global user.email "你的 GitHub 邮箱"
ls -al ~/.ssh
```

如果没有可用的 `id_ed25519.pub`：

```bash
ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

把最后一条命令输出的整行公钥添加到 GitHub：Settings → SSH and GPG keys → New SSH key。随后验证：

```bash
ssh -T git@github.com
```

首次连接时应核对 GitHub 官方公布的 host fingerprint，再输入 `yes`。成功消息会包含你的 GitHub 用户名。

## 4. 推荐上传方式：保留原项目历史

保留历史最有利于论文代码的来源追踪。先查看已有远端，不要直接覆盖：

```bash
git remote -v
```

把新仓库作为名为 `publication` 的第二远端：

```bash
git remote add publication git@github.com:YOUR_NAME/grpo.git
git remote -v
```

提交本次改动：

```bash
git switch -c codex/five-gpu-grpo
git add .
git status --short
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add reproducible five-GPU GRPO workflow"
```

把当前提交作为新仓库的 `main` 推送，但不重命名本地工作分支：

```bash
git push -u publication HEAD:main
```

在 GitHub 页面确认文件完整、Actions 通过，再决定是否把 `publication` 改成默认 `origin`。保留原上游地址时，推荐命名为：

```bash
git remote rename origin upstream
git remote rename publication origin
git remote -v
```

若本地已经没有 `origin`，只需执行第二条。不要在不确认 `git remote -v` 的情况下照抄 rename 命令。

## 5. 可选方式：完全干净的新历史

只有在你明确不想保留上游 commit 历史时使用。不要在当前工作目录删除 `.git`。在服务器另建目录并复制已跟踪文件：

```bash
mkdir -p "$HOME/repos/grpo"
git archive --format=tar HEAD | tar -x -C "$HOME/repos/grpo"
cd "$HOME/repos/grpo"
git init -b main
git add .
git commit -m "chore: initialize GRPO project"
git remote add origin git@github.com:YOUR_NAME/grpo.git
git push -u origin main
```

注意：`git archive HEAD` 只包含已经提交的内容。因此应先在原目录提交本次修改，或者继续使用上一节的保留历史方案。论文复现项目通常更推荐保留历史。

## 6. 日常分支与实验管理

建议约定：

- `main`：始终保持可复现；
- `experiment/<主题>`：改变算法或超参数的实验；
- `fix/<问题>`：修复 bug；
- 一个 Pull Request 只解决一个主题；
- recipe 或算法变化必须同步修改 `EXPERIMENT_GRPO_5GPU.md`；
- 每次正式实验创建一个 GitHub Experiment issue，填写 commit SHA、启动命令、run directory 和结论。

典型流程：

```bash
git switch main
git pull --ff-only origin main
git switch -c experiment/grpo-ranking-ultrachat
# 修改并验证
git add <明确的文件列表>
git commit -m "exp: add UltraChat ranking GRPO run"
git push -u origin experiment/grpo-ranking-ultrachat
```

然后从 GitHub 创建 Pull Request，等待 `lightweight-ci` 通过后合并。CI 检查配置解析、服务协议、进程生命周期以及 Python/shell 语法，不会下载模型或占用 GPU。

## 7. 保护 main 与发布版本

在 GitHub 仓库 Settings → Rules → Rulesets 创建 branch ruleset：

- Target branch：`main`；
- Require a pull request before merging；
- Require status checks，选择 `static-checks`；
- Block force pushes；
- Require conversation resolution（多人协作时建议）。

完成一次论文里程碑后打带注释的 tag：

```bash
git switch main
git pull --ff-only
git tag -a v0.1-grpo-baseline -m "Five-GPU GRPO baseline used for initial experiments"
git push origin v0.1-grpo-baseline
```

实验 issue 中同时记录 tag、commit SHA 和服务器结果目录。服务器目录可能被清理，因此正式结果还应复制到长期存储，并保存校验值：

```bash
sha256sum /path/to/result.json > /path/to/result.json.sha256
```

## 8. 大文件与密钥原则

GitHub 不应成为训练 checkpoint 仓库。即使 Git LFS 能存大文件，数十到数百 GB 的训练产物也会带来配额、下载和版本管理成本。推荐：

- GitHub：代码、YAML、文档、轻量汇总指标；
- Hugging Face Hub：最终 adapter/merged model（确认许可后）；
- `/data` 或对象存储：checkpoint、reward data、完整 logs/completions；
- W&B：曲线和表格；
- GitHub issue：指向上述 artifact，并记录 SHA256。

任何 API key 只放环境变量或受权限保护的 secret manager。不要把 key 写进 shell 脚本、issue、日志或 Actions workflow。
