# 手写数字识别项目一天闭环实施计划

> 执行方式：按任务依次使用测试驱动开发、规格审查和代码质量审查。除正式训练、桌面人工冒烟和打包外，测试不得下载 MNIST。

**目标：** 在一天内交付一个以 PyQt 桌面端为唯一产品入口、具备可复现训练评测、自动测试和 Windows 打包能力的求职作品。

**架构：** 单一 Python 包维护 CNN、预处理、空白判断、路径解析和 Predictor。训练模块复用同一模型及预处理配置并生成可信产物；PyQt 作为薄适配层直接调用 Predictor。无后端服务、Web、Docker或在线部署。

**技术栈：** Python 3.11、PyTorch、torchvision、Pillow、NumPy、PyQt5、pytest、Ruff、matplotlib、PyInstaller、GitHub Actions。

## 文件地图

- `src/digit_recognizer/core/`：模型、预处理、路径解析和推理。
- `src/digit_recognizer/training/`：确定性配置、训练、最终评测和产物生成。
- `src/digit_recognizer/desktop/app.py`：PyQt 画板、窗口和入口。
- `scripts/train.py`：正式训练命令入口。
- `scripts/sync_docs.py`：从评测 JSON 同步 README 和简历数字。
- `tests/core/`、`tests/training/`、`tests/desktop/`：快速确定性测试。
- `models/`、`reports/`：发布权重、元数据、指标和图表。
- `packaging/windows.spec`：Windows onedir 打包。
- `.github/workflows/quality.yml`：CPU 测试与代码检查。
- `.github/workflows/release.yml`：标签触发 Windows Release。
- `README.md`、`docs/resume-project.md`：作品集与简历材料。

## 范围约束

本计划明确取消 FastAPI、Uvicorn、API schema、Web 静态页、Dockerfile、Docker 工作流、在线部署和对应文档。不得以“增强展示”为理由重新加入。

---

### Task 1：建立可安装仓库骨架（已完成）

已完成：

- 包结构、`pyproject.toml`、MIT License、`.gitignore` 和 README 占位。
- Python 3.11 环境与 CPU/GPU 依赖契约。
- 初始安装和导入测试。

验收：项目可 editable install，快速测试与 Ruff 通过。

### Task 2：统一图像预处理（已完成）

已完成：

- PIL/NumPy 输入校验、透明白底合成、灰度与反色。
- 共享 28×28、均值和标准差配置。
- 空白输入判断与显式领域错误。
- 边界、非法输入和透明图测试。

验收：桌面端和训练端不复制尺寸或归一化常量。

### Task 3：实现模型与 Predictor（已完成）

已完成：

- 唯一 CNN 定义，参数量固定为 585,578。
- 结构化预测结果和安全 `weights_only` 权重加载。
- 环境变量、PyInstaller、工作目录和源码目录的模型路径优先级。
- 模型缺失、损坏、形状和路径回归测试。

验收：核心预测测试通过，开发和打包路径均可解析。

### Task 4：完成可复现训练与模型评测（代码完成，正式产物待重跑）

**文件：**

- `src/digit_recognizer/training/config.py`
- `src/digit_recognizer/training/runner.py`
- `scripts/train.py`
- `tests/training/test_config.py`

已完成的代码契约：

- seed 42、55,000/5,000 固定训练验证划分、默认 15 轮。
- 训练集增强，验证和测试不增强。
- Adam、ReduceLROnPlateau 和验证集最佳权重。
- 测试集只遍历一次，准确率低于 99.0% 非零退出。
- 临时 checkpoint、非有限指标拒绝、原子发布和 SHA-256。
- history、metrics、训练曲线、混淆矩阵和模型元数据。
- Matplotlib 在导入 `pyplot` 前强制使用 `Agg`，避免 Qt 后端阻塞。

- [x] 训练/核心轻量测试与双重审查通过。
- [x] RTX 4060、PyTorch 2.10.0+cu128 和 CUDA 可用性验证。
- [ ] 重新执行正式训练：

```powershell
.\.venv\Scripts\python.exe scripts\train.py --device auto --epochs 15
```

- [ ] 验证六类产物存在，准确率不低于 99.0%，元数据 SHA-256 与权重一致。
- [ ] 单独提交生成的模型和报告：

```powershell
git add models/mnist_cnn.pth models/model_metadata.json reports
git commit -m "data: publish verified MNIST model metrics"
```

### Task 5：重构现有 PyQt 桌面端

**文件：**

- Create: `src/digit_recognizer/desktop/app.py`
- Test: `tests/desktop/test_canvas.py`
- Modify: `pyproject.toml`（仅桌面依赖和 `digit-desktop` 入口）

- [ ] 先写 offscreen 失败测试：画板初始白色、转换为 280×280 PIL 图像。
- [ ] 实现 `DrawingCanvas`：自身坐标绘制、15 px 黑色圆角笔迹、`clear()`、内存转换 PIL。
- [ ] 测试清除和笔迹坐标，不启动真实窗口事件循环。
- [ ] 实现 `DigitRecognizerWindow`：注入 Predictor，显示数字、置信度和耗时。
- [ ] 测试识别成功、空白输入和推理错误的 UI 状态/提示映射。
- [ ] `main()` 使用 `default_model_path()` 加载模型；加载失败显示简洁错误并返回 1。
- [ ] 执行：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest tests\desktop -v
.\.venv\Scripts\python.exe -m ruff check src\digit_recognizer\desktop tests\desktop
```

- [ ] 人工冒烟：启动 `digit-desktop`，验证绘制、清除、空白拒绝、识别和关闭。
- [ ] 规格与质量审查通过后提交：

```powershell
git add pyproject.toml src/digit_recognizer/desktop tests/desktop
git commit -m "feat: add offline PyQt desktop client"
```

### Task 6：增加 CI 与 Windows 打包

**文件：**

- Create: `packaging/windows.spec`
- Create: `.github/workflows/quality.yml`
- Create: `.github/workflows/release.yml`

- [ ] PyInstaller 使用 onedir，不使用 onefile。
- [ ] 包含 `models/mnist_cnn.pth` 与必要 Qt/PyTorch 运行依赖，不包含 MNIST、训练数据、测试和报告源码。
- [ ] `console=False`，产物目录名为 `HandwrittenDigitRecognizer`。
- [ ] quality workflow 使用 Python 3.11 和 CPU PyTorch，执行 Ruff、pytest、包构建与桌面模块导入检查。
- [ ] release workflow 在 `v*` 标签使用 `windows-latest` 构建 ZIP，并上传 GitHub Release。
- [ ] 本地执行：

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\pyinstaller.exe packaging\windows.spec --clean --noconfirm
```

- [ ] 启动 `dist\HandwrittenDigitRecognizer\HandwrittenDigitRecognizer.exe` 完成人工冒烟。
- [ ] 不提交 `build/`、`dist/` 或解压后的运行目录。
- [ ] 审查通过后提交 CI 与打包配置。

### Task 7：生成可信 README、截图和简历材料

**文件：**

- Modify: `README.md`
- Create: `docs/resume-project.md`
- Create: `docs/images/desktop.png`
- Create: `scripts/sync_docs.py`
- Test: `tests/test_documentation.py`

- [ ] 先写文档一致性测试：README 与简历中的准确率、median、P95 必须等于 `reports/metrics.json`。
- [ ] `sync_docs.py` 从 JSON 模板化生成数字，禁止手填量化结果。
- [ ] README 包含英文短简介、中文正文、架构、桌面启动、训练/测试、真实评测、局限和 Release。
- [ ] 只拍一张干净的 PyQt 桌面截图，不创建 Web 截图。
- [ ] 简历项目固定三条：共享核心与 PyQt、可复现评测、测试/CI/Windows 交付。
- [ ] 如实说明课程项目来源、本人主导实现和单数字 MNIST 局限。
- [ ] 文档不得出现成员学号、教师信息、绝对路径或终端隐私。

### Task 8：最终验证与首版发布

- [ ] 运行全部质量门：

```powershell
.\.venv\Scripts\python.exe -m ruff check .
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m build
```

- [ ] 重新计算权重 SHA-256，验证元数据、参数量、准确率和文档数字一致。
- [ ] 启动源码桌面端与打包 EXE，验证完整离线识别闭环。
- [ ] 检查 Git 仅跟踪必要源码、模型、报告和文档；排除数据、缓存、原报告和构建目录。
- [ ] 搜索凭据、个人信息和绝对本机路径，结果必须为空。
- [ ] 提交所有必要变更；工作树干净后再推送 feature 分支并创建 PR。
- [ ] PR 检查通过后合并，创建 `v0.1.0` 标签并验证 Windows ZIP Release。

## 完成定义

以下条件同时满足才算首版闭环：

1. 正式模型准确率 ≥99.0%，六类产物一致且可核验。
2. PyQt 源码版和 Windows 打包版均可离线完成一次识别。
3. 全量测试、Ruff、包构建和 CI 通过。
4. README、桌面截图和简历材料只引用真实评测数据。
5. GitHub 仓库无后端/Web/Docker实现，也无隐私、数据集或构建垃圾。
