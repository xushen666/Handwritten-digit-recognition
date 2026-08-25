# Windows Release 启动修复设计

## 背景与根因

`v0.1.0` 的 Windows 发布包在普通桌面环境启动时，PyTorch 加载 `c10.dll` 失败并抛出
`WinError 1114`。源码已经让 `Predictor` 在 PyQt 前导入，但 PyInstaller 会先执行标准
`pyi_rth_pyqt5` 运行时钩子；该钩子通过 `create_embedded_qt_conf()` 导入
`PyQt5.QtCore`，因此打包程序仍然先加载 Qt。

现有 Release 冒烟测试只检查进程在 8 秒后是否存活。窗口模式的 PyInstaller 会用错误对话框
展示未捕获的导入异常，进程在对话框关闭前仍存活，导致测试产生假阳性。

## 目标与非目标

目标是让 Windows 打包程序在 PyQt 前加载 PyTorch，并让 Release 工作流能够区分正常启动和
错误对话框。修复版保留 `v0.1.0`，通过新标签 `v0.1.1` 发布。

本次不调整模型、训练流程、桌面交互、PyTorch 版本范围或发布包形态，也不增加后端服务。

## 设计

### PyInstaller 运行时顺序

新增 `packaging/pyi_rth_torch_first.py`，只执行一次 `import torch`。在
`packaging/windows.spec` 的 `Analysis.runtime_hooks` 中显式注册该文件。PyInstaller 的自定义
运行时钩子先于标准运行时钩子执行，因此 PyTorch 会在 `pyi_rth_pyqt5` 导入 QtCore 前完成加载。

### 可判定的打包冒烟测试

桌面入口支持内部参数 `--smoke-test`。该模式仍创建 `QApplication`、加载真实模型并构造
`DigitRecognizerWindow`，但不显示窗口或进入事件循环，成功时立即返回退出码 0。正常用户启动
路径保持不变。

Release 工作流用 `Start-Process` 启动 `EXE --smoke-test`，最多等待 30 秒：

- 30 秒内以 0 退出：通过；
- 非 0 退出：失败；
- 未退出：判定为错误对话框或启动挂起，终止进程并失败。

### 测试与验收

单元测试覆盖 smoke-test 不显示窗口、不进入事件循环且仍加载模型。交付配置测试覆盖自定义
运行时钩子存在、spec 注册顺序，以及 Release 工作流使用参数、超时和退出码而不是固定睡眠。

最终验收包括完整 pytest、Ruff、文档同步检查、PyInstaller onedir 构建、运行打包后的
`--smoke-test`，以及人工双击启动桌面窗口。`v0.1.1` 发布包必须能在当前 Windows 电脑上完成
上述验收。

