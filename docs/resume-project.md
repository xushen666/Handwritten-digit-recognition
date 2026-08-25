# 手写数字识别桌面应用

**技术栈：** Python 3.11 / PyTorch / torchvision / PyQt5 / pytest / GitHub Actions / PyInstaller

**GitHub：** https://github.com/xushen666/Handwritten-digit-recognition.git

**项目背景：** 课程设计项目，核心实现与公开版本工程化重构由本人主导完成。

- 重构共享模型、图像预处理与 Predictor 推理核心，驱动 PyQt5 离线手写画板展示数字、置信度与延迟，消除训练和桌面推理逻辑重复。
- 建立无泄漏的可复现训练评测流程，以固定训练/验证划分选择最佳权重，测试集仅最终评测；达到 99.57% 测试准确率，CPU 单样本延迟 median 2.63 ms、P95 3.38 ms。
- 使用 pytest 与 Ruff 覆盖核心、训练和桌面关键路径，通过 GitHub Actions 执行质量门，并以 PyInstaller onedir 交付 Windows 可执行程序。

适用边界：MNIST 风格单数字分类教学项目，并非生产级 OCR。
