# Vision Data Pipeline

通用视觉数据处理与 YOLO 训练工具仓库，面向数据清洗、数据筛选、数据增强、标注格式整理、数据集划分、YOLO 训练、模型评估和推理测试等流程。

## 仓库原则

代码、配置和文档上传 Git；数据保存在本地 `data/` 目录中，不上传 Git。

`data/` 用于保存原始采集图片、筛选结果、清洗结果、标注中间数据、YOLO 数据集、模型和报告。该目录已在 `.gitignore` 中忽略。

## 数据目录

完整数据流转规范见 [docs/dataset_structure.md](docs/dataset_structure.md)。

```text
data/
├── raw/          # 原始采集数据，不要改
├── selected/     # 从 raw 中挑选或提取出来的图片
├── cleaned/      # 清洗后的 good/bad 图片
├── annotation/   # 准备标注的图片
├── datasets/     # YOLO 格式数据集
├── models/       # 训练好的模型
└── reports/      # 清洗、训练、评估报告
```

## 不提交到 Git 的内容

- `data/`
- `runs/`
- `weights/`
- `models/`
- 模型权重和导出文件：`*.pt`、`*.pth`、`*.onnx`、`*.engine`
- 视频和压缩包：`*.mp4`、`*.avi`、`*.zip`、`*.rar`
- 环境变量文件：`.env`

## 维护方向

- 数据清洗
- 数据筛选
- 数据增强
- 标注格式整理
- 数据集划分
- YOLO 训练脚本
- 模型评估
- 推理测试

## 提交规范

提交信息和推送规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
