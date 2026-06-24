# 数据目录结构规范

本仓库是通用视觉数据处理与 YOLO 训练工具仓库。代码、配置和文档提交到 Git，数据统一放在仓库根目录下的 `data/` 中，并且 `data/` 必须被 `.gitignore` 忽略，不上传到 Git。

## 总体结构

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

## 原始采集数据

`data/raw/` 用于保存原始采集图片。这里的数据是最初始来源，后续清洗、筛选、标注和训练流程都应从它派生，不要直接修改、删除或重命名原始文件。

允许按采集批次组织，例如：

```text
data/raw/
├── base_001/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── base_002/
├── cover_001/
├── cover_002/
└── cover_base_001/
```

当前已有批次目录可以继续沿用现有命名格式，例如 `base_01/`、`cover_01/`。后续脚本应以目录结构为准，不强制改名历史数据。

注意：`.png` 是图片文件，不是文件夹。路径中不要出现下面这种形式：

```text
data/raw/base_001/000001.png/
```

正确形式是：

```text
data/raw/base_001/000001.png
```

## 双目 RGB 原始数据

如果一批原始数据来自双目 RGB 相机，可以在批次目录下使用 `left/` 和 `right/`：

```text
data/raw/base_001/
├── left/
│   ├── 000001.png
│   └── ...
└── right/
    ├── 000001.png
    └── ...
```

左右目图片应尽量保持同名编号，方便后续配对、筛选和导出。

## 流转目录说明

- `data/selected/`：从 `raw/` 中按规则挑选、抽帧或复制出的候选图片。
- `data/cleaned/`：清洗后的图片，可按 `good/`、`bad/` 或具体问题类型继续分组。
- `data/annotation/`：准备送入标注工具的图片和中间文件。
- `data/datasets/`：最终 YOLO 数据集，例如 `images/train`、`images/val`、`labels/train`、`labels/val`、`data.yaml`。
- `data/models/`：训练好的模型文件或本地模型备份。
- `data/reports/`：清洗统计、训练报告、评估结果和可视化图表。

## Git 管理规则

`data/` 目录只在本地使用，不提交到 Git。不要把原始图片、标注导出、训练数据集、模型权重、视频、压缩包或 `.env` 文件加入版本库。

提交前至少检查：

```bash
git status
git ls-files data
```

`git ls-files data` 应该没有输出。
