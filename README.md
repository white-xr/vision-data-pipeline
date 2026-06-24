# Vision Data Pipeline

通用视觉数据处理与 YOLO 训练工具仓库，面向数据清洗、数据筛选、数据增强、标注格式整理、数据集划分、YOLO 训练、模型评估和推理测试等流程。

## 仓库原则

代码、配置和文档上传 Git；数据保存在本地 `data/` 目录中，不上传 Git。

`data/` 用于保存原始采集图片、筛选结果、清洗结果、标注中间数据、YOLO 数据集、模型和报告。该目录已在 `.gitignore` 中忽略。

完整数据流转规范见 [docs/dataset_structure.md](docs/dataset_structure.md)。

## 数据目录

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

## FiftyOne 人工筛选

FiftyOne 用于快速浏览 `data/raw/` 下的原始图片，并通过人工打 tag 的方式筛选适合标注的图片。当前默认处理单目 Left 图，不处理 `left/right` 双目结构。

安装依赖：

```bash
pip install -r requirements.txt
```

启动浏览：

```bash
python tools/view_with_fiftyone.py --raw-dir data/raw --dataset-name hole_review_v1
```

如果同名 FiftyOne dataset 已存在，脚本默认会加载已有 dataset，避免误删人工筛选结果。确认要重建时再使用：

```bash
python tools/view_with_fiftyone.py --raw-dir data/raw --dataset-name hole_review_v1 --overwrite
```

在 FiftyOne App 中给图片打样本 tag：

```text
to_annotate
hard
bad
```

tag 含义：

- `to_annotate`：孔位清晰，适合标注。
- `hard`：有点暗、亮、反光、模糊或角度偏，但孔位还能判断，作为困难样本保留。
- `bad`：孔看不清、严重模糊、严重过曝、严重遮挡，不进入训练集。

筛选注意事项：

- 有孔但没标的图片不能进入训练集。
- 看不清孔中心的图片不要进入训练集。
- 困难样本如果人眼还能标，就应该保留。
- 真正没有孔但像孔的干扰图，后续可以作为负样本保留为空标注。

导出准备标注的图片：

```bash
python tools/export_annotation_candidates.py \
  --dataset-name hole_review_v1 \
  --out-dir data/annotation/hole_detect_v1 \
  --include-tags to_annotate hard
```

导出结果：

```text
data/annotation/hole_detect_v1/
├── images/
│   ├── base_01_000001.png
│   ├── base_01_000002.png
│   └── ...
└── export_manifest.csv
```

后续流程：

```text
FiftyOne 人工筛选
↓
导出 annotation/images
↓
CVAT 或 LabelImg 标注
↓
导出 YOLO Detect 数据集
↓
训练 YOLO
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
