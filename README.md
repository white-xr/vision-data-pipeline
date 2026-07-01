# Vision Data Pipeline

通用视觉数据处理与 YOLO 训练工具仓库，面向数据清洗、数据筛选、数据增强、标注格式整理、数据集划分、YOLO 训练、模型评估和推理测试等流程。

## 仓库原则

代码、配置和文档上传 Git；数据保存在本地 `data/` 目录中，不上传 Git。

`data/` 用于保存原始采集图片、筛选结果、清洗结果、标注中间数据、YOLO 数据集、模型和报告。该目录已在 `.gitignore` 中忽略。

完整数据流转规范见 [docs/dataset_structure.md](docs/dataset_structure.md)。

## 环境安装

推荐使用 Conda 管理项目环境。后续所有脚本默认在 `vision-data` 环境中运行。

创建环境：

```bash
conda env create -f environment.yml
```

激活环境：

```bash
conda activate vision-data
```

运行工具前请先激活环境。更多环境创建、更新和验证命令见 [docs/environment_setup.md](docs/environment_setup.md)。

## 工具目录

```text
tools/
├── dataset/  # 数据浏览、筛选导出、标注准备、数据集划分和格式转换
└── camera/   # 实时相机识别、YOLO 推理、显示和后处理
```

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

启动浏览：

```bash
python tools/dataset/view_with_fiftyone.py --raw-dir data/raw --dataset-name hole_review_v1
```

如果同名 FiftyOne dataset 已存在，脚本默认会加载已有 dataset，避免误删人工筛选结果。确认要重建时再使用：

```bash
python tools/dataset/view_with_fiftyone.py --raw-dir data/raw --dataset-name hole_review_v1 --overwrite
```

在 FiftyOne App 中给图片打样本 tag：

```text
to_annotate
hard
bad
```

导出准备标注的图片：

```bash
python tools/dataset/export_annotation_candidates.py \
  --dataset-name hole_review_v1 \
  --out-dir data/annotation/hole_detect_v1 \
  --include-tags to_annotate hard
```

## AnyLabeling 检测框标注

当前工业视觉孔位检测任务只做目标检测，不做分割。只标边缘安装孔，不标内部孔。完整标注规则见 [docs/labeling_rules.md](docs/labeling_rules.md)。

类别文件：

```text
configs/classes.txt
```

类别只使用：

```text
cover_edge_hole
base_edge_hole
```

从 `data/raw/` 准备 AnyLabeling 标注目录：

```bash
python tools/dataset/prepare_anylabeling_dataset.py --raw-dir data/raw --out-dir data/annotation/hole_detect_v1
```

该命令只复制原图，不会删除、移动、重命名 `data/raw/` 中的原始图片。复制后的图片会带上 batch 名，例如：

```text
data/raw/base_01/000001.png
data/annotation/hole_detect_v1/images/base_01_000001.png
```

启动 AnyLabeling：

```bash
anylabeling
```

如果命令不可用，可以尝试：

```bash
python -m anylabeling.app
```

打开 AnyLabeling 后：

1. 打开 `data/annotation/hole_detect_v1/images/`
2. 使用矩形框标注边缘安装孔
3. 类别只使用 `cover_edge_hole` 和 `base_edge_hole`
4. 导出 YOLO Detect 格式，类别顺序必须与 `configs/classes.txt` 一致
5. 标签保存到 `data/annotation/hole_detect_v1/labels/`

训练前必须检查：

- 每张参与训练的图片是否有对应 `.txt` 标签文件。
- 有孔但没标注的图片不能进入训练集。
- 真正无目标的负样本可以保留为空标签。
- 标签类别 id 必须只包含 `0` 或 `1`。

## YOLO Detect 数据集划分

从已标注目录生成 YOLO Detect 训练集和验证集：

```bash
python tools/dataset/split_yolo_dataset.py \
  --src-dir data/annotation/hole_detect_v1 \
  --out-dir data/datasets/hole_detect_v1 \
  --train-ratio 0.8 \
  --seed 42
```

脚本只会复制有同名 `.txt` 标签的图片，不会把未标注图片放进训练集。输出结构：

```text
data/datasets/hole_detect_v1/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── data.yaml
```

## 后续流程

```text
FiftyOne 人工筛选
↓
导出或准备 annotation/images
↓
AnyLabeling 标注边缘安装孔
↓
划分 YOLO Detect 数据集
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

## 相机实时识别

通用实时识别入口是 `tools/camera/camera_detect.py`，可接入奥比中光 RGB-D 相机、普通 USB 相机或 RTSP/HTTP 视频流，并自动兼容 YOLO 检测模型和分割模型。

默认配置文件是：

```text
configs/camera_detect.yaml
```

启动实时识别：

```bash
python tools/camera/camera_detect.py
```

运行前检查配置、模型路径和后处理插件：

```bash
python tools/camera/camera_detect.py --dry-run
```

只预览相机、不加载模型：

```bash
python tools/camera/camera_detect.py --preview-only
```

扫描相机：

```bash
python tools/camera/camera_detect.py --list-cameras
```

`configs/camera_detect.yaml` 是当前要运行的模型配置。日常换模型通常只改 `model`、`task`、`imgsz`、`conf`、`draw` 和可选 `classes/postprocess`，不需要改 Python 代码。

奥比中光 335L/305 默认走 `camera.source: orbbec`。首次使用 Orbbec 模式前安装：

```bash
python -m pip install --no-deps pyorbbecsdk2
```

后处理通过插件接入，业务逻辑不写进通用检测脚本。当前 Base/Cover 检测配置启用了 `runtime.postprocess_plugins.base_cover_alignment`，用于按 bbox 内 anchor 点锁定 Base 基准点，并实时显示 Cover 相对 Base 的 `dx/dy`。如果某个模型不需要后处理，把 `postprocess.enabled` 改为 `false` 即可。

插件配置示例：

```yaml
model: runs/xxx/weights/best.pt
draw: detect
postprocess:
  enabled: true
  module: runtime.postprocess_plugins.base_cover_alignment
  function: process
  params: {}
```

窗口模式下按 `q` 或 `Esc` 退出，按 `s` 保存当前识别截图，按 `R` 重置后处理锁定状态。检测模型会绘制框和中心点；分割模型会按配置绘制 mask、框、标签和中心点。

多相机、多模型串行验证入口是：

```bash
python tools/camera/multi_camera_detect.py --dry-run
python tools/camera/multi_camera_detect.py
```

默认配置文件是 `configs/camera_pipelines.yaml`。pipeline 里用 `camera: orbbec_335l` 或 `camera: orbbec_305` 绑定相机别名，模型写在该 pipeline 的 `models` 列表里，同一相机内多个模型会按顺序串行推理。只运行某个 pipeline：

```bash
python tools/camera/multi_camera_detect.py --pipeline base_cover_335l
```
