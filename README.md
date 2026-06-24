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

导出准备标注的图片：

```bash
python tools/export_annotation_candidates.py \
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
python tools/prepare_anylabeling_dataset.py --raw-dir data/raw --out-dir data/annotation/hole_detect_v1
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
python tools/split_yolo_dataset.py \
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

训练完成后，可以使用 `tools/camera_detect.py` 接入本机相机或网络相机实时识别边缘安装孔。脚本默认加载当前训练得到的模型：

```text
runs/detect/data/models/hole_detect_v1/yolo11n_1280_v1/weights/best.pt
```

使用本机默认相机：

```bash
python tools/camera_detect.py --camera-index 0
```

当前环境如果安装的是 `opencv-python-headless`，请使用 Tkinter 窗口显示：

```bash
python tools/camera_detect.py --camera-index 1 --backend dshow --display-backend tkinter
```

如果打开相机后提示读不到画面，先扫描可用相机编号：

```bash
python tools/camera_detect.py --list-cameras
```

如果窗口能打开但画面全黑，先只看原始相机画面，不加载模型：

```bash
python tools/camera_detect.py --camera-index 1 --backend dshow --display-backend tkinter --preview-only --print-frame-stats
```

窗口左上角的 `min/max/mean` 是原始帧亮度范围。`mean` 很低并且 `max` 接近 0 时，说明相机输出本身就是黑图，需要换相机编号、后端、光源或检查镜头/曝光。

奥比中光 RGB-D 相机不要用普通 OpenCV 编号读取，否则可能拿到深度、红外或空流。请先安装 Orbbec Python wrapper：

```bash
python -m pip install --no-deps pyorbbecsdk2
```

这里使用 `--no-deps` 是为了避免 `pyorbbecsdk2` 自动升级 `numpy`，破坏 AnyLabeling、onnxruntime 和当前 YOLO 环境。

然后直接读取 Orbbec `COLOR_SENSOR` 彩色流：

```bash
python tools/camera_detect.py \
  --camera-source orbbec \
  --orbbec-sdk-dir D:/OrbbecSDK_v2 \
  --display-backend tkinter \
  --preview-only \
  --print-frame-stats
```

确认能看到彩色画面后，再运行 YOLO 实时识别：

```bash
python tools/camera_detect.py \
  --camera-source orbbec \
  --orbbec-sdk-dir D:/OrbbecSDK_v2 \
  --display-backend tkinter \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0
```

Orbbec 模式默认同时开启深度流，并使用软件 depth-to-color 对齐。检测结果中的红色坐标会显示：

```text
(x,y,Z=1234mm)
```

其中 `x,y` 是 RGB 图像中的孔中心像素坐标，`Z` 是该中心点对应的深度值，单位为毫米。如果显示 `Z=?`，通常表示该点深度无效、深度帧未对齐到彩色帧，或者该像素位置没有有效深度。

可以手动切换深度对齐模式：

```bash
python tools/camera_detect.py --camera-source orbbec --orbbec-depth-align sw --display-backend tkinter
python tools/camera_detect.py --camera-source orbbec --orbbec-depth-align hw --display-backend tkinter
```

Windows 上本机相机默认使用 DirectShow 后端。如果 `0` 号相机不可用，可以换编号或后端：

```bash
python tools/camera_detect.py --camera-index 1 --backend dshow
python tools/camera_detect.py --camera-index 0 --backend msmf
python tools/camera_detect.py --camera-index 0 --backend any
```

指定模型、置信度和推理尺寸：

```bash
python tools/camera_detect.py \
  --model runs/detect/data/models/hole_detect_v1/yolo11n_1280_v1/weights/best.pt \
  --camera-index 0 \
  --imgsz 1280 \
  --conf 0.25 \
  --device 0
```

使用 RTSP/HTTP 网络相机：

```bash
python tools/camera_detect.py --camera-url rtsp://user:password@192.168.1.10/stream1
```

如果当前环境使用的是 `opencv-python-headless`，窗口显示可能不可用，可以改为保存识别视频：

```bash
python tools/camera_detect.py \
  --camera-index 0 \
  --no-window \
  --save-video data/reports/hole_detect_v1/camera_detect.mp4
```

窗口模式下按 `q` 或 `Esc` 退出，按 `s` 保存当前识别截图到 `data/reports/hole_detect_v1/camera_snapshots/`。脚本会在检测框中心画红点，并显示中心像素坐标，后续可用于机械臂对准流程接入。
