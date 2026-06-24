# 环境安装说明

## Conda 环境创建

```bash
conda env create -f environment.yml
```

## 激活环境

```bash
conda activate vision-data
```

## 更新环境

```bash
conda env update -f environment.yml --prune
```

## 验证环境

```bash
python -c "import fiftyone as fo; print('FiftyOne OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import anylabeling; print('AnyLabeling OK')"
```

## 运行 FiftyOne 浏览工具

```bash
python tools/view_with_fiftyone.py --raw-dir data/raw --dataset-name hole_review_v1
```

## 导出准备标注的图片

```bash
python tools/export_annotation_candidates.py --dataset-name hole_review_v1 --out-dir data/annotation/hole_detect_v1 --include-tags to_annotate hard
```

## 准备 AnyLabeling 标注目录

```bash
python tools/prepare_anylabeling_dataset.py --raw-dir data/raw --out-dir data/annotation/hole_detect_v1
```

## 启动 AnyLabeling

```bash
anylabeling
```

如果命令不可用，可以尝试：

```bash
python -m anylabeling.app
```
