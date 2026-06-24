# AnyLabeling 孔位检测标注规则

## 标注目标

当前任务只做目标检测框标注，不做分割。后续模型使用 YOLO Detect，用于眼在手内相机识别边缘安装孔，并辅助机械臂定位对准。

只标红色箭头所指的外边缘安装孔。

类别文件为 [configs/classes.txt](../configs/classes.txt)：

```text
cover_edge_hole
base_edge_hole
```

类别含义：

- `cover_edge_hole`：盖板边缘安装孔
- `base_edge_hole`：底座边缘安装孔

## 不标目标

以下内容全部不标：

- 内部孔
- 内部柱子
- 内部圆点
- 内部凹槽
- 内部黑色结构
- 非后续机械臂对准目标的孔

## 框选规则

每个边缘安装孔画一个检测框。

框住：

```text
黑色孔洞 + 周围一圈孔位结构 / 倒角 / 凹槽边缘
```

不要：

- 只框黑色小点
- 框太大
- 框整个盖板
- 框整个底座
- 框内部孔

## 坏图处理

标注时人工判断：

- 孔位清楚：标注。
- 有点暗、反光、轻微模糊，但还能判断孔中心：标注，作为困难样本。
- 孔看不清或不确定中心：不要标，后续不要进入训练集。
- 有孔但没有标注的图片不能进入训练集。

## AnyLabeling 使用流程

安装：

```bash
pip install anylabeling "imgviz<2"
```

启动：

```bash
anylabeling
```

如果命令不可用，可以尝试：

```bash
python -m anylabeling.app
```

打开 AnyLabeling 后：

1. 打开图片目录：

```text
data/annotation/hole_detect_v1/images/
```

2. 创建或导入类别，只使用：

```text
cover_edge_hole
base_edge_hole
```

3. 使用矩形框标注，每个边缘安装孔一个框。
4. 导出 YOLO Detect 格式，类别顺序必须与 [configs/classes.txt](../configs/classes.txt) 一致。
5. 标签保存到：

```text
data/annotation/hole_detect_v1/labels/
```

标注规则：每个边缘安装孔一个框，只标边缘孔，内部孔不标。

## 训练前检查

训练前必须检查：

- 每张参与训练的图片是否有对应 `.txt` 标签文件。
- 有孔但没标注的图片不能进入训练集。
- 真正无目标的负样本可以保留为空标签。
- 标签类别 id 必须只包含 `0` 或 `1`。

