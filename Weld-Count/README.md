# Weld-Count

焊点计数与 OK/NG 判定基线，支持 **一张原图对应一个 JSON 标注文件**、16/25 个焊点、阈值校准和离线验收。

## 推荐数据目录

```text
dataset/
├── images/
│   ├── 0001.jpg
│   └── 0002.jpg
└── annotations/
    ├── 0001.json
    └── 0002.json
```

每个 JSON 对应同名图片。支持自定义 `imagePath`（LabelMe 常见格式）；如果没有图片字段，会按同名 `.jpg/.png/.bmp/.tif` 自动查找。标注文件可以是 LabelMe 风格，也可以是本项目风格。

### 本项目格式

```json
{
  "image": "0001.jpg",
  "total": 16,
  "label": "OK",
  "annotations": [
    {"bbox": [10, 20, 30, 40], "category": "weld"},
    {"point": [80, 90], "category": "weld"}
  ]
}
```

### LabelMe 风格

```json
{
  "imagePath": "0001.jpg",
  "imageWidth": 1920,
  "imageHeight": 1080,
  "total": 25,
  "status": "NG",
  "shapes": [
    {"label": "weld", "shape_type": "point", "points": [[100, 200]]},
    {"label": "weld", "shape_type": "rectangle", "points": [[300, 400], [350, 450]]}
  ]
}
```

字段兼容关系：

- 图片：`image`、`imagePath`、`file_name`、`filename`
- 总焊点数：`total`、`expected`、`expected_count`、`weld_count`
- 质量标签：`label`、`status`、`result`、`quality`，也支持 `合格/不合格`、`良品/不良`
- 标注对象：`annotations`、`objects`、`shapes`、`points`、`instances`

如果 JSON 没有 `total`，可在命令中通过 `--total 16` 或 `--total 25` 提供默认值。建议正式数据直接写入每个 JSON，避免 16/25 混料时误判。

## 安装

```bash
pip install -r Weld-Count/requirements.txt
```

从 `Weld-Count` 目录执行以下命令，或将其加入 `PYTHONPATH`。

## 运行流程

### 1. 生成传统视觉 baseline 预测

```bash
cd Weld-Count
python -m weld_count.cli baseline \
  --annotations ../dataset/annotations \
  --images-dir ../dataset/images \
  --out ../dataset/val_predictions.jsonl \
  --min-area 8
```

`baseline` 会扫描标注目录下所有 `*.json`，每个 JSON 读取一次对应图片，并生成一条预测记录。实际生产建议将此步骤替换为 YOLO/ONNX 等检测器，只要输出以下格式即可：

```json
{"image":"0001.jpg","detections":[{"score":0.93,"bbox":[10,20,30,40]}]}
```

### 2. 验证集校准

```bash
python -m weld_count.cli calibrate \
  --annotations ../dataset/val/annotations \
  --predictions ../dataset/val_predictions.jsonl \
  --out ../dataset/policy.json
```

校准目标严格是：

- NG→OK 漏杀数量为 0
- OK→NG 过杀率不超过 6%

如果不存在满足条件的阈值，程序会失败并提示改善检测器或数据，而不是输出一个不达标的策略。

### 3. 独立测试集验收

```bash
python -m weld_count.cli evaluate \
  --annotations ../dataset/test/annotations \
  --predictions ../dataset/test_predictions.jsonl \
  --policy ../dataset/policy.json
```

输出包括样本数、混淆矩阵、NG recall、漏杀数、过杀率和标注计数误差。

## 判定规则

- 16 个焊点：`count > 8`，即至少 9 个才是 OK
- 25 个焊点：`count > 12`，即至少 13 个才是 OK

判定策略位于 `weld_count/policy.py`。置信度过滤和多数计数必须在验证集校准，不能只凭单张图片调整。

## 上线注意事项

1. train/val/test 按工件、批次和拍摄条件隔离，同一工件不得跨集合。
2. 0 漏杀约束只在独立测试集上验收，不能只在训练数据上统计。
3. 6% 是点估计；样本少时应同时报告置信区间。
4. 低置信度、检测数量异常或定位超出 ROI 的样本建议输出 `UNCERTAIN` 并人工复核，不要为了形式上的 0 漏杀强行判 OK。
5. LabelMe 的 `shapes` 默认每个 shape 计为一个候选焊点；若一个焊点拆成多个 shape，需要在数据清洗阶段合并。

运行测试：

```bash
cd Weld-Count
python -m unittest discover -s tests -v
```
