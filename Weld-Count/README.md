# Weld-Count

焊点计数与 OK/NG 判定基线，支持 16/25 个焊点、JSON 标注、传统视觉推理、阈值校准和离线验收。

## 目标与判定定义

- `total=16`：合格下限可配置为 `>8`（严格多数，默认）或 `>9`（业务规则）。
- `total=25`：严格多数为 `>12`，即至少 13 个。
- **零漏杀约束**：这里将“漏杀”定义为真实 NG 被判为 OK（NG→OK），校准时强制 `NG recall=100%`。
- **过杀**：真实 OK 被判为 NG（OK→NG）。在验证集上搜索最宽松且仍满足零漏杀的规则；若验证过杀率超过 6%，系统会明确报警，而不会伪造达标结果。

## 推荐生产流程

1. 按产品型号和光学/曝光条件划分 train/val/test，禁止同一工件的图片跨集合。
2. 用 JSON 标注焊点框（或点），并为每张图片提供 `total`、`label`（OK/NG）。
3. 训练/接入一个焊点检测器，输出每个候选焊点的置信度和框；本仓库同时提供无需训练的 OpenCV 连通域基线用于快速打通流程。
4. 在 `val` 上运行 `calibrate`：只使用验证集选择置信度阈值和计数门限，并保存 `policy.json`。
5. 在完全独立的 `test` 上运行 `evaluate`，报告计数误差、NG 漏检数、NG recall、OK 过杀率和混淆矩阵。
6. 上线时固定模型、阈值、版本和数据分布；任何变更必须重新校准并复验。

## JSON 格式

每行一个样本（JSONL），或一个 JSON 数组均可。图片路径相对于标注文件目录。

```json
{"image":"images/a.jpg","total":16,"label":"OK","annotations":[{"bbox":[10,20,30,40],"category":"weld"}]}
{"image":"images/b.jpg","total":16,"label":"NG","annotations":[{"point":[15,25],"category":"weld"}]}
```

也接受 `objects`、`shapes`、`points` 字段，以及 `x,y,w,h` 或 `x1,y1,x2,y2` 框格式。实际检测器输出格式为 `[{"score":0.93,"bbox":[x1,y1,x2,y2]}]`。

## 安装与命令

```bash
pip install -r Weld-Count/requirements.txt
python -m weld_count.cli calibrate --annotations data/val.jsonl --predictions val_predictions.jsonl --out policy.json
python -m weld_count.cli evaluate --annotations data/test.jsonl --predictions test_predictions.jsonl --policy policy.json
python -m weld_count.cli baseline --annotations data/val.jsonl --out val_predictions.jsonl --min-area 8
```

预测 JSONL 每行：`{"image":"images/a.jpg","detections":[{"score":0.9,"bbox":[...] }]}`。若接入 YOLO/ONNX，只需将其输出转换成该格式。

## 性能验收与风险

- 样本量较小时，6% 只应作为点估计不能作为可靠保证；建议每种型号至少覆盖不同批次、光照、角度，并报告 Wilson 置信区间。
- 如果零漏杀时验证集过杀仍 >6%，优先改善检测器（召回、去重、定位、光照增强），不要简单放宽计数阈值。
- 生产中建议保留 `UNCERTAIN`：低于检测置信度或超出 ROI/几何规则的图片进入人工复核，不能为了“零漏杀”强行判 OK。

详见 `Weld-Count/weld_count` 和 `Weld-Count/tests`。
