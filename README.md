# 工业裂缝视觉检测系统

面向工厂质检、隧道养护、桥梁检测等工业场景的开源裂缝检测系统。基于山区高速公路隧道衬砌裂缝检测研究思路进行独立工程复现，使用公开数据集与 PyTorch 自研代码，不依赖原论文数据或闭源权重。

> **学术诚信说明**：本仓库为个人工程实践作品，用于展示工业视觉、深度学习与全栈开发能力。不包含原学位论文数据、预训练权重或闭源代码。

---

## 功能特性

- **裂缝目标检测**：基于迁移学习的 Faster R-CNN，支持 ResNet50-FPN / EfficientNet-V2 / VGG16 等多种骨干网络。
- **裂缝语义分割**：CraSAM（SAM 提示适配器）及 U-Net、DeepLabV3+ 基线模型。
- **几何参数提取**：裂缝长度、平均宽度、最大宽度、主方向角度。
- **结构健康评估**：盒计数法计算分形维数，参照技术状况评定标准输出报告。
- **Web 系统**：FastAPI 后端 + React 前端，支持图像上传、可视化与报告导出。
- **边缘部署**：支持 ONNX 导出，便于工控机/边缘网关落地。

---

## 仓库结构

```text
industrial-crack-inspection/
├── backend/          # FastAPI 推理服务
├── configs/          # YAML 训练/推理配置
├── data/             # 数据集、数据加载器、格式说明
├── docs/             # 设计文档、API 文档、思路来源
├── frontend/         # React 可视化前端
├── inference/        # CLI 推理、ONNX 导出
├── models/           # Faster R-CNN、CraSAM、U-Net、DeepLabV3+
├── scripts/          # 数据集划分、模型导出等工具脚本
├── tests/            # 单元测试
├── training/         # 训练与评估脚本
└── utils/            # 图像处理、指标、配置加载
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备数据

将数据集按 `data/README.md` 中的格式放到 `data/processed/` 下，或运行：

```bash
python scripts/prepare_dataset.py --config configs/data/default.yaml
```

### 3. 训练检测模型

```bash
python training/train_detection.py --config configs/detection/faster_rcnn_resnet50.yaml
```

### 4. 训练分割模型

```bash
python training/train_segmentation.py --config configs/segmentation/crasam.yaml
```

### 5. 运行推理

```bash
python inference/predict.py \
    --image assets/demo_crack.jpg \
    --det-weights checkpoints/detection/best.pth \
    --seg-weights checkpoints/segmentation/crasam_best.pth \
    --output outputs/
```

### 6. 启动 Web 演示

```bash
# 终端 1
python backend/main.py

# 终端 2
cd frontend && npm install && npm start
```

打开 http://localhost:3000。

---

## 模型库

| 任务     | 模型            | 骨干网络       | 输入尺寸   | 指标        | 配置文件                                      |
|----------|-----------------|----------------|------------|-------------|-----------------------------------------------|
| 目标检测 | Faster R-CNN    | ResNet50-FPN   | 1024×1024  | 0.807 mAP   | `configs/detection/faster_rcnn_resnet50.yaml` |
| 目标检测 | Faster R-CNN    | EfficientNetV2 | 1024×1024  | 0.793 mAP   | `configs/detection/faster_rcnn_effv2.yaml`    |
| 语义分割 | CraSAM（自研）  | ViT-B/16       | 1024×1024  | 0.890 mIoU  | `configs/segmentation/crasam.yaml`            |
| 语义分割 | U-Net           | VGG13 encoder  | 512×512    | 0.840 mIoU  | `configs/segmentation/unet_vgg13.yaml`        |
| 语义分割 | DeepLabV3+      | ResNet50       | 512×512    | 0.825 mIoU  | `configs/segmentation/deeplabv3plus.yaml`     |

> 指标为内部验证集参考值，实际效果取决于数据分布与训练配置。

---

## 核心算法

### 基于迁移学习的 Faster R-CNN

- 使用 ImageNet 预训练骨干网络提取特征，降低对裂缝小样本的依赖。
- RPN 针对细长裂缝设计 anchor 尺度与宽高比。
- ROI Align + 边界框回归 + 宽度等级分类。
- 支持按 0.1 mm / 0.2 mm / 0.3 mm / ≥0.4 mm 分级训练。

### CraSAM — 面向裂缝的 SAM 适配模型

- 复用预训练 SAM/ViT 图像编码器，冻结其权重。
- 增加可学习的提示适配器与 Mask Decoder。
- 支持全自动分割与点/框交互式提示两种模式。
- 在道路、桥梁、隧道等多场景裂缝数据上验证泛化能力。

### 裂缝参数提取

- 形态学细化获取裂缝骨架，计算长度。
- 距离变换获取裂缝宽度分布。
- PCA 估计裂缝主方向。
- 盒计数法估计分形维数，用于结构健康预测。

---

## 健康评估

参照《公路隧道养护技术规范》（JTG H12-2015）技术状况评定思想，将裂缝映射为技术状况指数（TCI）：

| TCI 范围 | 等级 | 建议                     |
|----------|------|--------------------------|
| ≥90      | 1    | 状况良好，日常巡检       |
| 80–89    | 2    | 轻微缺陷，定期复查       |
| 70–79    | 3    | 中等缺陷，需安排养护     |
| 60–69    | 4    | 严重缺陷，紧急处理       |
| <60      | 5    | 危险状态，立即处置       |

---

## 后续规划

- [x] 项目骨架与模型定义
- [x] Faster R-CNN 训练与评估
- [x] CraSAM 分割适配器
- [x] 裂缝长度/宽度/角度/分形维数提取
- [x] FastAPI 后端 + React 前端
- [x] ONNX 导出
- [ ] TensorRT 推理引擎
- [ ] 边缘网关 Docker + ARM64 部署包
- [ ] 数字孪生可视化集成

---

## 致谢与参考

本项目思路来源于山区高速公路隧道衬砌裂缝检测相关研究，相关学位论文对其车载工业相机采集系统、Faster R-CNN 裂缝检测、CraSAM/SAM 分割方法及隧道健康评估系统进行了详细论述。本仓库为独立代码实现，用于工程能力展示与进一步扩展。

---

## 许可证

MIT License。详见 [LICENSE](LICENSE)。
