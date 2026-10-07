# LabelBench Figure 1 复现

本项目用于复现论文 *LabelBench: A Framework for Benchmarking Label-Efficient Learning*
的 Figure 1(a)：在 CIFAR-10 上结合 CLIP ViT-B/32、FlexMatch 和多种主动学习算法，
比较不同标签预算下的测试准确率。

这是一个课程作业项目。考虑到 ImageNet 数据量很大，并且相关实验需要更多磁盘空间、
内存、GPU 显存和训练时间，本项目明确不复现 Figure 1(b) 和 Figure 1(c)。项目范围
包含 CIFAR-10 的 Figure 1(a) 和新增的 Figure 5(a/b/c)。Figure 5 的独立训练入口、
绘图入口和输出文件说明见 [FIGURE5.md](FIGURE5.md)。

## 目录结构

```text
UCPH_ATDL_2026_Group16_Assignment2_LabelBench/
├── data/                   # CIFAR-10 和 CIFAR-100
├── model/                  # CLIP ViT-B/32 预训练权重
├── scripts/
│   ├── prepare_assets.py   # 下载源码和模型，检查数据
│   ├── run_fig1a.py        # Figure 1(a) 实验入口
│   └── plot_fig1a.py       # Figure 1(a) 绘图入口
├── results/                # 训练指标、检查点和最终图片
├── requirements.txt
└── README.md
```

LabelBench 官方源码存放在隐藏目录 `.labelbench/`。其中已经包含 FlexMatch 和全部
主动学习算法，无需在本项目中重复复制它们。

## Figure 1 的范围

| 子图 | 数据集 | 指标 | 当前状态 |
|---|---|---|---|
| Figure 1(a) | CIFAR-10 | 测试集 Top-1 accuracy | 本作业复现目标 |
| Figure 1(b) | ImageNet | 测试集 Top-1 accuracy | 不纳入本作业 |
| Figure 1(c) | ImageNet | 未标注池预测准确率 | 不纳入本作业 |

本作业只对 Figure 1(a) 的复现结果负责。Figure 1(b) 和 Figure 1(c) 不作为缺失实验，
也不属于后续待完成事项。

## Figure 1(a) 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 训练池 | 官方训练集50,000张图片 |
| 验证集和测试集 | 官方10,000张测试图片按 NumPy seed 42 固定分成5,000/5,000 |
| 模型 | OpenAI CLIP ViT-B/32 |
| 模型训练方式 | 端到端微调图像编码器和10类分类头 |
| 半监督算法 | FlexMatch |
| 主动学习算法 | Random、Confidence、Entropy、Margin、CORESET、GALAXY、BADGE、BAIT |
| 初始标注数量 | 1,000 |
| 每轮增加标签 | 1,000 |
| 标签预算范围 | 1,000、2,000、…、10,000 |
| 重复次数 | 每种方法1次（seed 1234） |
| 总实验数 | 8种方法 × 1次 = 8 |
| 汇总方式 | 展示单次实验结果，不计算标准误 |

### FlexMatch 超参数

| 项目 | 设置 |
|---|---|
| 优化器 | AdamW |
| 学习率 | `1e-5` |
| Weight decay | `3e-5` |
| 学习率调度 | Cosine，500 warm-up steps |
| 每轮最大训练时间 | 20 epochs |
| Early stopping | patience 3 |
| 有标签 batch size | 64 |
| 无标签比例 | 每个有标签样本对应3个无标签样本 |
| FlexMatch 基础阈值 | 0.95 |
| 无监督损失权重 | 1.0 |
| 强增强 | RandAugment `(3, 5)` |

FlexMatch 为无标签图片生成弱增强和强增强两个版本。模型根据弱增强图片产生伪标签，
再用伪标签监督强增强图片。FlexMatch 还会根据各类别已经积累的可靠伪标签数量，动态
调整不同类别的接受阈值。

## 已准备的本地资源

```text
.labelbench/               # 固定版本的官方 LabelBench 源码
data/
├── cifar-10-batches-py/   # CIFAR-10 解压数据
├── cifar-10-python.tar.gz
├── cifar-100-python/      # CIFAR-100 解压数据
└── cifar-100-python.tar.gz
model/
└── ViT-B-32.pt            # OpenAI CLIP ViT-B/32 权重
```

| 组件 | 本地位置 | 作用 |
|---|---|---|
| CIFAR-10 | `data/` | Figure 1(a) 数据集 |
| CIFAR-100 | `data/` | 为后续实验保留 |
| CLIP ViT-B/32 | `model/ViT-B-32.pt` | 预训练视觉模型 |
| FlexMatch | `.labelbench/LabelBench/trainer/` | 半监督训练 |
| 主动学习算法 | `.labelbench/LabelBench/strategy/` | 选择下一批需要标注的图片 |

## 安装环境

作者推荐 Python 3.9，但没有发布完整的历史依赖锁：

```bash
cd /path/to/UCPH_ATDL_2026_Group16_Assignment2_LabelBench
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 重新准备资源

如果源码、数据或模型权重丢失，可以运行：

```bash
python scripts/prepare_assets.py
```

该脚本不训练模型，只会：

1. 下载 LabelBench 官方源码；
2. 固定到 commit `9e23393ba5dd48fc7c23293e441337a5e45a3edf`；
3. 获得 FlexMatch 和8种主动学习算法的官方实现；
4. 下载 OpenAI CLIP ViT-B/32 权重到 `model/`；
5. 离线加载并检查 CIFAR-10 和 CIFAR-100。

## 为什么本作业不复现 ImageNet 实验

Figure 1(b) 和 Figure 1(c) 使用 ImageNet。本作业经过范围和计算成本评估后，决定
放弃这两个子图，只复现使用 CIFAR-10 的 Figure 1(a)。原因如下：

1. **存储成本高**：ImageNet ILSVRC 2012 包含约128万张训练图片和5万张验证图片，
   下载包和解压后的数据需要大量磁盘空间。
2. **内存和显存需求高**：在大型图像池上使用 CLIP ViT-B/32、FlexMatch 和主动学习
   策略，需要保存或计算大量预测、特征及梯度表示。CORESET、BADGE、BAIT 和
   GALAXY 等选择算法还会增加内存消耗。
3. **训练时间过长**：论文实验需要在多个标签预算下反复微调模型，并对每种主动学习
   方法进行多次随机重复。ImageNet 的总体计算量明显超出本课程作业的合理预算。
4. **获取方式受限**：ImageNet 要求使用者注册、接受使用条款并获得数据访问权限，
   不能像 CIFAR 一样由 torchvision 从公开地址直接下载。

CIFAR-10 和 CIFAR-100 提供公开下载地址，torchvision 可以直接下载，并使用官方
MD5 验证文件。ImageNet 要求使用者先在 ImageNet 官网注册、接受使用条款并取得
访问权限。程序不能代表使用者注册、接受数据许可，也不能绕过权限重新分发数据。

因此，`prepare_assets.py` 不包含 ImageNet 下载或检查逻辑，项目代码也不提供
Figure 1(b) 和 Figure 1(c) 的运行入口。缩小范围后，我们可以把计算资源集中在
CIFAR-10 上，更完整地比较8种主动学习方法，并完成4次重复实验和标准误分析。

## 来源与复现边界

- 论文：LabelBench Figure 1、Sections 4.1-4.3 和 Appendix D；
- 官方代码：<https://github.com/EfficientTraining/LabelBench>；
- 数据：torchvision 官方 CIFAR-10/CIFAR-100；
- 预训练模型：OpenAI CLIP ViT-B/32。

作者没有锁定全部历史依赖版本，也没有提供完整的历史运行环境。因此，可以按照官方
代码和超参数重建实验，但不同 GPU、CUDA 和依赖版本不保证逐位得到完全相同的数值。
