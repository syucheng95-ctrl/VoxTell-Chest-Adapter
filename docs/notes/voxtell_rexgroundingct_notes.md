# ReXGroundingCT 方案记录

## 1. 背景与目标

本文档用于记录当前针对 ReXGroundingCT 数据集的调研结论，重点包括两部分：

1. VoxTell 方法拆解，理解其为何能在当前 leaderboard 上领先。
2. 面向比赛场景设计一个比 VoxTell 更适合答辩与展示的改进方案。

当前结论是：VoxTell 是目前最值得学习的公开基线，但它更偏通用的 free-text 3D 分割 foundation model；如果面向比赛而不是纯 benchmark 冲榜，更优策略是构建一个更懂胸部 CT、报告结构和可解释量化的专用系统。

## 2. VoxTell 方法拆解

### 2.1 VoxTell 的任务本质

VoxTell 的目标可以概括为：

- 输入：一整个 3D 医学影像体数据，加上一段自由文本 prompt。
- 输出：与该文本语义对应的 3D segmentation mask。

它不是传统的固定类别分割模型，而是一个 text-to-mask 的 3D 医学视觉语言模型。对于 ReXGroundingCT 这类“报告短语到病灶掩码”的任务，VoxTell 与任务形式天然匹配，因此领先是合理的。

### 2.2 VoxTell 的核心结构

根据论文与官方代码，VoxTell 可以拆成以下四个主要模块：

#### 模块 A：Image Encoder

作用是把输入的 3D CT 体数据编码成多尺度视觉特征。

这一模块承担的不是最终预测，而是提供不同层级的视觉表示：

- 低层特征：边缘、密度变化、局部纹理。
- 中层特征：结构轮廓、局部区域形态。
- 高层特征：病灶模式、器官级语义。

这一步的重要性在于胸部异常具有明显的尺度差异：

- 结节、肿块偏小目标。
- 磨玻璃影、实变、弥漫性病变偏中大尺度模式。
- 纹理性异常往往需要更大感受野和上下文。

因此，如果没有多尺度视觉特征，模型很难同时兼顾局灶病灶和弥漫异常。

#### 模块 B：Prompt Encoder

VoxTell 使用冻结的 Qwen3-Embedding-4B 作为文本嵌入模型，将文本 prompt 映射成高质量语义向量。

这一步的意义在于：

- 它支持自由文本，而不仅是预定义类别名。
- 它对语言变体更鲁棒。
- 它能较好捕捉医学术语之间的语义关系。

例如，“pulmonary nodule”、“subpleural consolidation”、“groundglass opacity”等短语，不再只是标签，而是被表示为可以与视觉空间对齐的语义向量。

#### 模块 C：Prompt Decoder

文本嵌入本身还不足以直接用于分割。VoxTell 进一步通过 Prompt Decoder，把文本语义变换成适合不同视觉尺度融合的多层文本特征。

它的作用可以理解为：

- 将文本语义加工成多尺度查询条件。
- 在粗层级上更适合做病灶区域定位。
- 在细层级上更适合做边界精修与区域筛选。

这一步是 VoxTell 的关键设计之一，因为它不是简单把一句话嵌入后直接拼接，而是让文本在多个层面参与分割推理。

#### 模块 D：Image Decoder

Image Decoder 负责在多个分辨率层面持续融合视觉特征和文本特征，并逐步输出目标 mask。

VoxTell 官方将这一过程概括为多阶段 vision-language fusion，并采用类似 MaskFormer 的 query-image fusion 以及 deep supervision。

翻译成人话，它做的是：

- 在解码过程中多次问图像：“哪里符合这段文本？”
- 每一层都进行视觉-语言对齐，而不是只在最后一层做 late fusion。
- 中间层也受监督，从而让模型更早学习定位。

### 2.3 VoxTell 为什么能在 ReXGroundingCT 上领先

VoxTell 的领先可以从任务适配性上理解：

1. 它本来就是 free-text promptable 3D segmentation 模型。
2. 它可以处理完整临床短语，而不是只能处理固定标签。
3. 它依赖多尺度视觉-语言融合，更适合 3D 异常区域定位。
4. 它使用大规模、多模态、多数据源预训练，具备更好的语义泛化能力。

因此，当任务是“从报告短语中理解异常，再在胸部 CT 中进行空间定位和分割”时，VoxTell 的结构天然比普通分割模型更占优。

### 2.4 VoxTell 的主要弱点

尽管 VoxTell 当前排名靠前，但它并没有形成绝对统治级表现。从 leaderboard 指标看，它仍然存在明显短板。

#### 弱点 1：实例级 precision 偏低

VoxTell 的 Global Dice 和 HIT Rate 尚可，但 Instance Precision 较低，说明它在“找得到”之外，仍有较多误报或区域过扩张的问题。

这意味着：

- 它能覆盖到目标，但不一定找得准。
- 一个文本 prompt 可能激活过多候选区域。
- 预测区域可能比真实病灶更大。

#### 弱点 2：通用模型而非胸部专用模型

VoxTell 是跨模态、跨解剖部位、跨病灶的大型通用模型，这带来强泛化能力，但也意味着：

- 它对胸部 CT 的专门先验利用有限。
- 它未必显式编码肺叶、胸膜、气道树等结构信息。
- 它对胸部报告中的位置关系词未必有最优处理。

例如，ReXGroundingCT 中常见的描述如：

- left lower lobe
- lingular segment
- subpleural
- dependent density
- tree-in-bud

这些词除了语义含义，更隐含强烈的解剖和空间约束。通用 foundation model 未必能显式用好这些信息。

#### 弱点 3：对 diffuse / non-focal 异常建模不足

从 leaderboard 的 per-category 表现可以看出：

- nodules / masses、groundglass opacity、consolidation 等局灶或半局灶异常相对更容易。
- emphysema、septal thickening、bronchiectasis 等弥漫性或纹理性异常明显更难。

这说明 VoxTell 更擅长“目标区域较明确”的文本引导分割，而对边界弱、范围广、模式型的异常仍然不足。

## 3. 从 VoxTell 中可复现的关键思想

如果不直接复刻 VoxTell，而是学习其最有价值的思路，建议保留以下五个部分：

### 3.1 3D 视觉编码器

可选实现：

- nnUNet 风格 encoder-decoder
- 3D ResUNet
- Swin UNETR

建议在比赛里优先选 nnUNet 风格主干，因为：

- 工程稳定。
- 医学图像训练经验成熟。
- 与多尺度解码天然兼容。

### 3.2 文本编码器

若资源有限，不必完全复制 VoxTell 的文本 backbone。可使用：

- ClinicalBERT
- PubMedBERT
- MedCPT
- 其他稳定的 biomedical text encoder

重点不是追求最大，而是确保医学短语语义表达可靠。

### 3.3 文本预处理与结构化 prompt

建议不要把原句生硬输入模型，而是先拆成结构化字段：

- 异常类型
- 解剖位置
- 空间关系
- 形态描述
- focal / diffuse 属性

这一步能显著降低自由文本噪声，并增强 prompt 的可控性。

### 3.4 多尺度视觉-语言融合

建议在 decoder 多层中引入 cross-attention 或 gated fusion，而不是只在 bottleneck 做一次融合。

### 3.5 多级监督

在 coarse、mid、fine 多层输出上施加监督，提升训练稳定性和定位能力。

## 4. 面向比赛的改进方案

### 4.1 设计目标

我们不把目标设为“做一个更大的通用 foundation model”，而是设计一个：

- 更懂胸部 CT 报告语义；
- 更懂解剖结构位置；
- 更适合比赛展示和答辩；
- 更强调可解释性和量化分析的专用系统。

建议方案命名为：

**ChestGrounder++：基于报告语义解析与解剖结构约束的胸部 CT 可解释病灶定位分割系统**

### 4.2 系统整体框架

系统由六个模块组成：

1. Report Parser：报告语义解析
2. Anatomy Prior Branch：解剖结构先验分支
3. Dual-Path Localizer：粗定位与精分割两阶段模块
4. Focal / Diffuse Decoupling：局灶与弥漫分支解耦
5. Instance Filtering：实例级质量控制
6. Explainable Quantification：可解释量化输出

### 4.3 模块一：Report Parser

输入是一条自由文本报告短语，输出为结构化语义 token。

建议拆分为以下五类信息：

- 异常类型 token
- 解剖位置 token
- 空间关系 token
- 形态描述 token
- focal / diffuse token

例如句子：

`subpleural consolidation in the superior segment of the left lower lobe`

可以拆解为：

- lesion type: consolidation
- lung side: left
- lobe: lower lobe
- segment: superior segment
- relation: subpleural
- focality: focal

#### 为什么这样设计

因为 ReXGroundingCT 的本质并不是开放域自然语言理解，而是“临床语义短语到病灶空间位置”的映射。将弱结构化的报告重新拆解为显式结构信息，能够：

- 降低 prompt 噪声；
- 提高位置词和病灶词的利用效率；
- 让后续模型模块更容易做精确对齐；
- 让方法更容易解释和答辩。

### 4.4 模块二：Anatomy Prior Branch

这一分支首先对胸部 CT 做基础结构解析，输出解剖先验图。

建议获得以下结构：

- 左右肺
- 各肺叶
- 胸膜邻近区
- 气道树粗分区

再将这些结构编码为 anatomy prior feature maps，与视觉主干特征和文本 token 联合使用。

#### 为什么这样设计

胸部报告中的大量描述本质上是位置约束：

- left upper lobe
- right middle lobe
- bilateral lower lungs
- subpleural
- peribronchial

如果没有 anatomy prior，模型只能根据纹理或密度变化盲目搜索。加入解剖结构先验后，文本中的位置短语才真正具备空间落点。

这也是相对 VoxTell 最核心的差异化改进之一，因为 VoxTell 更通用，而这里是对胸部报告进行专门建模。

### 4.5 模块三：Dual-Path Localizer

这一部分采用粗定位与精分割两阶段设计。

#### 第一阶段：粗定位

输入：

- 视觉主干特征
- 文本语义 token
- 解剖先验图

输出：

- 病灶 proposal heatmap
- 候选 ROI 区域

#### 第二阶段：精分割

在候选 ROI 内进行高分辨率 refinement，输出更准确的分割边界。

#### 为什么这样设计

它直接针对 VoxTell 的短板：

- 降低整肺乱激活；
- 提高小病灶定位能力；
- 提升实例级 precision；
- 让模型先回答“在哪”，再回答“边界在哪”。

这也符合医生阅片逻辑，更有助于答辩展示。

### 4.6 模块四：Focal / Diffuse Decoupling

根据 Report Parser 的输出，先判断短语属于：

- focal lesion
- diffuse pattern

然后进入不同的建模分支。

#### focal 分支

更偏向：

- 候选区域搜索
- 局部边界 refinement
- 小目标与半局灶病灶识别

典型目标包括：

- nodules / masses
- focal consolidation
- focal GGO

#### diffuse 分支

更偏向：

- 大范围纹理建模
- 区域分布模式分析
- 边界弱异常定位

典型目标包括：

- emphysema
- septal thickening
- bronchiectasis

#### 为什么这样设计

结节和肺气肿本质上不是一类问题。如果强行使用统一 head，模型会倾向折中，难以同时兼顾局灶和弥漫性表现。将二者解耦，不仅合理，而且很容易在方法创新性上得到评委认可。

### 4.7 模块五：Instance Filtering

为减少误报，需要在输出端增加实例级质量控制模块。

建议结合以下三个分数：

- 文本一致性分数
- 位置一致性分数
- mask confidence 分数

利用这三类分数对候选实例进行筛选和重排序，过滤明显不符合文本描述的伪阳性区域。

#### 为什么这样设计

当前 leaderboard 里一个明显问题就是 instance precision 普遍偏低。只要你们在实例筛选上做得更稳，整体指标和展示可信度都有机会明显提升。

### 4.8 模块六：Explainable Quantification

除了输出 mask，还建议输出结构化分析结果：

- 病灶体积
- 所在肺叶或侧别
- 距胸膜关系
- focal / diffuse 类型
- 与文本描述的一致性说明

#### 为什么这样设计

这一步对比赛非常重要。即便 benchmark 核心是分割，你们也可以把系统包装成一个：

- 可解释
- 可量化
- 更接近临床报告工作流

的智能分析系统，从而提升展示度和答辩表现。

## 5. 为什么这个方案比直接套 VoxTell 更适合比赛

### 5.1 它更贴 ReXGroundingCT 的数据特点

ReXGroundingCT 不是普通的开放词汇分割，而是胸部 CT 报告短语到病灶掩码的映射。这个数据集高度依赖：

- 临床文本结构
- 解剖位置信息
- 局灶与弥漫模式差异

因此，胸部专用设计比通用 foundation model 更有针对性。

### 5.2 它更容易做出答辩亮点

评委不只看分数，也看系统完整性、创新点和展示效果。该方案具有天然展示优势：

- 可展示报告解析流程
- 可展示解剖结构先验
- 可展示病灶粗定位到精分割
- 可展示量化指标与可解释输出

### 5.3 它更容易讲清楚创新点

相比“我们使用了一个很大的通用模型”，这个方案可以明确讲出以下创新：

- 临床报告语义结构化
- 解剖结构先验约束
- 局灶/弥漫异质性解耦
- 两阶段定位精分割
- 实例级质量控制与量化输出

## 6. 实现顺序建议

为保证工程可落地，建议按以下顺序实现：

1. 先搭建基础的 3D segmentation backbone + text encoder + fusion module。
2. 跑通最基础的 free-text 3D 分割流程。
3. 加入 Report Parser。
4. 加入 Anatomy Prior Branch。
5. 加入 coarse-to-fine 两阶段模块。
6. 最后补充 Instance Filtering 和 Explainable Quantification。

这样能保证每一阶段都有中间结果，不会一开始把系统做得过于庞杂。

## 7. 关于竞赛任务 1/2/3 是否可以一起做

命题文件中任务 1、2、3 分别是：

1. 影像智能分割
2. 影像量化分析
3. 影像智能分类

从技术上看，三者是可以在一个系统中串联起来的，而且如果包装得好，反而会增强系统完整性。

### 7.1 最合理的关系

建议把三者设计成：

- 分割作为底座
- 量化分析作为中层输出
- 分类作为高层决策输出

具体链条是：

文本或图像输入 -> 病灶分割/定位 -> 结构化量化特征提取 -> 疾病分类或严重程度判断

### 7.2 VoxTell 本身能不能直接覆盖 1/2/3

严格来说，VoxTell 本身主要解决的是 **1. 分割/定位**，而不是完整覆盖 2 和 3。

它可以作为底座，帮助你们做：

- 病灶分割
- 基于文本的病灶定位

但它不天然包含：

- 完整的量化分析模块
- 完整的分类决策模块

不过，VoxTell 的输出 mask 完全可以成为后续量化分析和分类的输入基础。

### 7.3 更适合比赛的整合方式

最推荐的方案不是“把三个任务平行做成三摊”，而是做成一个统一系统：

- 主体亮点：文本引导病灶分割与定位
- 拓展能力：病灶体积、位置、形态量化
- 最终输出：病变类型识别或风险分级

这样做的好处是：

- 题目完整度高；
- 展示上更像真实临床系统；
- 分割、分析、分类三部分逻辑闭环；
- 对答辩现场很友好。

## 8. 当前建议

当前最建议的比赛策略是：

- 以 ReXGroundingCT 为主数据集；
- 以胸部报告驱动的可解释病灶分割为核心创新点；
- 以量化分析和分类作为系统扩展输出；
- 用比 VoxTell 更胸部专用、更结构化的设计形成差异化。

简而言之：

**VoxTell 可以作为学习对象和强基线，但不应作为最终系统形态。更好的比赛方案是：在其 free-text 3D segmentation 思路基础上，补上胸部先验、结构化语义、量化分析和分类闭环。**
