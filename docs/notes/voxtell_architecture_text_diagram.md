# VoxTell 文字版结构图

本文档用于把 VoxTell 的整体架构按“输入到输出”的顺序梳理清楚，便于后续写技术报告、申请书和答辩说明。

## 1. 总体流程

```text
输入1：3D医学影像体数据（CT / MRI / PET）
输入2：自由文本 prompt
        ↓
[A] 图像读取与预处理
        ↓
[B] Image Encoder：3D视觉编码器提取多尺度图像特征
        ↓
[C] Prompt Encoder：大语言文本嵌入模型将 prompt 编成语义向量
        ↓
[D] Prompt Decoder / Fusion Module：
    用 transformer decoder 做文本-图像跨模态融合
        ↓
[E] Image Decoder：
    在多个尺度上用文本引导图像解码，逐步恢复空间分辨率
        ↓
[F] Segmentation Head：
    为每个文本 prompt 输出一张3D mask
        ↓
输出：与文本语义对应的3D分割结果
```

## 2. 从输入到输出的逐层解释

### Step 1. 输入阶段

```text
输入影像：一整个3D体数据
例如：胸部CT NIfTI文件

输入文本：一句自然语言提示
例如：
- "liver"
- "right kidney"
- "ground glass opacity in the right lower lobe"
```

VoxTell 不是固定类别分割模型，而是“给一句话，找对应区域”的模型。

---

### Step 2. 图像预处理

```text
原始3D图像
  ↓
重定向到统一方向（RAS）
  ↓
裁剪非零区域
  ↓
Z-score归一化
  ↓
得到预处理后的3D张量
```

这一步的作用是：

- 统一影像方向，避免左右、前后、上下关系错乱
- 去掉无效背景，减少计算量
- 标准化强度分布，提高推理稳定性

---

### Step 3. Image Encoder：图像编码器

```text
预处理后的3D图像
  ↓
Stem卷积层
  ↓
Stage 1
  ↓
Stage 2
  ↓
Stage 3
  ↓
Stage 4
  ↓
Stage 5
  ↓
输出多尺度视觉特征（skip features）
```

这部分本质上是一个 3D residual encoder，风格接近 nnU-Net / 3D U-Net 的医学图像编码器。

它负责学习：

- 低层纹理和边缘信息
- 中层结构信息
- 高层语义和病灶模式

输出不是一个单一特征，而是一组多尺度特征，供后续 decoder 使用。

---

### Step 4. Prompt Encoder：文本编码器

```text
自然语言 prompt
  ↓
包装为 instruction-style 文本
  ↓
Tokenizer
  ↓
Qwen3-Embedding-4B
  ↓
得到文本语义向量
```

例如：

```text
输入 prompt:
"ground glass opacity in the right lower lobe"

会先包装成类似：
"Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents
Query: ground glass opacity in the right lower lobe"
```

然后送入 Qwen3-Embedding-4B，得到高维文本 embedding。

这一步的作用是：

- 理解自然语言中的病灶类型
- 理解位置描述
- 理解形态和范围描述

---

### Step 5. 选择图像 bottleneck 特征

```text
多尺度图像特征
  ↓
选取一个较深层的 feature map
  ↓
线性投影到 query 空间
  ↓
作为 transformer decoder 的 memory
```

这一步可以理解为：

- 图像编码器已经提取出高层视觉语义
- 从中选一个合适层级作为“图像记忆库”
- 让文本去查询这个图像语义空间

---

### Step 6. Prompt Decoder / Fusion Module：文本-图像融合

```text
文本 embedding
  ↓
project_text_embed
  ↓
作为 query

图像 bottleneck feature
  ↓
project_bottleneck_embed
  ↓
作为 memory

query + memory
  ↓
Transformer Decoder
  ↓
得到 mask embedding
```

这是 VoxTell 的关键模块。

它不是简单拼接文本和图像特征，而是让文本作为 query，去和图像 memory 做 cross-attention。

可以把它理解成：

```text
“根据这句话，到图像里问：哪里最符合这个描述？”
```

输出的 `mask embedding` 本质上是：

- 已经融合了文本语义
- 并且和图像空间对齐过的查询结果

---

### Step 7. 将融合结果投影回 decoder 多个尺度

```text
mask embedding
  ↓
project_to_decoder_channels
  ↓
生成多个尺度的条件向量
  ↓
分别送到 decoder 各个阶段
```

这一步意味着：

- 文本信息不是只在最后一层起作用
- 而是以多尺度方式参与后续分割解码

这样做的好处是：

- 粗尺度有助于定位
- 细尺度有助于边界恢复

---

### Step 8. Image Decoder：图像解码器

```text
最深层图像特征
  ↓
上采样
  ↓
与浅层 skip feature 融合
  ↓
叠加文本引导信息
  ↓
再上采样
  ↓
再融合
  ↓
……
  ↓
恢复到更高空间分辨率
```

这一部分本质上类似 U-Net decoder，但它不是纯视觉解码，而是“被文本条件调制的解码过程”。

因此它在每一层都在回答一个问题：

```text
“当前恢复出来的这个区域，是否符合这句文本描述？”
```

---

### Step 9. Segmentation Head：输出分割掩码

```text
decoder输出特征
  ↓
segmentation layer
  ↓
logits
  ↓
sigmoid
  ↓
threshold > 0.5
  ↓
二值3D mask
```

如果输入多个 prompt，则输出多个 mask：

```text
Prompt 1 -> Mask 1
Prompt 2 -> Mask 2
Prompt 3 -> Mask 3
```

最终结果是一个或多个与文本语义对应的三维分割掩码。

---

## 3. 更精简的一行版结构图

```text
3D图像
  → 预处理
  → 3D视觉编码器
  → 选取深层视觉特征

文本prompt
  → tokenizer
  → Qwen3-Embedding-4B
  → 文本语义向量

图像深层特征 + 文本语义向量
  → Transformer Decoder 融合
  → 生成 mask embedding
  → 多尺度投影到 decoder
  → 文本引导的3D解码
  → 输出对应 prompt 的3D分割mask
```

---

## 4. 一句话本质总结

```text
VoxTell = 3D医学图像分割骨干 + 大文本嵌入模型 + Transformer跨模态融合 + 多尺度文本引导解码
```

它的核心不是“固定类别分割”，而是：

```text
把自然语言描述映射成与影像空间对齐的3D分割结果
```

