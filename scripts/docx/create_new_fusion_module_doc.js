const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  PageBreak,
} = require("docx");

const outputPath = path.resolve(__dirname, "胸部专用融合模块文字版结构图.docx");

function para(text, options = {}) {
  return new Paragraph({
    spacing: { after: options.after ?? 120, before: options.before ?? 0, line: 360 },
    alignment: options.alignment ?? AlignmentType.LEFT,
    children: [
      new TextRun({
        text,
        bold: options.bold ?? false,
        size: options.size ?? 24,
        font: "Arial",
      }),
    ],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 340 },
    children: [new TextRun({ text, size: 23, font: "Arial" })],
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: 180, after: 120 },
    children: [new TextRun({ text, font: "Arial", bold: true })],
  });
}

const doc = new Document({
  creator: "OpenAI Codex",
  title: "胸部专用融合模块文字版结构图",
  description: "Chest-specific fusion module text diagram based on VoxTell-inspired redesign",
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 24 },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 34, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: "bullet",
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [
            new TextRun({ text: "胸部专用融合模块文字版结构图", bold: true, size: 34, font: "Arial" }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [
            new TextRun({
              text: "基于 VoxTell 启发的胸部CT文本引导病灶定位分割模块设计",
              size: 22,
              font: "Arial",
            }),
          ],
        }),

        heading("一、设计目标", HeadingLevel.HEADING_1),
        para("本设计的目标不是简单沿用 VoxTell 的统一文本融合策略，而是面向胸部 CT 报告中病灶类型、解剖位置和形态描述高度耦合的特点，重新设计一个更适合 ReXGroundingCT 的胸部专用跨模态融合模块。"),
        bullet("保留 VoxTell 在三维医学图像分割和文本引导方面的总体思想。"),
        bullet("不再只使用单一整句 embedding 去驱动所有尺度的分割过程。"),
        bullet("强调位置先验、病灶类型语义和边界细化在不同尺度上的不同作用。"),

        heading("二、整体结构图", HeadingLevel.HEADING_1),
        para("下面给出推荐的新融合模块的文字版结构图。"),
        para("输入1：胸部CT 3D体数据", { after: 60 }),
        para("输入2：自由文本报告短语", { after: 60 }),
        para("        ↓", { after: 30 }),
        para("[A] 图像支路：3D视觉编码器", { after: 60 }),
        para("        ↓", { after: 30 }),
        para("输出多尺度视觉特征：F_coarse, F_mid, F_fine", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("[B] 文本支路：结构化 prompt 编码", { after: 60 }),
        para("原始文本", { after: 20 }),
        para("→ 异常类型 token", { after: 20 }),
        para("→ 解剖部位 token", { after: 20 }),
        para("→ 空间位置 token", { after: 20 }),
        para("→ 形态描述 token", { after: 20 }),
        para("→ 范围属性 token（局灶 / 弥漫）", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("轻量化医学文本模型 / 结构化嵌入层", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("输出文本语义向量组：E_lesion, E_anatomy, E_location, E_morphology, E_extent", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("[C] 解剖先验支路", { after: 60 }),
        para("肺区 / 肺叶 / 胸膜下区域 / 左右肺位置编码", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("输出解剖先验特征：P_anatomy", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("[D] 多尺度胸部专用融合模块", { after: 60 }),
        para("粗尺度融合：F_coarse + E_anatomy + E_location + P_anatomy", { after: 20 }),
        para("→ anatomy-guided coarse localization", { after: 80 }),
        para("中尺度融合：F_mid + E_lesion + E_location + P_anatomy", { after: 20 }),
        para("→ lesion-aware semantic fusion", { after: 80 }),
        para("细尺度融合：F_fine + E_morphology + E_extent + P_anatomy", { after: 20 }),
        para("→ morphology-aware boundary refinement", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("[E] 局灶 / 弥漫双分支解码", { after: 60 }),
        para("根据 E_extent 选择不同的门控策略", { after: 100 }),
        para("        ↓", { after: 30 }),
        para("[F] 分割输出 + 实例过滤 + 量化分析", { after: 60 }),
        para("输出：病灶3D mask、体积、范围、位置和标准化结果", { after: 120 }),

        heading("三、模块含义解释", HeadingLevel.HEADING_1),

        heading("1. 图像支路", HeadingLevel.HEADING_2),
        para("图像支路沿用 VoxTell 启发下的三维视觉主干，优先复用预训练视觉端能力。它负责提取从粗到细的胸部 CT 多尺度特征。"),
        bullet("粗尺度特征更适合定位病灶大致区域。"),
        bullet("中尺度特征更适合识别病灶类型和分布模式。"),
        bullet("细尺度特征更适合恢复边界。"),

        heading("2. 结构化 prompt 编码", HeadingLevel.HEADING_2),
        para("与 VoxTell 直接用整句 embedding 不同，这里把胸部报告短语拆成多种语义成分。"),
        bullet("异常类型用于回答“是什么病灶”。"),
        bullet("解剖部位和空间位置用于回答“在哪里”。"),
        bullet("形态描述用于回答“长什么样”。"),
        bullet("范围属性用于区分局灶病灶和弥漫病灶。"),

        heading("3. 解剖先验支路", HeadingLevel.HEADING_2),
        para("胸部报告里的位置词具有很强的先验含义，例如 right lower lobe、subpleural、bilateral 等。这些信息单靠整句语义不一定能稳定表达，因此额外加入解剖先验支路。"),
        bullet("帮助模型把不合理的响应区域抑制掉。"),
        bullet("提高位置相关 prompt 的可靠性。"),
        bullet("增强胸部专用性。"),

        new Paragraph({ children: [new PageBreak()] }),

        heading("4. 多尺度胸部专用融合模块", HeadingLevel.HEADING_2),
        para("这是相对于 VoxTell 最核心的改动。原始 VoxTell 更偏“单一文本 embedding 先和深层图像特征融合，再广播到多个 decoder 阶段”。新模块则强调不同语义成分在不同尺度上发挥不同作用。"),
        bullet("粗尺度重点使用 anatomy 和 location 语义，先保证找对区域。"),
        bullet("中尺度重点使用 lesion 语义，增强病灶识别能力。"),
        bullet("细尺度重点使用 morphology 和 extent 语义，提升边界与范围表达。"),

        heading("5. 局灶 / 弥漫双分支解码", HeadingLevel.HEADING_2),
        para("胸部病灶并不是同一种空间模式。小结节、钙化灶、局灶磨玻璃影和肺气肿、弥漫性病变在空间分布上差异很大，因此推荐在解码阶段做局灶与弥漫的差异化处理。"),
        bullet("局灶分支更强调小目标和边界精确性。"),
        bullet("弥漫分支更强调区域覆盖和模式识别。"),

        heading("四、和 VoxTell 原始融合逻辑的差别", HeadingLevel.HEADING_1),
        bullet("VoxTell：整句文本 embedding 为主。"),
        bullet("新方案：结构化 prompt，多种语义成分分别参与融合。"),
        bullet("VoxTell：从单个深层层级开始融合，再传播到 decoder。"),
        bullet("新方案：不同尺度直接承接不同语义重点。"),
        bullet("VoxTell：统一融合逻辑。"),
        bullet("新方案：胸部专用融合 + 解剖先验 + 局灶 / 弥漫差异化策略。"),

        heading("五、推荐对外表述", HeadingLevel.HEADING_1),
        para("相较于 VoxTell 采用单一文本 embedding 与统一跨模态融合的策略，我们针对胸部 CT 报告中病灶类型、解剖位置与形态描述高度耦合的特点，设计了结构化 prompt 驱动的多尺度胸部专用融合模块，使不同语义成分在不同空间尺度上参与病灶定位、区域筛选与边界细化。"),

        heading("六、一句话总结", HeadingLevel.HEADING_1),
        para("新融合模块的核心思想是：不再让整句文本统一控制全部分割过程，而是让“位置、病灶类型、形态、范围”这些语义在不同尺度上分工合作，并结合胸部解剖先验完成更适合 ReXGroundingCT 的病灶定位分割。"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
});
