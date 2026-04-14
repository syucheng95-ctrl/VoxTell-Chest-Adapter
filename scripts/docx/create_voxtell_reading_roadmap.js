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

const outputPath = path.resolve(__dirname, "VoxTell阅读路线图.docx");

function p(text, options = {}) {
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
  title: "VoxTell阅读路线图",
  description: "VoxTell foundational reading roadmap",
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
            new TextRun({ text: "VoxTell阅读路线图", bold: true, size: 34, font: "Arial" }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [
            new TextRun({
              text: "按“先懂骨架，再懂query，再回看VoxTell”组织的论文阅读指南",
              size: 22,
              font: "Arial",
            }),
          ],
        }),

        heading("一、这份路线图的目标", HeadingLevel.HEADING_1),
        p("这份阅读路线图不是为了把相关论文全都读一遍，而是为了用最短路径真正读懂 VoxTell 的核心来源。读完后，应该能回答三个问题：第一，VoxTell 的视觉骨架从哪里来；第二，文本为什么不是简单拼接，而是作为 query 参与分割；第三，为什么它适合 ReXGroundingCT 这类 free-text 3D 病灶定位任务。"),
        bullet("先建立 3D 医学分割骨架的直觉，再看文本如何进入模型。"),
        bullet("优先理解结构图、模块连接关系和设计动机，不必一开始陷入全部数学细节。"),
        bullet("对当前项目最重要的是搞清楚哪些思想值得保留，哪些部分后续可以替换或简化。"),

        heading("二、推荐阅读顺序", HeadingLevel.HEADING_1),
        p("建议按下面这个顺序读，前面的论文是后面的前置概念。"),
        bullet("第1层：U-Net -> 建立 encoder-decoder 与 skip connection 的基础直觉。"),
        bullet("第2层：nnU-Net -> 理解医学图像为什么常用 patch-based、预处理和多尺度解码。"),
        bullet("第3层：DETR -> 理解 query、transformer decoder、memory 的基本语言。"),
        bullet("第4层：MaskFormer / Mask2Former -> 理解 query-based segmentation，而不是传统逐像素分类。"),
        bullet("第5层：VoxTell -> 回来看它如何把 3D segmentation backbone、文本 embedding 和 query fusion 组合起来。"),

        heading("三、逐篇阅读建议", HeadingLevel.HEADING_1),

        heading("1. U-Net", HeadingLevel.HEADING_2),
        p("阅读目的：先把分割网络里最基本的 encoder-decoder 结构搞清楚。你不需要从 U-Net 学会所有医学分割知识，但必须看懂它为什么要下采样、为什么要上采样、为什么要有 skip connection。"),
        bullet("重点看：整体结构图，尤其是 contracting path 和 expansive path。"),
        bullet("重点看：skip connection 为什么能帮助恢复空间细节。"),
        bullet("重点看：分割任务和普通分类任务在输出形式上的差别。"),
        bullet("可以先跳过：详细实验设置、当年的数据集细节。"),
        bullet("读完应当得到：VoxTell 的视觉骨架为什么看起来像 3D U-Net / nnU-Net 系。"),

        heading("2. nnU-Net", HeadingLevel.HEADING_2),
        p("阅读目的：理解医学图像分割为什么和自然图像分割不完全一样。VoxTell 的视觉部分虽然不是直接照搬 nnU-Net，但非常明显站在这条医学分割范式上。"),
        bullet("重点看：preprocessing、patch size、sliding window inference、deep supervision 这些医学分割中的工程关键点。"),
        bullet("重点看：为什么医学图像常用 3D patch-based 推理而不是整张图直接进模型。"),
        bullet("重点看：架构自动适配和数据驱动配置的思想，不必执着于每个工程细节。"),
        bullet("可以先跳过：大量 benchmark 表格和不同数据集逐项比较。"),
        bullet("读完应当得到：VoxTell 的视觉 backbone 为什么更像医学分割系统，而不是普通视觉大模型。"),

        heading("3. DETR", HeadingLevel.HEADING_2),
        p("阅读目的：理解 query 这个词到底是什么意思。后面 VoxTell 的文本 prompt 之所以能引导分割，核心语言就来自这条线。"),
        bullet("重点看：object queries 的概念，以及 decoder query 如何去和 image memory 交互。"),
        bullet("重点看：transformer decoder 的输入输出关系，而不是先纠结损失函数。"),
        bullet("重点看：从“分类每个像素”转向“用一组 query 找目标”的思路变化。"),
        bullet("可以先跳过：Hungarian matching 的数学细节和训练稳定性分析。"),
        bullet("读完应当得到：VoxTell 里文本 embedding 为什么更像 segmentation query，而不是普通条件向量。"),

        heading("4. MaskFormer / Mask2Former", HeadingLevel.HEADING_2),
        p("阅读目的：这是理解 VoxTell 最关键的过渡层。DETR 解决的是目标检测，MaskFormer 系列解决的是 query-based segmentation。"),
        bullet("重点看：mask classification 的核心思想。"),
        bullet("重点看：query 如何对应一个 mask，而不是一个类别标签。"),
        bullet("重点看：多尺度特征与 decoder 的关系，尤其是为什么 segmentation 不再只是 per-pixel classification。"),
        bullet("重点看：Mask2Former 里对多尺度和 masked attention 的改进思路。"),
        bullet("可以先跳过：不同 benchmark 上的大量实验比较。"),
        bullet("读完应当得到：VoxTell 的“文本到 mask”不是一句话拼接进网络，而是 query 驱动的分割过程。"),

        new Paragraph({ children: [new PageBreak()] }),

        heading("5. VoxTell", HeadingLevel.HEADING_2),
        p("阅读目的：在理解前面几篇的基础上，重新看 VoxTell 的结构，你会发现它不是凭空出现的新模型，而是把三条成熟路线组合到了一起。"),
        bullet("重点看：整体架构图。先把 Image Encoder、Prompt Encoder、Prompt Decoder、Image Decoder 四块位置认清。"),
        bullet("重点看：文本 embedding 是如何投影到 query 空间，并与深层图像特征做 transformer 融合的。"),
        bullet("重点看：融合后的 mask embedding 如何再次投影回 decoder 多个尺度。"),
        bullet("重点看：为什么它天然适合 ReXGroundingCT 这类自由文本病灶分割任务。"),
        bullet("可以先跳过：一开始不必深究所有实现细节，先把“结构图和信息流”看懂。"),
        bullet("读完应当得到：VoxTell = 3D医学分割骨架 + 大文本 embedding + query-based segmentation fusion。"),

        heading("四、每篇论文最该盯的图和模块", HeadingLevel.HEADING_1),
        bullet("U-Net：盯整体结构图，尤其是左边编码、右边解码、横向 skip 的连接。"),
        bullet("nnU-Net：盯 pipeline 图和 inference / preprocessing 相关图表。"),
        bullet("DETR：盯 encoder-decoder 与 object query 的示意图。"),
        bullet("MaskFormer / Mask2Former：盯 query 到 mask 的流程图，以及多尺度融合部分。"),
        bullet("VoxTell：盯四模块架构图，以及 text prompt 如何进入 transformer decoder 和 decoder stages。"),

        heading("五、哪些地方可以先不深究", HeadingLevel.HEADING_1),
        bullet("损失函数的所有公式推导。"),
        bullet("所有 benchmark 表格和每个数据集上的细节对比。"),
        bullet("训练技巧里的所有超参数。"),
        bullet("实现层面的每一行代码。先懂结构，再回头看代码。"),

        heading("六、最小必要阅读集", HeadingLevel.HEADING_1),
        p("如果时间不够，至少读下面 5 篇。"),
        bullet("U-Net"),
        bullet("nnU-Net"),
        bullet("DETR"),
        bullet("MaskFormer 或 Mask2Former"),
        bullet("VoxTell"),

        heading("七、面向当前项目的阅读目标", HeadingLevel.HEADING_1),
        p("你当前的目标不是成为这些方向的理论专家，而是为 ReXGroundingCT 项目建立足够清晰的技术判断。也就是说，读完之后要能回答：第一，VoxTell 里哪些模块值得保留；第二，哪些模块成本太高可以替换；第三，如果做胸部CT专用方案，视觉 backbone、文本编码器和融合模块分别该怎么取舍。"),
        bullet("读懂视觉 backbone：决定后续是否保留 encoder 或 encoder-decoder。"),
        bullet("读懂 query-based segmentation：决定是否继续走 prompt-guided 路线。"),
        bullet("读懂文本侧：决定是否继续用大文本 embedding，还是换成更轻量的医学文本模型。"),

        heading("八、一句话总结", HeadingLevel.HEADING_1),
        p("读懂 VoxTell 的最短路径，不是直接死磕它的代码，而是先用 U-Net 和 nnU-Net 建立 3D 医学分割直觉，再用 DETR 和 MaskFormer 建立 query-based segmentation 直觉，最后回到 VoxTell，把它理解成一个建立在医学分割骨架之上的文本引导 3D 分割模型。"),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(outputPath);
});
