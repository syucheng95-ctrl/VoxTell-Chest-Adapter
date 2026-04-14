const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  ShadingType,
  PageOrientation,
} = require("docx");

const ROOT = __dirname;
const METRICS_PATH = path.join(ROOT, "outputs", "voxtell_val_metrics.json");
const OUT_DOCX = path.join(ROOT, "VoxTell_full_val_report.docx");
const OUT_MD = path.join(ROOT, "VoxTell_full_val_report.md");

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function fmt(num, digits = 4) {
  return Number(num).toFixed(digits);
}

function delta(a, b) {
  return b - a;
}

function para(text, options = {}) {
  return new Paragraph({
    ...options,
    children: [new TextRun(text)],
  });
}

function cell(text, width, shaded = false) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: { top: border, bottom: border, left: border, right: border },
    shading: shaded ? { fill: "D9EAF4", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [para(String(text))],
  });
}

function makeTable(headers, rows, widths) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        children: headers.map((h, i) => cell(h, widths[i], true)),
      }),
      ...rows.map((row) =>
        new TableRow({
          children: row.map((v, i) => cell(v, widths[i], false)),
        }),
      ),
    ],
  });
}

function bullet(text) {
  return new Paragraph({
    text,
    bullet: { level: 0 },
  });
}

function extractCaseRows(data) {
  const rows = [];
  for (const caseItem of data.cases) {
    for (const item of caseItem.items) {
      rows.push({
        case: caseItem.case,
        category: item.category_name,
        lesion: item.lesion_name,
        prompt: item.raw_prompt,
        rawDice: item.raw.dice,
        structuredDice: item.structured.dice,
        rawPrecision: item.raw.precision,
        structuredPrecision: item.structured.precision,
        rawRecall: item.raw.recall,
        structuredRecall: item.structured.recall,
        deltaDice: item.structured.dice - item.raw.dice,
      });
    }
  }
  rows.sort((a, b) => a.deltaDice - b.deltaDice);
  return rows;
}

function topGroupRows(groupObj, topN = 12) {
  return Object.entries(groupObj)
    .map(([name, stats]) => ({
      name,
      count: stats.raw.count,
      rawDice: stats.raw.mean_dice,
      structuredDice: stats.structured.mean_dice,
      deltaDice: stats.structured.mean_dice - stats.raw.mean_dice,
      rawMicro: stats.raw.micro_dice,
      structuredMicro: stats.structured.micro_dice,
      rawPrecision: stats.raw.mean_precision,
      structuredPrecision: stats.structured.mean_precision,
      rawRecall: stats.raw.mean_recall,
      structuredRecall: stats.structured.mean_recall,
    }))
    .sort((a, b) => b.count - a.count || b.deltaDice - a.deltaDice)
    .slice(0, topN);
}

function buildNarrative(data) {
  const overallRaw = data.aggregate.raw;
  const overallStructured = data.aggregate.structured;
  const categoryRows = topGroupRows(data.by_category, 20);
  const lesionRows = topGroupRows(data.by_lesion, 20);
  const caseRows = extractCaseRows(data);
  const best = caseRows.slice(-8).reverse();
  const worst = caseRows.slice(0, 8);

  const overallConclusion =
    overallStructured.mean_dice > overallRaw.mean_dice
      ? "structured 在 mean Dice 上略优，但总体优势有限。"
      : "structured 在 full-val 整体上没有超过 raw，当前版本不适合作为默认替代输入。";

  const reasons = [
    "structured prompt 把自然语言压缩成固定字段后，确实提升了语义约束，但也丢失了原句里的上下文细节和描述强度。",
    "对局灶、位置明确的 finding，结构化字段通常更有利，因为 lesion type、laterality、anatomy、location 能直接提供空间先验。",
    "对范围大、边界弱、语义带模糊程度或病程变化的 finding，模板化表达容易把有效语义过度压缩，导致召回下降。",
    "当前模板里的字段覆盖仍然不完整，尤其是 extent、distribution、severity、comparison、texture pattern 这类信息没有被稳定保留下来。",
    "structured 目前的收益更像是『按类型选择性启用』，而不是『对所有 prompt 一刀切替换』。",
  ];

  const summaryParagraphs = [
    `本报告基于 ReXGroundingCT public val 全部 50 个 case、共 115 个 finding，对 VoxTell 在 raw prompt 和 structured prompt 两种输入方式下的表现进行了对比。`,
    `总体结果显示：raw 的 mean Dice 为 ${fmt(overallRaw.mean_dice)}，structured 为 ${fmt(overallStructured.mean_dice)}；raw 的 micro Dice 为 ${fmt(overallRaw.micro_dice)}，structured 为 ${fmt(overallStructured.micro_dice)}。${overallConclusion}`,
  ];

  return { categoryRows, lesionRows, best, worst, reasons, summaryParagraphs };
}

function buildMarkdown(data, narrative) {
  const lines = [];
  lines.push("# VoxTell on ReXGroundingCT Public Val");
  lines.push("");
  for (const p of narrative.summaryParagraphs) lines.push(p, "");
  lines.push("## Overall");
  lines.push("");
  lines.push(`- Cases: 50`);
  lines.push(`- Findings: ${data.aggregate.raw.count}`);
  lines.push(`- Raw mean Dice: ${fmt(data.aggregate.raw.mean_dice)}`);
  lines.push(`- Structured mean Dice: ${fmt(data.aggregate.structured.mean_dice)}`);
  lines.push(`- Raw micro Dice: ${fmt(data.aggregate.raw.micro_dice)}`);
  lines.push(`- Structured micro Dice: ${fmt(data.aggregate.structured.micro_dice)}`);
  lines.push("");
  lines.push("## Conclusion");
  lines.push("");
  lines.push("- Structured prompt is not a stable global replacement for raw prompt.");
  lines.push("- It helps some focal and location-constrained findings, but hurts a substantial share of findings, especially those requiring broader contextual wording.");
  lines.push("");
  lines.push("## Reasons");
  lines.push("");
  for (const r of narrative.reasons) lines.push(`- ${r}`);
  lines.push("");
  return lines.join("\n");
}

async function main() {
  const data = loadJson(METRICS_PATH);
  const narrative = buildNarrative(data);
  fs.writeFileSync(OUT_MD, buildMarkdown(data, narrative), "utf8");

  const overall = data.aggregate;

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { size: 30, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 200, after: 160 }, outlineLevel: 0 },
        },
        {
          id: "Heading2",
          name: "Heading 2",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { size: 26, bold: true, font: "Arial" },
          paragraph: { spacing: { before: 160, after: 120 }, outlineLevel: 1 },
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE },
            margin: { top: 1000, right: 900, bottom: 1000, left: 900 },
          },
        },
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 240 },
            children: [new TextRun({ text: "VoxTell Full-Val Raw vs Structured Report", bold: true, size: 34, font: "Arial" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { after: 320 },
            children: [new TextRun(`Generated on 2026-04-12`)],
          }),
          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Summary")] }),
          ...narrative.summaryParagraphs.map((t) => para(t)),
          bullet("Dataset: ReXGroundingCT public val, 50 cases, 115 findings."),
          bullet("Comparison target: same VoxTell checkpoint and inference pipeline, only raw prompt vs structured prompt changed."),
          bullet("Evaluation metrics: Dice, IoU, Precision, Recall, including mean and micro summaries."),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Overall Metrics")] }),
          makeTable(
            ["Mode", "Count", "Mean Dice", "Mean IoU", "Mean Precision", "Mean Recall", "Micro Dice", "Micro IoU"],
            [
              [
                "Raw",
                overall.raw.count,
                fmt(overall.raw.mean_dice),
                fmt(overall.raw.mean_iou),
                fmt(overall.raw.mean_precision),
                fmt(overall.raw.mean_recall),
                fmt(overall.raw.micro_dice),
                fmt(overall.raw.micro_iou),
              ],
              [
                "Structured",
                overall.structured.count,
                fmt(overall.structured.mean_dice),
                fmt(overall.structured.mean_iou),
                fmt(overall.structured.mean_precision),
                fmt(overall.structured.mean_recall),
                fmt(overall.structured.micro_dice),
                fmt(overall.structured.micro_iou),
              ],
              [
                "Delta",
                "",
                fmt(delta(overall.raw.mean_dice, overall.structured.mean_dice)),
                fmt(delta(overall.raw.mean_iou, overall.structured.mean_iou)),
                fmt(delta(overall.raw.mean_precision, overall.structured.mean_precision)),
                fmt(delta(overall.raw.mean_recall, overall.structured.mean_recall)),
                fmt(delta(overall.raw.micro_dice, overall.structured.micro_dice)),
                fmt(delta(overall.raw.micro_iou, overall.structured.micro_iou)),
              ],
            ],
            [1800, 900, 1200, 1200, 1400, 1200, 1200, 1200],
          ),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("By Category")] }),
          makeTable(
            ["Category", "Count", "Raw Mean Dice", "Structured Mean Dice", "Delta", "Raw Micro Dice", "Structured Micro Dice"],
            narrative.categoryRows.map((r) => [
              r.name,
              r.count,
              fmt(r.rawDice),
              fmt(r.structuredDice),
              fmt(r.deltaDice),
              fmt(r.rawMicro),
              fmt(r.structuredMicro),
            ]),
            [2900, 800, 1300, 1500, 1000, 1300, 1500],
          ),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("By Lesion")] }),
          makeTable(
            ["Lesion", "Count", "Raw Mean Dice", "Structured Mean Dice", "Delta", "Raw Micro Dice", "Structured Micro Dice"],
            narrative.lesionRows.map((r) => [
              r.name,
              r.count,
              fmt(r.rawDice),
              fmt(r.structuredDice),
              fmt(r.deltaDice),
              fmt(r.rawMicro),
              fmt(r.structuredMicro),
            ]),
            [2500, 800, 1300, 1500, 1000, 1300, 1500],
          ),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Typical Wins for Structured")] }),
          makeTable(
            ["Case", "Category", "Lesion", "Raw Dice", "Structured Dice", "Delta", "Prompt"],
            narrative.best.map((r) => [
              r.case.replace(".nii.gz", ""),
              r.category,
              r.lesion,
              fmt(r.rawDice),
              fmt(r.structuredDice),
              fmt(r.deltaDice),
              r.prompt,
            ]),
            [1700, 1500, 1300, 900, 1200, 900, 4400],
          ),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Typical Losses for Structured")] }),
          makeTable(
            ["Case", "Category", "Lesion", "Raw Dice", "Structured Dice", "Delta", "Prompt"],
            narrative.worst.map((r) => [
              r.case.replace(".nii.gz", ""),
              r.category,
              r.lesion,
              fmt(r.rawDice),
              fmt(r.structuredDice),
              fmt(r.deltaDice),
              r.prompt,
            ]),
            [1700, 1500, 1300, 900, 1200, 900, 4400],
          ),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Conclusion")] }),
          bullet("Structured prompt did not outperform raw prompt on the full public val set as a default strategy."),
          bullet("Mean Dice was slightly lower for structured, and micro Dice dropped substantially, indicating that some large-volume predictions were harmed badly."),
          bullet("Structured was still useful for part of the dataset, especially several ground-glass opacity, consolidation, and some nodule findings with strong location cues."),
          bullet("The current practical takeaway is not to fully replace raw prompt, but to consider selective use by finding type."),

          new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Reason Analysis")] }),
          ...narrative.reasons.map((r) => bullet(r)),
        ],
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT_DOCX, buffer);
  console.log(OUT_DOCX);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
