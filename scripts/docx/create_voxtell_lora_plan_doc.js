const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  PageNumber,
  Header,
  Footer,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  ShadingType,
} = require("docx");

const root = path.resolve(__dirname, "..", "..");
const outPath = path.join(root, "docs", "plans", "VoxTell_LoRA_minimal_plan.docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    ...opts,
    children: [new TextRun(text)],
  });
}

function bullet(text) {
  return new Paragraph({
    text,
    bullet: { level: 0 },
    spacing: { after: 80 },
  });
}

const doc = new Document({
  creator: "Codex",
  title: "VoxTell LoRA Minimal Plan",
  description: "Minimal viable LoRA fine-tuning plan for VoxTell on ReXGroundingCT.",
  styles: {
    default: {
      document: {
        run: {
          font: "Arial",
          size: 22,
        },
        paragraph: {
          spacing: { after: 120 },
        },
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 32, bold: true },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 26, bold: true },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 },
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
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [new TextRun({ text: "VoxTell LoRA Plan", size: 18, color: "666666" })],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun("Page "),
                new TextRun({ children: [PageNumber.CURRENT] }),
              ],
            }),
          ],
        }),
      },
      children: [
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          alignment: AlignmentType.CENTER,
          children: [new TextRun("VoxTell LoRA Minimal Plan")],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: [new TextRun({ text: "Date: 2026-04-13", color: "666666" })],
        }),
        p("Goal: add a minimal, low-risk fine-tuning path for VoxTell without depending on an official training script."),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Current Findings")] }),
        bullet("VoxTell model structure is visible in local source code and is not a black box."),
        bullet("The actual inference path is: Qwen text backbone -> project_text_embed -> transformer_decoder -> project_to_decoder_channels -> decoder."),
        bullet("The most likely bottleneck for domain adaptation is text-to-mask alignment, not the CNN backbone."),
        bullet("A first LoRA experiment should avoid full Qwen fine-tuning and avoid touching 3D convolution blocks."),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Recommended LoRA Scope")] }),
        p("Phase A is the recommended minimum viable setup. Freeze the text backbone, encoder, and convolutional decoder. Add LoRA only to the alignment path."),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2700, 1800, 4860],
          rows: [
            new TableRow({
              children: [
                new TableCell({
                  width: { size: 2700, type: WidthType.DXA },
                  borders: { top: border, bottom: border, left: border, right: border },
                  shading: { fill: "DCE6F1", type: ShadingType.CLEAR },
                  margins: cellMargins,
                  children: [p("Module Group")],
                }),
                new TableCell({
                  width: { size: 1800, type: WidthType.DXA },
                  borders: { top: border, bottom: border, left: border, right: border },
                  shading: { fill: "DCE6F1", type: ShadingType.CLEAR },
                  margins: cellMargins,
                  children: [p("Action")],
                }),
                new TableCell({
                  width: { size: 4860, type: WidthType.DXA },
                  borders: { top: border, bottom: border, left: border, right: border },
                  shading: { fill: "DCE6F1", type: ShadingType.CLEAR },
                  margins: cellMargins,
                  children: [p("Reason")],
                }),
              ],
            }),
            ...[
              ["text_backbone (Qwen)", "Freeze", "Large, expensive, and not the best first target."],
              ["encoder", "Freeze", "General 3D visual features are already usable."],
              ["decoder conv blocks", "Freeze", "Higher engineering cost and weaker first signal."],
              ["project_text_embed", "LoRA", "Directly controls text embedding adaptation into query space."],
              ["transformer_decoder", "LoRA", "Core text-image fusion module."],
              ["project_to_decoder_channels", "LoRA", "Controls how prompt embeddings map into mask-generation channels."],
            ].map(([a, b, c]) =>
              new TableRow({
                children: [a, b, c].map((text, idx) =>
                  new TableCell({
                    width: { size: [2700, 1800, 4860][idx], type: WidthType.DXA },
                    borders: { top: border, bottom: border, left: border, right: border },
                    margins: cellMargins,
                    children: [p(text)],
                  })
                ),
              })
            ),
          ],
        }),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Concrete Target Modules")] }),
        bullet("project_text_embed.0 and project_text_embed.2"),
        bullet("transformer_decoder.layers.{0..5}.linear1"),
        bullet("transformer_decoder.layers.{0..5}.linear2"),
        bullet("transformer_decoder.layers.{0..5}.self_attn.out_proj"),
        bullet("transformer_decoder.layers.{0..5}.multihead_attn.out_proj"),
        bullet("project_to_decoder_channels.{0..4}.0 and project_to_decoder_channels.{0..4}.2"),
        p("Do not target MultiheadAttention in_proj_weight in Phase A. PyTorch packs q, k, v together there, which makes off-the-shelf PEFT less clean. If Phase A works, Phase B can refactor attention into explicit q_proj, k_proj, v_proj layers."),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Training Recipe")] }),
        bullet("Initialize from the existing checkpoint_final.pth."),
        bullet("Use current text prompts first; prompt engineering and LoRA should not be mixed in the first ablation."),
        bullet("Loss: Dice loss + BCEWithLogits loss."),
        bullet("Optimizer: AdamW on LoRA parameters only."),
        bullet("Start with a small train subset for smoke testing, then expand."),
        bullet("Validation target: compare against current raw prompt baseline on the same public val split."),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Why This Is the Lowest-Risk Path")] }),
        bullet("No need to reproduce the official VoxTell training pipeline."),
        bullet("No need to full-finetune Qwen 4B."),
        bullet("Keeps trainable scope concentrated on prompt-to-mask alignment."),
        bullet("If it fails, rollback cost is low and diagnosis is straightforward."),

        new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Next Engineering Step")] }),
        p("Build a small standalone fine-tuning script that reuses current model loading code, injects LoRA into the listed modules, and trains on CT plus finding-mask pairs. That should be the next implementation task."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outPath, buffer);
  console.log(outPath);
});
