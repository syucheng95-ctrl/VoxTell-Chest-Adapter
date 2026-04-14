# Project Layout

This workspace is now organized around a small number of stable top-level directories.

## Core

- `VoxTell/`: upstream model source
- `models/`: checkpoints and split weights
- `datasets/`: local datasets and metadata mirrors
- `outputs/`: inference outputs and evaluation metrics

## Working Docs

- `docs/plans/`: implementation plans
- `docs/reports/`: experiment reports
- `docs/notes/`: reading notes and architecture notes
- `docs/references/`: papers and original reference PDFs

## Scripts

- `scripts/experiments/`: run inference experiments
- `scripts/analysis/`: evaluation and analysis scripts
- `scripts/utils/`: utility helpers
- `scripts/docx/`: Word report generators

## Status

- `status/`: current progress snapshots and handoff notes

## Notes

- `package.json` and `node_modules/` stay at project root because the docx generation scripts depend on them.
- `hf_cache/` remains at root because it is used as a runtime cache, not as a source asset.
