---
license: cc-by-nc-sa-4.0
pretty_name: ReXGroundingCT
---

# ReXGroundingCT

[ReXGroundingCT](https://www.arxiv.org/abs/2507.22030) is a dataset designed to link free-text radiology findings with pixel-level segmentations in 3D chest CT scans. Each sample consists of a volumetric CT scan, associated segmentation masks for one or more findings, and detailed textual descriptions.  
The dataset has segmentations for 8,028 findings across 14 different categories in 3,142 CT scans. There are 2,992 scans allocated for training, 50 for public validation, and 100 held privately to be hosted for a public leaderboard on [ReXrank](https://arxiv.org/abs/2411.15122).

---

## Dataset Structure

Each item in the dataset is represented as a JSON entry with the following structure:

```json
{
    "name": "train_1741_b_2.nii.gz",
    "findings": {
        "0": "Irregularly circumscribed nodular consolidation area adjacent to the diaphragm in the basal segment of the lower lobe of the right lung"
    },
    "entity_counts": {
        "0": 1
    },
    "shape": [512, 512, 238],
    "pixels": {
        "0": 15571
    },
    "categories": {
        "0": "2b"
    },
    "protocol": "protocol1"
}
```

### Field Descriptions

- **`name`**: The filename of the CT volume (e.g., `train_1741_b_2.nii.gz`).
- **`findings`**: A dictionary mapping finding indices in the segmentation mask to free-text radiology findings.
- **`entity_counts`**: Number of segmented entities for each finding.
- **`shape`**: The shape of the CT volume in `[H, W, D]` (height, width, depth).
- **`pixels`**: Number of non-zero pixels for each finding's segmentation mask.
- **`categories`**: The category code assigned to each finding (e.g., `"2b"`).
- **`protocol`**: Annotation protocol used for the case (e.g., `protocol1`, `protocol2`).

---

## Segmentation Masks

Each CT scan has an associated segmentation mask volume with the shape:

```
(F, H, W, D)
```

- **F**: Number of findings in the scan  
- **H, W, D**: Spatial dimensions of the CT volume (matching the `shape` field)

For example, a scan with 3 findings and shape `[512, 512, 238]` will have a segmentation mask of shape `[3, 512, 512, 238]`, where each slice along the F dimension corresponds to one finding.

## Citations

If you find this dataset useful, please cite the following papers:

```
@article{baharoon2025rexgroundingct,
  title={ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports},
  author={Baharoon, Mohammed and Luo, Luyang and Moritz, Michael and Kumar, Abhinav and Kim, Sung Eun and Zhang, Xiaoman and Zhu, Miao and Alabbad, Mahmoud Hussain and Alhazmi, Maha Sbayel and Mistry, Neel P and others},
  journal={arXiv preprint arXiv:2507.22030},
  year={2025}
}

@article{hamamci2024developing,
  title={Developing generalist foundation models from a multimodal dataset for 3d computed tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Doga, Irem and Durugol, Omer Faruk and Dai, Weicheng and Xu, Murong and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}
```