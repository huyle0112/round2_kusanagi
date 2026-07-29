# GaussianPro / Scaffold-GS

Minimal training and inference workspace for the GaussianPro extension of
Scaffold-GS.

## Layout

- `pose/`: the retained datasets and camera poses.
- `train.py`: model training, validation, and checkpoint generation.
- `render.py`: inference and submission-image rendering.
- `arguments/`, `scene/`, `gaussian_renderer/`, `utils/`: model code.
- `lpipsPyTorch/`: validation metric used during training.
- `submodules/`: CUDA rasterizer and nearest-neighbour dependencies.

Generated outputs, notebooks, viewers, packaging artifacts, and duplicate
datasets are intentionally excluded.

## Environment

```bash
conda env create -f environment.yml
conda activate scaffold_gs
```

The CUDA extensions under `submodules/` are installed by the environment
definition.

## Train

```bash
python train.py \
  -s pose/HCM0421 \
  -m outputs/HCM0421 \
  --appearance_dim 0 \
  --use_gaussianpro \
  --gaussianpro_start_iter 3000 \
  --gaussianpro_add_until_iter 15000 \
  --gaussianpro_refine_until_iter 24000 \
  --gaussianpro_references_per_step 2 \
  --iterations 30000
```

Add `--gaussianpro_paper_faithful` to use temporal source views, one-shot
reference processing, plane-induced NCC, fixed `sigma=0.8`, full-representation
minimum-scale regularization, and the original Scaffold densification cadence.
The adapted default processes two reference views per propagation event and
uses overlap-aware source selection.

Use `--gpu <index>` to select a CUDA device. With the default `--gpu -1`, the
process respects the existing `CUDA_VISIBLE_DEVICES` environment setting.

For a deterministic validation split, add:

```bash
--validation_ratio 0.1 --validation_seed 42
```

## Render

```bash
python render.py -m outputs/HCM0421
```

Use `python train.py --help` and `python render.py --help` for the complete
option lists.
