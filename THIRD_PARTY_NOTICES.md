# Third-party code

- `experiments/hrf_models/` — the dual-branch U-Net of *Towards Hierarchical
  Rectified Flow* (Zhang, Yan, Schwing, Zhao; ICLR 2025), which is adapted from
  OpenAI's guided-diffusion (`guided_diffusion/unet.py`, `nn.py`, `fp16_util.py`,
  `logger.py`), released under the MIT License, Copyright (c) 2021 OpenAI.
  Used unchanged to keep the HRF2 comparison architecture-matched.
- `experiments/celebahq_dcae_dit.py` — a re-implementation of the DiT block
  design (Peebles & Xie, 2023: adaLN-Zero conditioning, 2D sin-cos position
  embeddings); no code copied.
- Pretrained models downloaded at run time and not redistributed here: DC-AE
  f32c32 (`mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers`, via `diffusers`) and the
  torchvision InceptionV3 weights used for FID. Their own licenses apply.
