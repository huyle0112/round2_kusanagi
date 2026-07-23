from pathlib import Path

import torch
from PIL import Image


JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif"}


def save_pil_high_quality(pil_image, path, jpeg_quality=100):
    path = Path(path)
    if path.suffix.lower() in JPEG_EXTENSIONS:
        pil_image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        pil_image.save(path)


def save_image_high_quality(image, path, jpeg_quality=100):
    """Save a CHW tensor without torchvision's default JPEG quality loss."""
    path = Path(path)
    array = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    if array.shape[-1] == 1:
        array = array[..., 0]
    pil_image = Image.fromarray(array)
    # 4:4:4 chroma and quality 100 avoid the default JPEG-75 penalty.
    save_pil_high_quality(pil_image, path, jpeg_quality=jpeg_quality)
