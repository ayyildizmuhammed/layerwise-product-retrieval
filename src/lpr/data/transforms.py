"""Image preprocessing and training augmentation.

Every crop is padded to a square with a neutral value (128), centred, resized to 224x224 with
area interpolation and normalised with the default image statistics of the pretrained encoder:
mean 0.5 and standard deviation 0.5 for PE, and each baseline's own statistics for CLIP and
SigLIP2 (see :mod:`lpr.models.baselines`).

Training augmentation is deliberately mild: small colour, brightness and contrast jitter, low-rate
JPEG compression, small shifts, scaling and rotation, and occasional three-pixel motion blur.
Horizontal flips and cutout are disabled because they mirror asymmetric package text or erase
small identifying regions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

PE_MEAN = (0.5, 0.5, 0.5)
PE_STD = (0.5, 0.5, 0.5)


def load_rgb(path: str) -> Image.Image:
    """Open an image as RGB; transparent and palette images are composited on white."""
    image = Image.open(path)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def pad_to_square(pixels: np.ndarray, pad_value: int = 128) -> np.ndarray:
    """Pad an ``H x W x 3`` array to a square, centred; an odd extra pixel goes bottom/right."""
    height, width = pixels.shape[:2]
    if height == width:
        return pixels
    diff = abs(height - width)
    before, after = diff // 2, diff - diff // 2
    if height < width:
        padding = ((before, after), (0, 0), (0, 0))
    else:
        padding = ((0, 0), (before, after), (0, 0))
    return np.pad(pixels, padding, mode="constant", constant_values=pad_value)


@dataclass
class LightAugmentation:
    """The ``retail_sku_light`` policy, applied after padding and resizing.

    Each step and its probability:

    1. colour - always one of: colour jitter (brightness, contrast and saturation 0.08, hue
       0.01) or brightness/contrast jitter (0.08 each), chosen with equal probability;
    2. JPEG compression with quality 80-95, p = 0.12;
    3. shift (0.5 %), scale (4 %) and rotation (2 degrees) with reflected borders, p = 0.18;
    4. motion blur with a 3-pixel kernel, p = 0.08.
    """

    jitter: float = 0.08
    hue: float = 0.01
    jpeg_p: float = 0.12
    jpeg_quality: tuple[int, int] = (80, 95)
    affine_p: float = 0.18
    shift: float = 0.005
    scale: float = 0.04
    rotate: float = 2.0
    blur_p: float = 0.08

    def __call__(self, pixels: np.ndarray, rng: random.Random) -> np.ndarray:
        pixels = self._colour(pixels, rng)
        if rng.random() < self.jpeg_p:
            quality = rng.randint(*self.jpeg_quality)
            bgr = np.ascontiguousarray(pixels[..., ::-1])
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                pixels = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
        if rng.random() < self.affine_p:
            pixels = self._affine(pixels, rng)
        if rng.random() < self.blur_p:
            pixels = self._motion_blur(pixels, rng)
        return np.ascontiguousarray(pixels)

    def _colour(self, pixels: np.ndarray, rng: random.Random) -> np.ndarray:
        image = pixels.astype(np.float32) / 255.0
        brightness = 1.0 + rng.uniform(-self.jitter, self.jitter)
        contrast = 1.0 + rng.uniform(-self.jitter, self.jitter)
        if rng.random() < 0.5:
            # Colour jitter: brightness, contrast, saturation and hue in random order.
            saturation = 1.0 + rng.uniform(-self.jitter, self.jitter)
            hue = rng.uniform(-self.hue, self.hue)
            steps = [
                lambda x: x * brightness,
                lambda x: (x - x.mean()) * contrast + x.mean(),
                lambda x: self._saturation(x, saturation),
                lambda x: self._hue(x, hue),
            ]
            rng.shuffle(steps)
            for step in steps:
                image = np.clip(step(image), 0.0, 1.0)
        else:
            # Brightness/contrast jitter: brightness as an additive shift, contrast around mean.
            image = (image - image.mean()) * contrast + image.mean() + (brightness - 1.0)
        return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _saturation(image: np.ndarray, factor: float) -> np.ndarray:
        grey = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return grey[..., None] + (image - grey[..., None]) * factor

    @staticmethod
    def _hue(image: np.ndarray, shift: float) -> np.ndarray:
        hsv = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + shift * 360.0) % 360.0
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _affine(self, pixels: np.ndarray, rng: random.Random) -> np.ndarray:
        height, width = pixels.shape[:2]
        angle = rng.uniform(-self.rotate, self.rotate)
        scale = 1.0 + rng.uniform(-self.scale, self.scale)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[0, 2] += rng.uniform(-self.shift, self.shift) * width
        matrix[1, 2] += rng.uniform(-self.shift, self.shift) * height
        return cv2.warpAffine(
            pixels, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _motion_blur(pixels: np.ndarray, rng: random.Random) -> np.ndarray:
        kernel = np.zeros((3, 3), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[1, :] = 1.0
        else:
            kernel[:, 1] = 1.0
        return cv2.filter2D(pixels, -1, kernel / kernel.sum())


class ImageTransform:
    """PIL image to a normalised ``[3, size, size]`` tensor.

    Args:
        size: Output side length (224).
        pad_value: Grey level used to pad the crop to a square (128).
        mean, std: Per-channel normalisation on [0, 1] values.
        augment: Optional training augmentation, applied after resizing.
        seed: Seed of the augmentation's random generator. Without a seed the global ``random``
            module is used, which PyTorch re-seeds in every DataLoader worker, so workers do not
            repeat each other's augmentations.
    """

    def __init__(
        self,
        size: int = 224,
        pad_value: int = 128,
        mean: tuple[float, float, float] = PE_MEAN,
        std: tuple[float, float, float] = PE_STD,
        augment: LightAugmentation | None = None,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.pad_value = pad_value
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.augment = augment
        self.rng = random.Random(seed) if seed is not None else None

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixels = pad_to_square(np.asarray(image.convert("RGB")), self.pad_value)
        pixels = cv2.resize(pixels, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.augment is not None:
            # The global generator is re-seeded by PyTorch in every DataLoader worker.
            pixels = self.augment(pixels, self.rng or random.Random(random.getrandbits(64)))
        pixels = np.require(pixels, requirements=["C", "W"])
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
        return (tensor - self.mean) / self.std


def pe_eval_transform() -> ImageTransform:
    """Evaluation transform of a PE representation."""
    return ImageTransform()


def pe_train_transform(seed: int | None = None) -> ImageTransform:
    """Training transform of the fine-tuned windows: the evaluation transform plus light
    augmentation."""
    return ImageTransform(augment=LightAugmentation(), seed=seed)
