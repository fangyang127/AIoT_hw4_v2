import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def random_flip(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.2:
        img = ImageOps.flip(img)
    return img


def random_rotate(img, max_angle=25):
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False)


def random_crop_resize(img, scale=(0.8, 1.0)):
    w, h = img.size
    scale_factor = random.uniform(scale[0], scale[1])
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    if new_w >= w or new_h >= h:
        return img
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img_cropped = img.crop((left, top, left + new_w, top + new_h))
    return img_cropped.resize((w, h), Image.Resampling.LANCZOS)


def random_color(img):
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    return img


def add_gaussian_noise(img, mean=0.0, std=5.0):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def random_augment(img):
    """Apply 1-3 random augmentation ops and return a PIL Image."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    ops = [random_flip, random_rotate, random_crop_resize, random_color, add_gaussian_noise]
    random.shuffle(ops)
    n_ops = random.randint(1, 3)
    out = img.copy()
    for op in ops[:n_ops]:
        out = op(out)
    return out
