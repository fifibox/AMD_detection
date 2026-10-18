"""data.py -- dataset listing, label tasks, patient-grouped splits, transforms.

Single source of truth for everything Chapter 3 Sections 3.2-3.4 describe.

Folder layout (unchanged from DATASET/):
    DATASET/0  No AMD
    DATASET/1  Early AMD
    DATASET/2  Intermediate AMD
    DATASET/3  Late AMD (neovascular or centre-involving GA)

Label regrouping is done HERE from the original 4-class folder label, so no
symlinked "binary_views" folders are needed any more.
"""

import hashlib
import os
from pathlib import Path
import re

import numpy as np
import pydicom
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTS = {".dcm", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Original Beckman folder label -> task label.  Matches thesis Table 3.2.
TASKS = {
    "binary": {
        "mapping": {0: 0, 1: 0, 2: 1, 3: 1},
        "class_names": ["non_referable", "referable"],
    },
    "three_class": {
        "mapping": {0: 0, 1: 0, 2: 1, 3: 2},
        "class_names": ["non_referable", "intermediate", "late"],
    },
    # Trained on all four Beckman grades; scored as referable via P(2)+P(3)
    # ("4 -> 2" fusion, evaluate.py).
    "four_class": {
        "mapping": {0: 0, 1: 1, 2: 2, 3: 3},
        "class_names": ["no_amd", "early", "intermediate", "late"],
    },
}

_PYDICOM_MAJOR = int(pydicom.__version__.split(".")[0])


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------
def subject_id_from_path(path):
    """'SUBJ_4B7977_...' or 'B2_SUBJ_4B7977_...' -> '4B7977'."""
    m = re.search(r"(?:^|_)SUBJ_([^_]+)_", Path(path).name)
    if m is None:
        raise ValueError(f"No SUBJ_<id> token in filename: {Path(path).name}")
    return m.group(1)


def list_dataset(root, task):
    """Return samples [(path, task_label, original_label)] and class names."""
    mapping = TASKS[task]["mapping"]
    root = Path(root)
    samples = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        orig = int(d.name)
        for f in sorted(d.iterdir()):
            if f.is_file() and not f.name.startswith("._") and f.suffix.lower() in IMG_EXTS:
                samples.append((str(f), mapping[orig], orig))
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    names = TASKS[task]["class_names"]
    counts = np.bincount([s[1] for s in samples], minlength=len(names))
    print(f"[data] task={task}: {len(samples)} images, "
          f"{len(set(subject_id_from_path(s[0]) for s in samples))} patients, "
          f"{dict(zip(names, counts.tolist()))}")
    return samples, names


def groups_of(samples):
    return np.asarray([subject_id_from_path(s[0]) for s in samples])


# --------------------------------------------------------------------------
# Splits (Section 3.3)
# --------------------------------------------------------------------------
def outer_folds(samples, n_splits=5, seed=42):
    """Patient-grouped, label-stratified K-fold over all images.

    Stratification target is the TASK label of each image; group is the
    patient ID, so every image (both eyes, repeat captures) of a patient
    falls in exactly one test fold.
    """
    y = np.array([s[1] for s in samples])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups_of(samples)))


def inner_train_val(train_samples, val_frac=0.1, seed=42):
    """Hold out ~val_frac of the TRAINING patients (grouped + stratified) for
    checkpoint selection / early stopping. Never touches the test fold."""
    y = np.array([s[1] for s in train_samples])
    n = max(2, round(1 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return next(iter(sgkf.split(np.zeros(len(y)), y, groups_of(train_samples))))


# --------------------------------------------------------------------------
# Image decoding
# --------------------------------------------------------------------------
# Optional on-disk cache of decoded RGB arrays. Set AMD_CACHE_DIR to a
# node-local directory (job.slurm uses /tmp on the compute node). The cached array is the exact uint8
# output of _decode_rgb, so results are identical with or without the cache;
# it only removes the per-epoch JPEG decode + YBR->RGB conversion cost.
_CACHE_DIR = os.environ.get("AMD_CACHE_DIR")


def _cache_path(path):
    return Path(_CACHE_DIR) / (hashlib.md5(str(path).encode()).hexdigest() + ".npy")


def load_rgb(path):
    """Decoded 8-bit RGB PIL image, via the on-disk cache when enabled."""
    if _CACHE_DIR:
        cp = _cache_path(path)
        if cp.exists():
            return Image.fromarray(np.load(cp))
        img = _decode_rgb(path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, np.asarray(img))
        os.replace(tmp, cp)  # atomic: safe with several DataLoader workers
        return img
    return _decode_rgb(path)


def _decode_rgb(path):
    """Decode a DICOM (or PNG) to an 8-bit RGB PIL image.

    The DICOMs are JPEG-baseline, PhotometricInterpretation YBR_FULL_422.
    pydicom 2.x returns raw YCbCr for these and needs an explicit conversion;
    pydicom 3.x already returns RGB (pixel_array as_rgb=True). Converting
    twice gives a blue/purple cast, so gate on the installed major version.
    """
    if not path.lower().endswith(".dcm"):
        return Image.open(path).convert("RGB")
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 colour image, got {img.shape}: {path}")
    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    if _PYDICOM_MAJOR < 3 and pi in {"YBR_FULL", "YBR_FULL_422"}:
        from pydicom.pixel_data_handlers.util import convert_color_space
        img = convert_color_space(img, pi, "RGB")
    return Image.fromarray(img.astype(np.uint8))


class SquareCenterCrop:
    """Crop the central min(H, W) x min(H, W) square.

    On these 1934x2576 / 1958x2576 Topcon images the fundus circle is
    ~1870-1900 px in diameter and horizontally centred, so the central
    H x H square keeps the whole circle and discards only black border.
    Applied identically to train, validation and test images.
    """
    def __call__(self, img):
        w, h = img.size
        s = min(w, h)
        return TF.center_crop(img, [s, s])


class FundusDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        return self.transform(load_rgb(path)), label


# --------------------------------------------------------------------------
# Transforms (Section 3.4) -- order here is the order written in the thesis
# --------------------------------------------------------------------------
def build_transforms_retfound(input_size=224, train=True):
    """Augmentation of the official RETFound fine-tuning code (util/datasets.py:
    timm.data.create_transform with the main_finetune.py defaults).
    Train: RandomResizedCrop(scale 0.08-1, ratio 3/4-4/3, bicubic) + h-flip +
           RandAugment 'rand-m9-mstd0.5-inc1' + random erasing p=0.25 (pixel mode).
    Eval : Resize(input_size / 0.875) + CenterCrop(input_size), bicubic.
    The same SquareCenterCrop is applied first (it only removes black border),
    standing in for the cropped fundus images RETFound was trained on."""
    from timm.data import create_transform
    if train:
        t = create_transform(input_size=input_size, is_training=True, color_jitter=None,
                             auto_augment="rand-m9-mstd0.5-inc1", interpolation="bicubic",
                             re_prob=0.25, re_mode="pixel", re_count=1,
                             mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return transforms.Compose([SquareCenterCrop(), t])
    bicubic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        SquareCenterCrop(),
        transforms.Resize(int(input_size / (224 / 256)), interpolation=bicubic),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_transforms(input_size=224, train=True, aug="default"):
    if aug == "retfound":
        return build_transforms_retfound(input_size, train)
    bicubic = transforms.InterpolationMode.BICUBIC
    common_head = [SquareCenterCrop()]
    common_tail = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        aug = [
            # Now feasible because the input is already square (the old
            # pipeline applied this to a 4:3 image, where every sample was
            # rejected and torchvision silently fell back to a fixed crop).
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(0.95, 1.05),
                                         interpolation=bicubic),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=bicubic),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ]
        return transforms.Compose(common_head + aug + common_tail)
    return transforms.Compose(common_head + [transforms.Resize(input_size, interpolation=bicubic)] + common_tail)
