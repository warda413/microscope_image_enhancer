import os
import glob
import random
import numpy as np
import torch
from PIL import Image


def load_image_as_array(path):
    """Load a PNG image as a float32 numpy array, scaled to [0, 1]."""
    arr = np.array(Image.open(path).convert("L")).astype(np.float32)

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    return arr
def find_image_paths(root_dir, extensions=(".png", ".jpg", ".jpeg")):
    """Recursively find all images under root_dir, including subfolders."""
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    return paths
def mask_blind_spots(patch, mask_ratio=0.02, neighborhood_radius=5):
    """
    Hide a random subset of pixels by replacing each one with a nearby
    pixel's value, and remember exactly which pixels were hidden.

    Returns:
        input_patch:  the corrupted patch (what the model will see)
        target_patch: the ORIGINAL, unmodified patch (the correct answer)
        mask:         boolean array, True at every hidden pixel location
    """
    h, w = patch.shape
    n_pixels = h * w
    n_mask = max(1, int(n_pixels * mask_ratio))

    input_patch = patch.copy()
    mask = np.zeros((h, w), dtype=bool)

    ys = np.random.randint(0, h, size=n_mask)
    xs = np.random.randint(0, w, size=n_mask)

    r = neighborhood_radius
    for y, x in zip(ys, xs):
        while True:
            dy = random.randint(-r, r)
            dx = random.randint(-r, r)
            if dy != 0 or dx != 0:
                break
        ny = min(max(y + dy, 0), h - 1)
        nx = min(max(x + dx, 0), w - 1)

        input_patch[y, x] = patch[ny, nx]
        mask[y, x] = True

    return input_patch, patch, mask
def stratified_split(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split images into train/val/test sets, keeping the anaphase/metaphase
    proportions consistent across all three splits (stratified split).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    class_folders = [
        f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))
    ]

    rng = random.Random(seed)  # fixed seed = same split every time we run this

    train_paths, val_paths, test_paths = [], [], []

    for class_name in class_folders:
        class_dir = os.path.join(root_dir, class_name)
        paths = find_image_paths(class_dir)
        rng.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train : n_train + n_val])
        test_paths.extend(paths[n_train + n_val :])

        print(f"{class_name}: {n} total -> {n_train} train / {n_val} val / {n - n_train - n_val} test")

    return train_paths, val_paths, test_paths
class MicroscopyDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that takes a list of image paths, cuts random patches
    from them, and applies the blind-spot masking trick on the fly.
    """

    def __init__(self, image_paths, patch_size=64, patches_per_image=50, mask_ratio=0.02):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.mask_ratio = mask_ratio

        # Load every image into memory once, up front
        self.images = [load_image_as_array(p) for p in image_paths]

        # Drop any image smaller than one patch (rare, but avoids crashes later)
        self.images = [img for img in self.images if min(img.shape[:2]) >= patch_size]

        if not self.images:
            raise ValueError(f"No images are large enough for patch_size={patch_size}")

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def _random_crop(self, img):
        h, w = img.shape
        ps = self.patch_size
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        return img[y : y + ps, x : x + ps].copy()

    def _augment(self, patch):
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        if random.random() < 0.5:
            patch = np.flipud(patch).copy()
        k = random.randint(0, 3)
        if k:
            patch = np.rot90(patch, k).copy()
        return patch

    def __getitem__(self, idx):
        img = self.images[idx % len(self.images)]
        patch = self._random_crop(img)
        patch = self._augment(patch)

        input_patch, target_patch, mask = mask_blind_spots(patch, mask_ratio=self.mask_ratio)

        input_t = torch.from_numpy(input_patch).unsqueeze(0).float()
        target_t = torch.from_numpy(target_patch).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return input_t, target_t, mask_t
    