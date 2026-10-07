import torch
import numpy as np
from PIL import Image

from model import SmallUNet
from dataset import load_image_as_array, stratified_split

CHECKPOINT_PATH = "../checkpoints/best.pth"
TILE_SIZE = 64
OVERLAP = 16

device = torch.device("cpu")


def enhance_full_image(model, img):
    """
    Run the model on a full-size image by sliding a 64x64 window across it,
    with overlap, then blending overlapping regions smoothly to avoid seams.
    """
    h, w = img.shape
    stride = TILE_SIZE - OVERLAP

    output = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    # A soft "fade in / fade out" window, so overlapping tiles blend smoothly
    # instead of just averaging bluntly at the seams
    window_1d = np.hanning(TILE_SIZE)
    window_2d = np.outer(window_1d, window_1d).astype(np.float32)
    window_2d = np.clip(window_2d, 0.1, 1.0)  # avoid zero-weight at tile edges

    y_positions = list(range(0, max(h - TILE_SIZE, 0) + 1, stride)) or [0]
    x_positions = list(range(0, max(w - TILE_SIZE, 0) + 1, stride)) or [0]

    # Make sure we always reach the far edge, even if it doesn't land on a clean stride
    if h > TILE_SIZE and y_positions[-1] != h - TILE_SIZE:
        y_positions.append(h - TILE_SIZE)
    if w > TILE_SIZE and x_positions[-1] != w - TILE_SIZE:
        x_positions.append(w - TILE_SIZE)

    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                tile = img[y:y_end, x:x_end]

                tile_h, tile_w = tile.shape
                pad_h = TILE_SIZE - tile_h
                pad_w = TILE_SIZE - tile_w
                tile_padded = np.pad(tile, ((0, pad_h), (0, pad_w)), mode="reflect")

                tile_tensor = torch.from_numpy(tile_padded).unsqueeze(0).unsqueeze(0).float()
                prediction = model(tile_tensor).squeeze().numpy()

                prediction = prediction[:tile_h, :tile_w]
                tile_weight = window_2d[:tile_h, :tile_w]

                output[y:y_end, x:x_end] += prediction * tile_weight
                weight[y:y_end, x:x_end] += tile_weight

    weight[weight == 0] = 1.0  # avoid division by zero, just in case
    return output / weight


def main():
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")

    _, _, test_paths = stratified_split("../Data/Raw")
    test_image_path = test_paths[0]
    print(f"Using test image: {test_image_path}")

    img = load_image_as_array(test_image_path)
    print(f"Image size: {img.shape}")

    enhanced = enhance_full_image(model, img)
    enhanced = np.clip(enhanced, 0, 1)

    original_img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    enhanced_img = (enhanced * 255).astype(np.uint8)

    comparison = np.concatenate([original_img, enhanced_img], axis=1)
    Image.fromarray(comparison).save("../checkpoints/full_image_comparison.png")
    print("Saved comparison to ../checkpoints/full_image_comparison.png")


if __name__ == "__main__":
    main()