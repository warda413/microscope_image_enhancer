import torch
import numpy as np
from PIL import Image

from model import SmallUNet
from dataset import load_image_as_array, stratified_split, mask_blind_spots

CHECKPOINT_PATH = "../checkpoints_128/best.pth"
OUTPUT_DIR = "../checkpoints_128"
NUM_SAMPLE_IMAGES_TO_SAVE = 5  # how many visual examples to save for inspection

device = torch.device("cpu")


def masked_mse_loss(prediction, target, mask):
    squared_error = (prediction - target) ** 2
    squared_error_at_hidden_spots = squared_error * mask
    return squared_error_at_hidden_spots.sum() / (mask.sum() + 1e-8)


def main():
    model = SmallUNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")

    _, _, test_paths = stratified_split("../Data/Raw")
    print(f"Evaluating on {len(test_paths)} test images (never seen during training or validation)")

    all_losses = []
    saved_count = 0

    with torch.no_grad():
        for path in test_paths:
            img = load_image_as_array(path)

            # Use the same blind-spot masking trick as training, so we can
            # measure accuracy the same honest way: hide real pixels, see
            # how close the model's guess comes to the true value.
            h, w = img.shape
            if h < 128 or w < 128:
                continue  # skip anything too small to evaluate fairly

            patch = img[:128, :128]
            input_patch, target_patch, mask = mask_blind_spots(patch, mask_ratio=0.02)

            input_tensor = torch.from_numpy(input_patch).unsqueeze(0).unsqueeze(0).float()
            target_tensor = torch.from_numpy(target_patch).unsqueeze(0).unsqueeze(0).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

            prediction = model(input_tensor)
            loss = masked_mse_loss(prediction, target_tensor, mask_tensor)
            all_losses.append(loss.item())

            # Save a handful of visual before/after examples for manual inspection
            if saved_count < NUM_SAMPLE_IMAGES_TO_SAVE:
                pred_np = prediction.squeeze().numpy()
                original_img = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
                enhanced_img = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
                comparison = np.concatenate([original_img, enhanced_img], axis=1)
                Image.fromarray(comparison).save(f"{OUTPUT_DIR}/test_sample_{saved_count}.png")
                saved_count += 1

    all_losses = np.array(all_losses)
    print(f"\nEvaluated {len(all_losses)} test images")
    print(f"Mean test loss:   {all_losses.mean():.6f}")
    print(f"Std deviation:    {all_losses.std():.6f}")
    print(f"Min loss:         {all_losses.min():.6f}")
    print(f"Max loss:         {all_losses.max():.6f}")
    print(f"\nSaved {saved_count} sample comparison images to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
    