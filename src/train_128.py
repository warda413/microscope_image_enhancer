import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import stratified_split, MicroscopyDataset
from model import SmallUNet


def masked_mse_loss(prediction, target, mask):
    squared_error = (prediction - target) ** 2
    squared_error_at_hidden_spots = squared_error * mask
    return squared_error_at_hidden_spots.sum() / (mask.sum() + 1e-8)


def main():
    DATA_DIR = "../Data/Raw"
    CHECKPOINT_DIR = "../checkpoints_128"

    EPOCHS = 20
    START_EPOCH = 1
    BATCH_SIZE = 16
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 8  # only 8 augmented versions possible at full image size
    LEARNING_RATE = 1e-4

    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_paths, val_paths, test_paths = stratified_split(DATA_DIR)

    train_dataset = MicroscopyDataset(train_paths, patch_size=PATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE)
    val_dataset = MicroscopyDataset(val_paths, patch_size=PATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if START_EPOCH > 5:
        checkpoint_path = f"{CHECKPOINT_DIR}/latest.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Resumed from {checkpoint_path}, continuing at epoch {START_EPOCH}")

    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    best_val_loss = float("inf")
    log_path = f"{CHECKPOINT_DIR}/training_log.csv"
    log_file_exists = False
    try:
        with open(log_path, "r"):
            log_file_exists = True
    except FileNotFoundError:
        pass

    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_file_exists:
        log_writer.writerow(["epoch", "train_loss", "val_loss"])

    for epoch in range(START_EPOCH, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0

        for input_patch, target_patch, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            input_patch = input_patch.to(device)
            target_patch = target_patch.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            prediction = model(input_patch)
            loss = masked_mse_loss(prediction, target_patch, mask)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for input_patch, target_patch, mask in val_loader:
                input_patch = input_patch.to(device)
                target_patch = target_patch.to(device)
                mask = mask.to(device)

                prediction = model(input_patch)
                loss = masked_mse_loss(prediction, target_patch, mask)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch {epoch}: train loss = {avg_train_loss:.5f}, val loss = {avg_val_loss:.5f}")

        log_writer.writerow([epoch, avg_train_loss, avg_val_loss])
        log_file.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best.pth")
            print(f"  -> New best model saved (val loss {avg_val_loss:.5f})")

        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/latest.pth")

    log_file.close()


if __name__ == "__main__":
    main()