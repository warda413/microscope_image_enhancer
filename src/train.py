import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import stratified_split, MicroscopyDataset
from model import SmallUNet


def masked_mse_loss(prediction, target, mask):
    """
    Measure how wrong the model's guess is — but ONLY at the pixels we
    deliberately hid. We never grade it on pixels it could already see.
    """
    squared_error = (prediction - target) ** 2
    squared_error_at_hidden_spots = squared_error * mask
    return squared_error_at_hidden_spots.sum() / (mask.sum() + 1e-8)
def main():
    # Where your images live, and where we save trained models
    DATA_DIR = "../Data/Raw"
    CHECKPOINT_DIR = "../checkpoints"

    # Training settings — we'll explain and tune these shortly
    EPOCHS = 20
    START_EPOCH = 1
    BATCH_SIZE = 16
    PATCH_SIZE = 64
    LEARNING_RATE = 1e-4

    device = torch.device("cpu")  # confirmed earlier: no NVIDIA GPU available
    print(f"Using device: {device}")

    # Reuse the exact split from Piece 4, so train/val/test stay consistent
    train_paths, val_paths, test_paths = stratified_split(DATA_DIR)

    train_dataset = MicroscopyDataset(train_paths, patch_size=PATCH_SIZE)
    val_dataset = MicroscopyDataset(val_paths, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Only resume if we're actually starting partway through
    if START_EPOCH > 1:
        checkpoint_path = f"{CHECKPOINT_DIR}/latest.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Resumed from {checkpoint_path}, continuing at epoch {START_EPOCH}")

    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    best_val_loss = float("inf")
    # Set up a CSV file to permanently record every epoch's results
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
        # --- Training phase ---
        model.train()  # tells the model "you're in learning mode"
        running_train_loss = 0.0

        for input_patch, target_patch, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            input_patch = input_patch.to(device)
            target_patch = target_patch.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()                          # clear old nudges
            prediction = model(input_patch)                 # make a guess
            loss = masked_mse_loss(prediction, target_patch, mask)  # how wrong was it
            loss.backward()                                  # figure out which knobs to nudge
            optimizer.step()                                  # actually nudge them

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()  # tells the model "you're being tested, don't learn right now"
        running_val_loss = 0.0

        with torch.no_grad():  # extra safety: guarantees no accidental learning happens here
            for input_patch, target_patch, mask in val_loader:
                input_patch = input_patch.to(device)
                target_patch = target_patch.to(device)
                mask = mask.to(device)

                prediction = model(input_patch)
                loss = masked_mse_loss(prediction, target_patch, mask)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch {epoch}: train loss = {avg_train_loss:.5f}, val loss = {avg_val_loss:.5f}")

        # Permanently record this epoch's results
        log_writer.writerow([epoch, avg_train_loss, avg_val_loss])
        log_file.flush()  # write to disk immediately, don't wait until the very end

        # Save the model whenever it gets a new best validation score
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best.pth")
            print(f"  -> New best model saved (val loss {avg_val_loss:.5f})")

    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/latest.pth")

    log_file.close()


if __name__ == "__main__":
    main()