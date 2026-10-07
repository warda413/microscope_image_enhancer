import csv
import matplotlib.pyplot as plt

LOG_PATH = "../checkpoints_128/training_log.csv"
OUTPUT_DIR = "../checkpoints_128"

# Step 1 — read the CSV file back into three simple lists
epochs = []
train_losses = []
val_losses = []

with open(LOG_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row["train_loss"]))
        val_losses.append(float(row["val_loss"]))

print(f"Loaded {len(epochs)} epochs from {LOG_PATH}")

# Step 2 — line graph: train vs validation loss over time
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Train loss")
plt.plot(epochs, val_losses, marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Masked MSE loss")
plt.title("Training progress")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150)
plt.close()
print(f"Saved line graph to {OUTPUT_DIR}/loss_curve.png")

# Step 3 — bar chart: validation loss per epoch, so individual epochs are easy to compare
plt.figure(figsize=(10, 5))
plt.bar(epochs, val_losses, color="steelblue")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Validation loss by epoch")
plt.xticks(epochs)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/val_loss_bar_chart.png", dpi=150)
plt.close()
print(f"Saved bar chart to {OUTPUT_DIR}/val_loss_bar_chart.png")