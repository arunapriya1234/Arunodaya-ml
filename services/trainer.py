import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from core.model import LandmarkMLP

def train_landmark_mlp(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    classes: list = None,
    output_dir: str = "outputs/models/mudra_mlp",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.001
):
    """
    Trains LandmarkMLP model and saves weights (model.pt) and metadata (meta.json).
    Supports arbitrary number of classes and optional validation split.
    """
    os.makedirs(output_dir, exist_ok=True)

    if classes is None:
        num_classes = int(torch.max(y_train).item()) + 1
        classes = [f"Class_{i}" for i in range(num_classes)]
    else:
        num_classes = len(classes)

    input_dim = X_train.shape[1]

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LandmarkMLP(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"--- Starting Landmark MLP Training ---")
    print(f"Input Dim: {input_dim} | Classes: {num_classes} | Train samples: {len(X_train)} | Val samples: {len(X_val) if X_val is not None else 0}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        val_info = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    v_out = model(vx)
                    v_loss = criterion(v_out, vy)
                    val_loss += v_loss.item() * vx.size(0)
                    _, v_pred = torch.max(v_out, 1)
                    val_total += vy.size(0)
                    val_correct += (v_pred == vy).sum().item()

            cur_val_loss = val_loss / val_total
            cur_val_acc = (val_correct / val_total) * 100.0
            val_info = f" | Val Loss: {cur_val_loss:.4f} | Val Acc: {cur_val_acc:.2f}%"

            if cur_val_acc >= best_val_acc:
                best_val_acc = cur_val_acc
                best_model_state = model.state_dict().copy()
        else:
            if train_acc >= best_train_acc:
                best_train_acc = train_acc
                best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%{val_info}")

    # Use best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save Model Weights
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Save Metadata
    meta_path = os.path.join(output_dir, "meta.json")
    meta_data = {
        "model_type": "landmark_mlp",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "classes": classes,
        "normalization": "wrist_origin_middle_mcp_scale_multi_hand",
        "best_val_accuracy": float(best_val_acc) if val_loader is not None else None,
        "final_train_accuracy": float(train_acc)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    print(f"\n[OK] Saved model weights to: {model_path}")
    print(f"[OK] Saved metadata to: {meta_path}")
    return model_path, meta_path
