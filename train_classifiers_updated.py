import os
import time
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==========================================
# CONFIGURATION & PATHS
# ==========================================
PROCESSED_DIR = "processed_data"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# L4 Leak-Free Paths
L4_PATHS = {
    "X_train": os.path.join(PROCESSED_DIR, "l4_X_train.npy"),
    "y_train": os.path.join(PROCESSED_DIR, "l4_y_train.npy"),
    "X_val": os.path.join(PROCESSED_DIR, "l4_X_val.npy"),
    "y_val": os.path.join(PROCESSED_DIR, "l4_y_val.npy"),
    "X_test": os.path.join(PROCESSED_DIR, "l4_X_test.npy"),
    "y_test": os.path.join(PROCESSED_DIR, "l4_y_test.npy"),
}

# L7 Leak-Free Paths
L7_PATHS = {
    "X_train": os.path.join(PROCESSED_DIR, "l7_X_train.npy"),
    "y_train": os.path.join(PROCESSED_DIR, "l7_y_train.npy"),
    "X_val": os.path.join(PROCESSED_DIR, "l7_X_val.npy"),
    "y_val": os.path.join(PROCESSED_DIR, "l7_y_val.npy"),
    "X_test": os.path.join(PROCESSED_DIR, "l7_X_test.npy"),
    "y_test": os.path.join(PROCESSED_DIR, "l7_y_test.npy"),
}

L7_VOCAB_SIZE = 5000
L7_EMBED_DIM = 64
L7_NUM_HEADS = 4
L7_NUM_EXPERTS = 4
L7_HIDDEN_DIM = 32
L7_MAX_SEQ_LENGTH = 128
L7_EPOCHS = 15
L7_BATCH_SIZE = 256


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def measure_inference_latency(model_func, dummy_input, num_runs=1000):
    """Measures single-input inference latency in microseconds."""
    for _ in range(50):
        _ = model_func(dummy_input)

    if isinstance(dummy_input, torch.Tensor) and dummy_input.device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(num_runs):
        _ = model_func(dummy_input)

    if isinstance(dummy_input, torch.Tensor) and dummy_input.device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    return ((end_time - start_time) / num_runs) * 1_000_000


def print_comprehensive_metrics(y_true, y_pred, y_probs, layer_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 40)
    print(f" {layer_name} FINAL 10% HOLDOUT EVALUATION ")
    print("=" * 40)
    print(f"[*] Accuracy  : {acc * 100:.4f}%")
    print(f"[*] Precision : {prec:.4f} (When it says attack, is it really?)")
    print(f"[*] Recall    : {rec:.4f} (Out of all real attacks, how many did it catch?)")
    print(f"[*] F1 Score  : {f1:.4f}")
    print(f"[*] AUC-ROC   : {auc:.4f} (Ability to separate Benign vs Attack)")

    print("\n--- Confusion Matrix ---")
    print(f"True Negatives (Benign as Benign) : {cm[0][0]}")
    print(f"False Positives (Benign as Attack): {cm[0][1]}  <-- False Alarms")
    print(f"False Negatives (Attack as Benign): {cm[1][0]}  <-- Missed Attacks!")
    print(f"True Positives  (Attack as Attack): {cm[1][1]}")
    print("------------------------\n")


# ==========================================
# 1. LAYER 4: XGBOOST TRAINING
# ==========================================
# ==========================================
# 1. LAYER 4: XGBOOST TRAINING WITH ONNX SYNC
# ==========================================
# ==========================================
# 1. LAYER 4: XGBOOST TRAINING WITH ONNX SYNC
# ==========================================
def train_xgboost_l4():
    print("\n" + "=" * 50)
    print("[*] INITIATING LAYER 4 XGBOOST (STATIC 70/20/10)")
    print("=" * 50)

    if not os.path.exists(L4_PATHS["X_train"]):
        raise FileNotFoundError(
            f"Missing {L4_PATHS['X_train']}. Run the leak-free preprocess_datasets.py first."
        )

    # Load the pre-split matrices directly
    X_train = np.load(L4_PATHS["X_train"])
    y_train = np.load(L4_PATHS["y_train"])
    X_val = np.load(L4_PATHS["X_val"])
    y_val = np.load(L4_PATHS["y_val"])
    X_test = np.load(L4_PATHS["X_test"])
    y_test = np.load(L4_PATHS["y_test"])

    print(
        f"[*] L4 Shapes -> Train: {X_train.shape} | "
        f"Val: {X_val.shape} | Test: {X_test.shape}"
    )

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=15,
        n_jobs=-1,
    )

    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(
        f"[+] XGBoost Training Complete in {time.time() - start_time:.2f} seconds. "
        f"(Stopped at tree {model.best_iteration})"
    )

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    print_comprehensive_metrics(
        y_test,
        test_preds,
        test_probs,
        "LAYER 4 (XGBoost)",
    )

    latency_us = measure_inference_latency(
        lambda x: model.predict(x),
        X_test[0:1],
    )
    print(f"[!] Single-Packet Inference Latency: {latency_us:.2f} microseconds")

    save_path = os.path.join(ARTIFACTS_DIR, "l4_xgb_model.json")
    model.save_model(save_path)
    print(f"[✔] Final L4 Model saved to {save_path}")

    # =====================================================================
    # ✅ FIX: Synchronize ONNX Types within the exact same onnxmltools domain
    # =====================================================================
    print("[*] Re-compiling ONNX binary graph to expect 50 dimensions...")
    from onnxmltools import convert_xgboost
    from onnxmltools.utils import save_model
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Extract the booster object handle
    bst = model.get_booster()

    # Define the 50-dimensional input shape schema safely
    initial_types = [('float_input', FloatTensorType([None, 50]))]

    onnx_model = convert_xgboost(
        bst, 
        initial_types=initial_types, 
        target_opset=15
    )

    onnx_out_path = os.path.join(ARTIFACTS_DIR, "l4_xgb_model.onnx")
    save_model(onnx_model, onnx_out_path)
    print(f"[✔] Successfully synchronized ONNX binary graph at: {onnx_out_path}")



# ==========================================
# 2. LAYER 7: ATTENTION-BASED SPARSE MoE
# ==========================================
class SparseMoEClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=5000,
        embed_dim=64,
        num_heads=4,
        num_experts=4,
        hidden_dim=32,
        num_classes=2,
        padding_idx=1,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.padding_idx = padding_idx

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, num_classes),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, input_ids):
        # [B, S]
        padding_mask = input_ids.eq(self.padding_idx)

        # [B, S, E]
        x = self.embedding(input_ids)

        # Ignore padding during attention.
        attn_output, _ = self.attention(
            x, x, x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        # Residual connection + normalization.
        x = self.layer_norm(x + attn_output)

        # Masked mean pooling.
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        summed = (x * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp(min=1.0)
        features = summed / counts

        # MoE routing.
        gate_logits = self.gate(features)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        top1_weights, top1_indices = torch.max(gate_probs, dim=-1)

        # ONNX-SAFE VECTORIZED ROUTING
        one_hot_mask = torch.nn.functional.one_hot(top1_indices, num_classes=self.num_experts).float()
        routing_weights = one_hot_mask * top1_weights.unsqueeze(-1)
        final_output = torch.zeros(features.size(0), 2, device=features.device)
        
        for i in range(self.num_experts):
            expert_out = self.experts[i](features)
            expert_weight = routing_weights[:, i].unsqueeze(-1)
            final_output += expert_out * expert_weight

        return final_output, gate_probs


def calculate_auxiliary_loss(gate_probs):
    expert_usage = gate_probs.mean(dim=0)
    num_experts = gate_probs.size(1)
    return torch.sum(expert_usage * expert_usage) * num_experts


def train_moe_l7():
    print("\n" + "=" * 50)
    print("[*] INITIATING LAYER 7 ATTENTION MoE (STATIC 70/20/10)")
    print("=" * 50)

    if not os.path.exists(L7_PATHS["X_train"]):
        raise FileNotFoundError(
            f"Missing {L7_PATHS['X_train']}. Run the leak-free preprocess_datasets.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using Device: {device}")

    # Load the pre-split matrices directly and cast to int64 for PyTorch Embeddings
    X_train = np.load(L7_PATHS["X_train"]).astype(np.int64)
    y_train = np.load(L7_PATHS["y_train"]).astype(np.int64)
    X_val = np.load(L7_PATHS["X_val"]).astype(np.int64)
    y_val = np.load(L7_PATHS["y_val"]).astype(np.int64)
    X_test = np.load(L7_PATHS["X_test"]).astype(np.int64)
    y_test = np.load(L7_PATHS["y_test"]).astype(np.int64)

    print(
        f"[*] L7 Shapes -> Train: {X_train.shape} | "
        f"Val: {X_val.shape} | Test: {X_test.shape}"
    )

    # Initialize PyTorch DataLoaders using the loaded arrays
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=L7_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=L7_BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.long),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=L7_BATCH_SIZE,
        shuffle=False,
    )

    model = SparseMoEClassifier(
        vocab_size=L7_VOCAB_SIZE,
        embed_dim=L7_EMBED_DIM,
        num_heads=L7_NUM_HEADS,
        num_experts=L7_NUM_EXPERTS,
        hidden_dim=L7_HIDDEN_DIM,
        padding_idx=1,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    print("\n--- Training Attention MoE ---")

    start_time = time.time()

    for epoch in range(L7_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits, gate_probs = model(batch_X)
            aux_loss = calculate_auxiliary_loss(gate_probs)
            loss = criterion(logits, batch_y) + (0.1 * aux_loss)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                logits, _ = model(batch_X)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(batch_y.numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(
            f"  Epoch {epoch + 1}/{L7_EPOCHS} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Validation Acc: {val_acc * 100:.2f}%"
        )

    print(
        f"[+] Final Training Complete in "
        f"{time.time() - start_time:.2f} seconds."
    )

    # ==========================================
    # Honest 10% Holdout Evaluation
    # ==========================================
    model.eval()

    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits, _ = model(batch_X)
            test_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_labels.extend(batch_y.numpy())

    print_comprehensive_metrics(
        test_labels,
        test_preds,
        test_probs,
        "LAYER 7 (Attention MoE)",
    )

    # ==========================================
    # Latency Measurement
    # ==========================================
    dummy_tensor = torch.tensor(
        X_test[0:1],
        dtype=torch.long,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        latency_us = measure_inference_latency(lambda x: model(x), dummy_tensor)

    print(f"[!] Single-Payload Inference Latency: {latency_us:.2f} microseconds")

    # ==========================================
    # Save Model
    # ==========================================
    save_path = os.path.join(ARTIFACTS_DIR, "l7_moe_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": L7_VOCAB_SIZE,
            "embed_dim": L7_EMBED_DIM,
            "num_heads": L7_NUM_HEADS,
            "num_experts": L7_NUM_EXPERTS,
            "hidden_dim": L7_HIDDEN_DIM,
            "max_seq_length": L7_MAX_SEQ_LENGTH,
            "padding_idx": 1,
        },
        save_path,
    )
    print(f"[✔] Final Layer 7 MoE weights saved to {save_path}")


# ==========================================
# EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    train_xgboost_l4()
    train_moe_l7()