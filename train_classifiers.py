import os
import time
import numpy as np
import xgboost as xgb
import torch
import torch.onnx
from sklearn.model_selection import StratifiedKFold
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Import the architecture and helpers from your updated training file
from train_classifiers_updated import (
    create_70_20_10_split, load_l7_token_data, SparseMoEClassifier,
    calculate_auxiliary_loss,
    L7_VOCAB_SIZE, L7_EMBED_DIM, L7_NUM_HEADS, L7_NUM_EXPERTS, L7_HIDDEN_DIM, L7_EPOCHS
)

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
PROCESSED_DIR = "processed_data"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

L4_FEAT_PATH = os.path.join(PROCESSED_DIR, "l4_features.npy")
L4_LABELS_PATH = os.path.join(PROCESSED_DIR, "l4_labels.npy")

CV_FOLDS = 3

def train_and_export_l4():
    print("\n" + "=" * 50)
    print("[*] L4: CV, TRAINING & ONNX EXPORT")
    print("=" * 50)

    if not os.path.exists(L4_FEAT_PATH):
        raise FileNotFoundError(f"Missing {L4_FEAT_PATH}. Check your paths.")

    X = np.load(L4_FEAT_PATH)
    y = np.load(L4_LABELS_PATH)

    # 1. Cross Validation
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    cv_acc = []
    print(f"[*] Running {CV_FOLDS}-Fold CV...")
    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1)
        model.fit(X[train_idx], y[train_idx])
        cv_acc.append(model.score(X[val_idx], y[val_idx]))
    print(f"[*] L4 CV Mean Accuracy: {np.mean(cv_acc)*100:.2f}%\n")

    # 2. Final Training
    X_train, X_val, X_test, y_train, y_val, y_test = create_70_20_10_split(X, y)
    final_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Save standard JSON for the SHAP/XAI Explainer
    json_path = os.path.join(ARTIFACTS_DIR, "l4_xgb_model.json")
    final_model.save_model(json_path)

    # 3. Export XGBoost to ONNX (Bypassing Treelite completely)
    print("[*] Exporting XGBoost to ONNX graph...")
    
    # Define the input shape for the ONNX graph (batch_size is dynamic 'None')
    num_features = X_train.shape[1]
    initial_type = [('float_input', FloatTensorType([None, num_features]))]
    
    # Convert and serialize
    onnx_model = onnxmltools.convert_xgboost(final_model, initial_types=initial_type)
    onnx_path = os.path.join(ARTIFACTS_DIR, "l4_xgb_model.onnx")
    
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"[✔] L4 successfully exported to {onnx_path}")

def train_and_export_l7():
    print("\n" + "=" * 50)
    print("[*] L7: TRAINING & ONNX EXPORT")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_l7_token_data()
    X_train, X_val, X_test, y_train, y_val, y_test = create_70_20_10_split(X, y)

    model = SparseMoEClassifier(
        vocab_size=L7_VOCAB_SIZE, embed_dim=L7_EMBED_DIM, num_heads=L7_NUM_HEADS,
        num_experts=L7_NUM_EXPERTS, hidden_dim=L7_HIDDEN_DIM, padding_idx=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("[*] Training Attention MoE...")
    for epoch in range(L7_EPOCHS):
        model.train()
        batch_x = torch.tensor(X_train[:256], dtype=torch.long).to(device)
        batch_y = torch.tensor(y_train[:256], dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        logits, gate_probs = model(batch_x)
        loss = criterion(logits, batch_y) + (0.1 * calculate_auxiliary_loss(gate_probs))
        loss.backward()
        optimizer.step()

    # ONNX Export
    print("[*] Exporting PyTorch model to ONNX...")
    model.eval()
    dummy_input = torch.tensor(X_test[0:1], dtype=torch.long).to(device)
    onnx_path = os.path.join(ARTIFACTS_DIR, "l7_moe_model.onnx")
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input_ids'], output_names=['logits', 'gate_probs'],
        dynamic_axes={'input_ids': {0: 'batch_size'}}
    )
    print(f"[✔] L7 MoE exported to {onnx_path}")

if __name__ == "__main__":
    train_and_export_l4()
    train_and_export_l7()