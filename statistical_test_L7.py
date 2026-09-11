import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ==========================================
# 1. ARCHITECTURE DEFINITIONS
# ==========================================
# BASELINE: Standard Attention (No Routing, Single MLP)
class StandardAttentionClassifier(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64, num_heads=4, hidden_dim=32, padding_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
        self.padding_idx = padding_idx

    def forward(self, x):
        mask = x.eq(self.padding_idx)
        emb = self.embedding(x)
        attn_out, _ = self.attention(emb, emb, emb, key_padding_mask=mask, need_weights=False)
        x = self.layer_norm(emb + attn_out)
        valid_mask = (~mask).unsqueeze(-1).float()
        features = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return self.mlp(features)

# PROPOSED: Sparse MoE (From your pipeline)
class SparseMoEClassifier(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64, num_heads=4, num_experts=4, hidden_dim=32, padding_idx=1):
        super().__init__()
        self.num_experts = num_experts
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 2))
            for _ in range(num_experts)
        ])
        self.padding_idx = padding_idx

    def forward(self, x):
        mask = x.eq(self.padding_idx)
        emb = self.embedding(x)
        attn_out, _ = self.attention(emb, emb, emb, key_padding_mask=mask, need_weights=False)
        x = self.layer_norm(emb + attn_out)
        valid_mask = (~mask).unsqueeze(-1).float()
        features = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        
        gate_logits = self.gate(features)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        top1_weights, top1_indices = torch.max(gate_probs, dim=-1)
        
        one_hot_mask = torch.nn.functional.one_hot(top1_indices, num_classes=self.num_experts).float()
        routing_weights = one_hot_mask * top1_weights.unsqueeze(-1)
        final_output = torch.zeros(features.size(0), 2, device=features.device)
        
        for i in range(self.num_experts):
            final_output += self.experts[i](features) * routing_weights[:, i].unsqueeze(-1)
            
        return final_output, gate_probs

def calculate_auxiliary_loss(gate_probs):
    expert_usage = gate_probs.mean(dim=0)
    return torch.sum(expert_usage * expert_usage) * gate_probs.size(1)

def calculate_routing_entropy(gate_probs):
    # H = -sum(p * log(p))
    return -torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=-1).mean()

# ==========================================
# 2. LOAD DATA & SETUP K-FOLD
# ==========================================
print("\n[*] Loading L7 Train Matrix for K-Fold CV...")
X_train = np.load("processed_data/l7_X_train.npy").astype(np.int64)
y_train = np.load("processed_data/l7_y_train.npy").astype(np.int64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using Device: {device}")

K = 5
EPOCHS = 15 # Set to 5 for CV speed; use 10-15 for actual final paper numbers
BATCH_SIZE = 256
cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

baseline_f1_scores = []
moe_f1_scores = []
moe_fold_entropies = []
moe_fold_aux_losses = []

dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))

# ==========================================
# 3. K-FOLD TRAINING LOOP
# ==========================================
print(f"\n[*] Executing {K}-Fold Cross Validation (Epochs per fold: {EPOCHS})...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"\n--- Fold {fold + 1}/{K} ---")
    
    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
    
    # Init Models
    baseline_model = StandardAttentionClassifier().to(device)
    moe_model = SparseMoEClassifier(num_experts=4).to(device)
    
    opt_base = optim.AdamW(baseline_model.parameters(), lr=1e-3)
    opt_moe = optim.AdamW(moe_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train Loop for this fold
    for epoch in range(EPOCHS):
        baseline_model.train()
        moe_model.train()
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Base Step
            opt_base.zero_grad()
            loss_base = criterion(baseline_model(batch_X), batch_y)
            loss_base.backward()
            opt_base.step()
            
            # MoE Step
            opt_moe.zero_grad()
            logits_moe, gate_probs = moe_model(batch_X)
            loss_moe = criterion(logits_moe, batch_y) + (0.1 * calculate_auxiliary_loss(gate_probs))
            loss_moe.backward()
            opt_moe.step()

    # Evaluation Step
    baseline_model.eval()
    moe_model.eval()
    
    base_preds, moe_preds, fold_y = [], [], []
    fold_entropy, fold_aux_loss = 0.0, 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            fold_y.extend(batch_y.numpy())
            
            # Baseline Eval
            base_logits = baseline_model(batch_X)
            base_preds.extend(torch.argmax(base_logits, dim=1).cpu().numpy())
            
            # MoE Eval
            moe_logits, gate_probs = moe_model(batch_X)
            moe_preds.extend(torch.argmax(moe_logits, dim=1).cpu().numpy())
            
            fold_entropy += calculate_routing_entropy(gate_probs).item()
            fold_aux_loss += calculate_auxiliary_loss(gate_probs).item()
            
    # Calculate Fold Metrics
    baseline_f1 = f1_score(fold_y, base_preds)
    moe_f1 = f1_score(fold_y, moe_preds)
    
    baseline_f1_scores.append(baseline_f1)
    moe_f1_scores.append(moe_f1)
    
    # Average entropy and aux loss over the batches in validation
    moe_fold_entropies.append(fold_entropy / len(val_loader))
    moe_fold_aux_losses.append(fold_aux_loss / len(val_loader))
    
    print(f"Base F1: {baseline_f1:.4f} | MoE F1: {moe_f1:.4f} | Aux Loss: {moe_fold_aux_losses[-1]:.4f} | Entropy: {moe_fold_entropies[-1]:.4f}")

# ==========================================
# 4. STATISTICAL HYPOTHESIS TESTING
# ==========================================
print("\n[*] Performing Statistical Testing...")
score_differences = np.array(moe_f1_scores) - np.array(baseline_f1_scores)
_, p_shapiro = shapiro(score_differences)

if p_shapiro > 0.05:
    t_stat, p_value = ttest_rel(moe_f1_scores, baseline_f1_scores)
    test_used = "Paired t-Test"
else:
    w_stat, p_value = wilcoxon(moe_f1_scores, baseline_f1_scores)
    test_used = "Wilcoxon Signed-Rank Test"
    t_stat = w_stat 

mean_entropy = np.mean(moe_fold_entropies)
mean_aux_loss = np.mean(moe_fold_aux_losses)
ideal_entropy = np.log(4) # log(num_experts)

# ==========================================
# 5. Q1 JOURNAL OUTPUT FORMATTING
# ==========================================
print("\n" + "=" * 65)
print(" STATISTICAL VALIDATION RESULTS FOR SECTION 6 (L7 MoE ABLATION) ")
print("=" * 65)
print(f"Baseline           : Single-Expert Attention FFN")
print(f"Proposed           : Sparse MoE (4 Experts)")
print(f"Metric Evaluated   : F1-Score")
print(f"Test Applied       : {test_used}")
print(f"Test Statistic     : {t_stat:.4f}")
print(f"p-value            : {p_value:.6f}")
print("-" * 65)
print(f"Mean Routing Entropy : {mean_entropy:.4f} (Ideal max for 4 experts is {ideal_entropy:.4f})")
print(f"Mean Auxiliary Loss  : {mean_aux_loss:.4f} (Closer to 0.25 indicates perfect load balancing)")
print("-" * 65)

if p_value < 0.05:
    print("Conclusion: The MoE routing mechanism provides a STATISTICALLY SIGNIFICANT")
    print("improvement over a standard transformer architecture.")
else:
    print("Conclusion: Brutal truth - your MoE routing did not significantly")
    print("outperform a standard dense layer. Your MoE is unnecessary complexity.")
print("=" * 65)