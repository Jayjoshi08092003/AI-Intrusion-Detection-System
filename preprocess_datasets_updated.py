import os
import re
import joblib
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, AddedToken

# =====================================================================
# EXACT FILE PATHS
# =====================================================================
L4_DATASET_PATH = r"C:\Users\ASUS\Desktop\IDS Datasets\NETWORK LAYER MODEL\cicids2017_cleaned.csv"
L7_DATASET_PATH = r"C:\Users\ASUS\Desktop\IDS Datasets\DATASETS\csic_database.csv"
ARTIFACTS_DIR = "artifacts"
PROCESSED_DIR = "processed_data"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =====================================================================
# 1. LAYER 4 PIPELINE (LEAK-FREE)
# =====================================================================
class L4DatasetPipeline:
    def __init__(self, correlation_threshold=0.999, top_k_features=50):
        self.correlation_threshold = correlation_threshold
        self.top_k_features = top_k_features
        self.scaler = StandardScaler()
        self.collinear_dropped_cols = []
        self.selected_features = []
        self.input_feature_names = []

    def fit_and_export(self, filepath: str):
        print(f"\n[*] [L4] Loading CICIDS2017 safely via chunks from: {filepath}")

        chunk_size = 250000
        sample_fraction = 0.15
        df_list = []
        
        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size, low_memory=False)):
            print(f"   -> Processing chunk {i+1}...")
            df_list.append(chunk.sample(frac=sample_fraction, random_state=42))

        df = pd.concat(df_list, ignore_index=True)
        print(f"[*] [L4] Aggregated {len(df)} rows after sampling.")

        df.columns = df.columns.str.strip()
        
        # Dynamically locate label column
        label_col = None
        candidates = ["attack type and label", "label", "classification", "attack type", "attack_type"]
        lower_cols = {col.lower(): col for col in df.columns}
        for candidate in candidates:
            if candidate in lower_cols:
                label_col = lower_cols[candidate]
                break

        if not label_col:
            for col in df.columns:
                if "label" in col.lower() or "attack" in col.lower():
                    label_col = col
                    break

        if not label_col:
            raise KeyError("Could not find a valid label column in the L4 dataset.")

        # Clean Targets & Features
        y = df[label_col].astype(str).str.upper().apply(
            lambda x: 0 if "BENIGN" in x or "NORMAL" in x or x.strip() == "0" else 1
        ).values

        X = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=[np.number])
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0.0, inplace=True)
        self.input_feature_names = X.columns.tolist()

        # ---------------------------------------------------------
        # THE FIX: 70/20/10 Split BEFORE fitting anything
        # ---------------------------------------------------------
        print("[*] [L4] Performing 70/20/10 Split to prevent data leakage...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        # 1. Fit Scaler ONLY on Train
        print("[*] [L4] Fitting Scaler & pruning collinear features strictly on Train set...")
        self.scaler.fit(X_train)
        
        # Transform all three
        X_train_s = pd.DataFrame(self.scaler.transform(X_train), columns=self.input_feature_names)
        X_val_s = pd.DataFrame(self.scaler.transform(X_val), columns=self.input_feature_names)
        X_test_s = pd.DataFrame(self.scaler.transform(X_test), columns=self.input_feature_names)

        # 2. Compute Correlation ONLY on Train
        corr_matrix = X_train_s.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.collinear_dropped_cols = [
            col for col in upper_tri.columns if any(upper_tri[col] > self.correlation_threshold)
        ]

        # Drop from all three
        X_train_p = X_train_s.drop(columns=self.collinear_dropped_cols)
        X_val_p = X_val_s.drop(columns=self.collinear_dropped_cols)
        X_test_p = X_test_s.drop(columns=self.collinear_dropped_cols)

        # 3. Fit Decision Tree ONLY on Train
        print(f"[*] [L4] Learning Feature Importance (Top {self.top_k_features}) strictly on Train set...")
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(X_train_p, y_train)

        importances = pd.Series(dt.feature_importances_, index=X_train_p.columns)
        self.selected_features = importances.sort_values(ascending=False).head(self.top_k_features).index.tolist()

        # Extract final arrays
        X_train_final = X_train_p[self.selected_features].values.astype(np.float32)
        X_val_final = X_val_p[self.selected_features].values.astype(np.float32)
        X_test_final = X_test_p[self.selected_features].values.astype(np.float32)

        # Save artifacts
        np.save(os.path.join(PROCESSED_DIR, "l4_X_train.npy"), X_train_final)
        np.save(os.path.join(PROCESSED_DIR, "l4_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, "l4_X_val.npy"), X_val_final)
        np.save(os.path.join(PROCESSED_DIR, "l4_y_val.npy"), y_val)
        np.save(os.path.join(PROCESSED_DIR, "l4_X_test.npy"), X_test_final)
        np.save(os.path.join(PROCESSED_DIR, "l4_y_test.npy"), y_test)
        joblib.dump(self, os.path.join(ARTIFACTS_DIR, "l4_pipeline.joblib"))

        print(f"[SUCCESS] L4 saved: Train {X_train_final.shape}, Val {X_val_final.shape}, Test {X_test_final.shape}")

    def transform_single(self, raw_l4_dict: dict):
        row = [float(raw_l4_dict.get(col, 0.0)) for col in self.input_feature_names]
        scaled = self.scaler.transform([row])
        df_single = pd.DataFrame(scaled, columns=self.input_feature_names)
        return df_single[self.selected_features].values.astype(np.float32)

# =====================================================================
# 2. LAYER 7 PIPELINE (LEAK-FREE)
# =====================================================================
class L7DatasetPipeline:
    def __init__(self, vocab_size=5000, max_seq_length=128):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = None
        self.pad_token_id = None

    @staticmethod
    def deobfuscate(text: str) -> str:
        if not isinstance(text, str):
            return ""
        for _ in range(2):
            decoded = urllib.parse.unquote(text)
            if decoded == text: break
            text = decoded
        text = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)
        return re.sub(r"\/{2,}", "/", text).lower().strip()

    def row_to_text(self, row: dict) -> str:
        tokens = [
            str(row.get("Method", "")), self.deobfuscate(str(row.get("URL", ""))),
            str(row.get("User-Agent", "")), str(row.get("host", "")),
            str(row.get("content-type", "")), self.deobfuscate(str(row.get("content", "")))
        ]
        return " ".join([t for t in tokens if t])

    def encode_corpus(self, corpus: list) -> np.ndarray:
        all_token_ids = []
        for i in range(0, len(corpus), 2048):
            batch_text = corpus[i : i + 2048]
            encodings = self.tokenizer.encode_batch(batch_text)
            all_token_ids.extend([enc.ids for enc in encodings])
        return np.asarray(all_token_ids, dtype=np.int64)

    def fit_and_export(self, filepath: str):
        print(f"\n[*] [L7] Loading CSIC 2010 from: {filepath}")
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip()

        label_col = "classification" if "classification" in df.columns else "label"
        y = df[label_col].astype(int).values

        print("[*] [L7] Protocol De-obfuscation & Corpus Generation...")
        corpus = [self.row_to_text(row) for _, row in df.iterrows()]

        # ---------------------------------------------------------
        # THE FIX: 70/20/10 Split BEFORE fitting Tokenizer
        # ---------------------------------------------------------
        c_train, c_temp, y_train, y_temp = train_test_split(corpus, y, test_size=0.3, random_state=42, stratify=y)
        c_val, c_test, y_val, y_test = train_test_split(c_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

        print("[*] [L7] Training BPE Tokenizer STRICTLY on Train set...")
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        special_tokens = [
            AddedToken("[UNK]", lstrip=False, rstrip=False),
            AddedToken("[PAD]", lstrip=False, rstrip=False),
            AddedToken("[CLS]", lstrip=False, rstrip=False),
            AddedToken("[SEP]", lstrip=False, rstrip=False),
        ]

        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=special_tokens)
        
        # Train ONLY on c_train
        self.tokenizer.train_from_iterator(c_train, trainer=trainer)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.tokenizer.enable_padding(pad_id=self.pad_token_id, pad_token="[PAD]", length=self.max_seq_length)

        print("[*] [L7] Transforming splits to Token IDs...")
        X_train_final = self.encode_corpus(c_train)
        X_val_final = self.encode_corpus(c_val)
        X_test_final = self.encode_corpus(c_test)

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        
        np.save(os.path.join(PROCESSED_DIR, "l7_X_train.npy"), X_train_final)
        np.save(os.path.join(PROCESSED_DIR, "l7_y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, "l7_X_val.npy"), X_val_final)
        np.save(os.path.join(PROCESSED_DIR, "l7_y_val.npy"), y_val)
        np.save(os.path.join(PROCESSED_DIR, "l7_X_test.npy"), X_test_final)
        np.save(os.path.join(PROCESSED_DIR, "l7_y_test.npy"), y_test)
        joblib.dump(self, os.path.join(ARTIFACTS_DIR, "l7_pipeline.joblib"))
        
        print(f"[SUCCESS] L7 saved: Train {X_train_final.shape}, Val {X_val_final.shape}, Test {X_test_final.shape}")

    def transform_single(self, raw_l7_dict: dict):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        cleaned_text = self.row_to_text(raw_l7_dict)
        return np.asarray([self.tokenizer.encode(cleaned_text).ids], dtype=np.int64)

# =====================================================================
# EXECUTION ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    l4_pipe = L4DatasetPipeline(correlation_threshold=0.999, top_k_features=50)
    l4_pipe.fit_and_export(L4_DATASET_PATH)

    l7_pipe = L7DatasetPipeline(vocab_size=5000, max_seq_length=128)
    l7_pipe.fit_and_export(L7_DATASET_PATH)

    print("\n[✔] Data ingestion, feature refining, and model persistence complete.")