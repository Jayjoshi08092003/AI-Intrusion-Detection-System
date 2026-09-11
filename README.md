# Two-Layer Intrusion Detection System

This project detects malicious traffic at two network layers:

- **Layer 4:** numeric network-flow features from CICIDS2017, classified with XGBoost.
- **Layer 7:** HTTP request text from the CSIC 2010 dataset, converted to BPE token IDs and classified with an attention-based sparse Mixture-of-Experts model.

The normal workflow is for Run:

```text
1. preprocess_datasets_updated.py
2.train_classifier.py
3. train_classifiers_updated.py
4.xai_explainer.py
5.llm_agent.py
6. Production_dispatcher.py
7. statistical_test_L4.py and statistical_test_L7.py
8. master_ids.py
   -> production hot-path inference
   -> optional XAI explanation
   -> optional Groq LLM incident report
```

This is a research/prototype system. It is not a complete packet-capture agent, firewall, or production SOC integration.

## Project structure

```text
.
|-- preprocess_datasets_updated.py    Leak-free preprocessing and dataset splitting
|-- train_classifiers_updated.py      Main XGBoost and PyTorch training script
|-- train_classifiers.py               Older training/export script; exports ONNX models
|-- statistical_test_L4.py             L4 XGBoost vs Random Forest significance test
|-- statistical_test_L7.py             L7 Attention baseline vs sparse MoE test
|-- production_dispatcher.py           Concurrent ONNX inference dispatcher
|-- master_ids.py                      Main hot-path, XAI, and LLM demonstration
|-- xai_explainer.py                   SHAP L4 explanation and L7 context output
|-- llm_agent.py                       Groq incident-report generation
|-- train.py                            L4 label inspection utility
|-- vocablory.py                        L7 vocabulary inspection utility
|-- DATASETS/                           CSIC HTTP dataset and related data
|-- NETWORK LAYER MODEL/                CICIDS2017 flow dataset and related files
|-- processed_data/                     Generated NumPy training/test arrays
`-- artifacts/                          Saved pipelines and trained models
```

## 1. Install and activate the environment

Run all commands from the repository root, the folder containing this README. PowerShell setup:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn scipy joblib tokenizers xgboost torch onnxruntime shap groq python-dotenv onnxmltools
```

The project uses Python 3.10 or newer. If the virtual environment is already present, activate it and skip the `venv` creation command.

### Libraries used by the codebase

| Library | Use |
| --- | --- |
| `numpy` | Arrays, labels, token matrices, numerical calculations |
| `pandas` | CSV loading and dataset preparation |
| `scikit-learn` | Scaling, feature selection, train/test splitting, metrics, cross-validation, Random Forest baseline |
| `scipy` | Shapiro-Wilk, paired t-test, and Wilcoxon significance tests |
| `joblib` | Saving and loading preprocessing pipelines |
| `tokenizers` | Byte-Level BPE tokenizer for HTTP requests |
| `xgboost` | Layer 4 binary classifier |
| `torch` | Layer 7 attention and sparse MoE classifier, training, and ONNX export |
| `onnxruntime` | Production ONNX inference |
| `onnxmltools` | XGBoost-to-ONNX conversion in the older export script |
| `shap` | TreeSHAP explanations for the L4 model |
| `groq` | Groq API client for SOC reports |
| `python-dotenv` | Loads local environment variables from `.env` |

The Python standard library is also used: `os`, `re`, `time`, `uuid`, `hashlib`, `urllib.parse`, `concurrent.futures`, `json`, and `collections`.

## 2. Add the Groq API key

The API key is only needed for the final LLM incident-report step. Set it in the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "your-new-groq-api-key"
```

The `llm_agent.py` module calls `load_dotenv()`, so a local `.env` file can also contain:

```text
GROQ_API_KEY=your-new-groq-api-key
```

Never commit the key to GitHub or share it in source code. If a real key has already been exposed in `.env`, revoke it in the Groq dashboard and create a new one. Keep `.env` untracked in `.gitignore`.

The LLM model configured in `llm_agent.py` is `openai/gpt-oss-120b`. The LLM call sends classifier scores, top SHAP features, selected L7 context, a request ID, and the SHA-256 hash of the request body. Do not send sensitive production payloads to an external service without approval.

## 3. Check dataset paths

The current paths are defined at the top of `preprocess_datasets_updated.py`:

```text
L4: NETWORK LAYER MODEL/cicids2017_cleaned.csv
L7: DATASETS/csic_database.csv
```

The script currently stores absolute Windows paths. If the project is moved, update `L4_DATASET_PATH` and `L7_DATASET_PATH` before running it.

### L4 input: CICIDS2017

The L4 pipeline finds a label column such as `Label`, `Attack Type`, or `Attack Type AND Label`. It maps benign/normal/`0` to `0` and all other labels to `1`. It keeps numeric features, replaces infinite values, fills missing values with zero, fits scaling and feature selection only on the training split, and selects up to 50 features.

### L7 input: CSIC 2010

The L7 pipeline combines `Method`, `URL`, `User-Agent`, `host`, `content-type`, and `content`. URL encoding and escaped hex values are decoded, text is normalized, and a 5,000-entry Byte-Level BPE tokenizer is trained only on the L7 training split. Requests are padded/truncated to 128 token IDs.

## 4. Run preprocessing first

This is the first project command after installation:

```powershell
python preprocess_datasets_updated.py
```

The script performs a stratified 70/20/10 split before fitting the scaler, correlation pruning, decision tree, or tokenizer. It creates these files:

```text
processed_data/l4_X_train.npy
processed_data/l4_y_train.npy
processed_data/l4_X_val.npy
processed_data/l4_y_val.npy
processed_data/l4_X_test.npy
processed_data/l4_y_test.npy
processed_data/l7_X_train.npy
processed_data/l7_y_train.npy
processed_data/l7_X_val.npy
processed_data/l7_y_val.npy
processed_data/l7_X_test.npy
processed_data/l7_y_test.npy
artifacts/l4_pipeline.joblib
artifacts/l7_pipeline.joblib
```

These files are required by the updated training script and statistical tests. Do not delete or mix them with outputs from the older preprocessing prototype.

## 5. Train the updated classifiers

After preprocessing finishes, run:

```powershell
python train_classifiers_updated.py
```

This trains and evaluates both models using the saved 70/20/10 arrays.

### L4 training

The L4 model is an XGBoost binary classifier with 200 estimators, depth 6, learning rate `0.1`, and validation-based early stopping. It prints accuracy, precision, recall, F1, AUC-ROC, confusion-matrix values, and latency, then saves:

```text
artifacts/l4_xgb_model.json
```

### L7 training

The L7 model uses a 5,000-token vocabulary, 64-dimensional embeddings, four attention heads, masked mean pooling, four top-1 routed experts, 32 hidden units per expert, and sequence length 128. It trains for 15 epochs with AdamW and an auxiliary expert-balancing loss, then saves:

```text
artifacts/l7_moe_model.pth
```

## 6. Export ONNX models for `master_ids.py`

`production_dispatcher.py` loads ONNX files. The updated training script saves the Python XGBoost JSON and PyTorch weights, but it does not export ONNX. To create or refresh the ONNX files, run:

```powershell
python train_classifiers.py
```

That older script imports the current model definitions, trains exportable models, and writes:

```text
artifacts/l4_xgb_model.json
artifacts/l4_xgb_model.onnx
artifacts/l7_moe_model.onnx
```

Because this export script is an older implementation, verify the resulting model files before using them for reported experiments. Whenever preprocessing settings, feature count, tokenizer settings, or model architecture changes, regenerate the ONNX files.

## 7. Run statistical significance tests

Run these after preprocessing. They use the training split for five-fold cross-validation and do not use the final 10 percent holdout.

### L4 statistical test

```powershell
python statistical_test_L4.py
```

This compares XGBoost with a 200-tree Random Forest baseline using fold-level F1 scores. It runs a Shapiro-Wilk normality test on paired score differences, then chooses either a paired t-test or Wilcoxon signed-rank test. It prints the test statistic, p-value, and a threshold-based conclusion using `alpha = 0.05`.

### L7 statistical test

```powershell
python statistical_test_L7.py
```

This compares the proposed four-expert sparse MoE with a standard single-MLP attention classifier over five folds. It trains both models per fold, compares F1 scores, selects a paired t-test or Wilcoxon signed-rank test after Shapiro-Wilk, and reports routing entropy and auxiliary loss.

The statistical scripts require:

```text
processed_data/l4_X_train.npy
processed_data/l4_y_train.npy
processed_data/l7_X_train.npy
processed_data/l7_y_train.npy
```

Statistical significance does not by itself prove practical superiority. Report the fold scores, mean difference, test choice, p-value, and effect size when using these results in a paper or evaluation.

## 8. Run the master hot path, XAI, and LLM

After preprocessing and model/ONNX generation, run the main demonstration:

```powershell
python master_ids.py
```

`master_ids.py` performs the final runtime sequence:

1. Loads both fitted pipelines and both ONNX models.
2. Creates a synthetic L4 flow and L7 HTTP request.
3. Generates a UUID request ID and hashes the L7 body with SHA-256.
4. Runs L4 and L7 inference concurrently through `ProductionDispatcher`.
5. Prints attack probabilities and inference timing.
6. If either score is above `0.5`, asks whether to start the cold path.
7. If you answer `y`, calls `xai_explainer.py` and then `llm_agent.py`.

XAI and LLM are therefore run through the master script, not as separate commands:

```text
master_ids.py
  -> ExplainabilityModule.explain_l4_attack()
  -> ExplainabilityModule.explain_l7_tokens()
  -> SOCReasoningAgent.generate_incident_report()
```

The L4 explanation uses TreeSHAP and the saved JSON model. The current L7 explanation returns illustrative fixed token context and is a placeholder, not a calculated LIME/SHAP explanation.

The dispatcher requires:

```text
artifacts/l4_pipeline.joblib
artifacts/l7_pipeline.joblib
artifacts/l4_xgb_model.onnx
artifacts/l7_moe_model.onnx
```

## Optional utilities

Inspect the L4 label column:

```powershell
python train.py
```

Inspect the approximate L7 vocabulary:

```powershell
python vocablory.py
```

The `Ingestion/` directory contains an earlier routing and feeder prototype. Its `runtime_feeder.py` expects legacy dense embeddings and is not the current path. `end_to_end_test.py` also references older class names, so use `master_ids.py` for the current demonstration.

## Generated files and roles

| File | Role |
| --- | --- |
| `l4_X_train.npy`, `l4_X_val.npy`, `l4_X_test.npy` | Scaled and selected L4 features |
| `l4_y_train.npy`, `l4_y_val.npy`, `l4_y_test.npy` | L4 binary labels |
| `l7_X_train.npy`, `l7_X_val.npy`, `l7_X_test.npy` | Padded L7 token IDs with sequence length 128 |
| `l7_y_train.npy`, `l7_y_val.npy`, `l7_y_test.npy` | L7 binary labels |
| `l4_pipeline.joblib` | L4 scaler, selected features, and metadata |
| `l7_pipeline.joblib` | Fitted L7 tokenizer and padding metadata |
| `l4_xgb_model.json` | Python XGBoost model used by SHAP |
| `l4_xgb_model.onnx` | L4 production inference model |
| `l7_moe_model.pth` | PyTorch L7 weights |
| `l7_moe_model.onnx` | L7 production inference model |

## Quick verification

```powershell
python -c "import numpy as np; print('L4 train:', np.load('processed_data/l4_X_train.npy').shape); print('L7 train:', np.load('processed_data/l7_X_train.npy').shape)"
python -c "import joblib; joblib.load('artifacts/l4_pipeline.joblib'); joblib.load('artifacts/l7_pipeline.joblib'); print('Pipelines loaded successfully')"
python master_ids.py
```

Expected L7 arrays have shape `[number_of_requests, 128]`. A successful master run prints both layer predictions, their probabilities, and measured inference times before optionally asking for XAI/LLM analysis.

## Important limitations

- Dataset paths in the updated preprocessing script are absolute Windows paths.
- The current models classify benign versus attack; they do not reliably identify attack families.
- XAI and LLM processing can expose payload-derived information, so review privacy requirements before enabling it.
- There is no pinned `requirements.txt`, automated test suite, or fully automated ONNX export pipeline.
- The statistical scripts and the updated training script depend on the leak-free split files created by `preprocess_datasets_updated.py`.

cicids2017_cleaned.csv link:https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed

csic 2010 link:https://www.kaggle.com/datasets/ispangler/csic-2010-web-application-attacks
