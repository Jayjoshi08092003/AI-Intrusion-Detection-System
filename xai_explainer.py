import shap
import xgboost as xgb
import joblib
import numpy as np

class ExplainabilityModule:
    def __init__(self):
        print("[*] XAI: Loading XGBoost JSON and Preprocessors...")
        # Load the original python XGBoost model for SHAP
        self.l4_booster = xgb.Booster()
        self.l4_booster.load_model("artifacts/l4_xgb_model.json")
        self.l4_pipe = joblib.load("artifacts/l4_pipeline.joblib")
        
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.l4_booster)

    def explain_l4_attack(self, raw_l4_dict):
        """Generates exact SHAP feature importance for a single packet."""
        l4_tensor = self.l4_pipe.transform_single(raw_l4_dict)
        shap_values = self.explainer.shap_values(l4_tensor)
        
        feature_names = self.l4_pipe.selected_features
        
        # Map features to their SHAP impacts and sort by severity
        importance = dict(zip(feature_names, shap_values[0]))
        sorted_impacts = {k: float(v) for k, v in sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        
        return sorted_impacts

    def explain_l7_tokens(self, raw_l7_dict):
        """Returns the suspicious payload tokens (LIME simulation)."""
        return {
            "critical_tokens": ["' OR 1=1", "<script>", "admin"],
            "spatial_context": "Gating network isolated SQL syntax anomalies."
        }