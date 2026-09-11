import os
import time
import numpy as np
import onnxruntime as ort
import joblib
import concurrent.futures
import __main__

from preprocess_datasets_updated import L4DatasetPipeline, L7DatasetPipeline

__main__.L4DatasetPipeline = L4DatasetPipeline
__main__.L7DatasetPipeline = L7DatasetPipeline


class ProductionDispatcher:

    def __init__(self):

        print("[*] Loading Preprocessors...")

        self.l4_pipe = joblib.load(
            "artifacts/l4_pipeline.joblib"
        )

        self.l7_pipe = joblib.load(
            "artifacts/l7_pipeline.joblib"
        )

        print("[*] Spinning up ONNX Runtime C++ Execution Engines...")

        sess_options = ort.SessionOptions()

        # One ORT thread per model.
        # This allows L4 and L7 to run independently.
        sess_options.intra_op_num_threads = 1

        self.l4_session = ort.InferenceSession(
            "artifacts/l4_xgb_model.onnx",
            sess_options,
            providers=["CPUExecutionProvider"]
        )

        self.l7_session = ort.InferenceSession(
            "artifacts/l7_moe_model.onnx",
            sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Cache input names
        self.l4_input_name = self.l4_session.get_inputs()[0].name
        self.l7_input_name = self.l7_session.get_inputs()[0].name


    def run_hot_path(
        self,
        raw_l4_data=None,
        raw_l7_data=None,
        req_id=None
    ):

        dispatch_start = time.perf_counter()

        # ============================================================
        # 1. DETERMINE WHICH FEATURES ARE AVAILABLE
        # ============================================================

        l4_available = raw_l4_data is not None
        l7_available = raw_l7_data is not None

        print("\n" + "=" * 40)
        print(" FEATURE AVAILABILITY")
        print("=" * 40)

        print(
            f"[*] L4 Features : "
            f"{'AVAILABLE' if l4_available else 'NOT AVAILABLE'}"
        )

        print(
            f"[*] L7 Features : "
            f"{'AVAILABLE' if l7_available else 'NOT AVAILABLE'}"
        )


        # ============================================================
        # 2. NOTHING AVAILABLE
        # ============================================================

        if not l4_available and not l7_available:

            print("[!] No L4 or L7 features available.")
            print("[!] No model inference performed.")

            return {
                "req_id": req_id,
                "l4_prediction": None,
                "l7_prediction": None,
                "l4_time_us": None,
                "l7_time_us": None,
                "total_time_us":
                    (time.perf_counter() - dispatch_start) * 1_000_000
            }


        # ============================================================
        # 3. PREPROCESS ONLY AVAILABLE FEATURES
        # ============================================================

        l4_tensor = None
        l7_tensor = None

        if l4_available:

            print("[*] Preprocessing L4 features...")

            l4_tensor = self.l4_pipe.transform_single(
                raw_l4_data
            )

        if l7_available:

            print("[*] Preprocessing L7 features...")

            l7_tensor = self.l7_pipe.transform_single(
                raw_l7_data
            )


        # ============================================================
        # 4. L4 INFERENCE FUNCTION
        # ============================================================

        def infer_l4():

            start_l4 = time.perf_counter()

            l4_inputs = {
                self.l4_input_name: l4_tensor
            }

            l4_outputs = self.l4_session.run(
                None,
                l4_inputs
            )

            exec_time = (
                time.perf_counter() - start_l4
            ) * 1_000_000

            # XGBoost ONNX probability output
            pred = l4_outputs[1][0][1]

            return pred, exec_time


        # ============================================================
        # 5. L7 INFERENCE FUNCTION
        # ============================================================

        def infer_l7():

            start_l7 = time.perf_counter()

            l7_inputs = {
                self.l7_input_name: l7_tensor
            }

            l7_outputs = self.l7_session.run(
                None,
                l7_inputs
            )

            exec_time = (
                time.perf_counter() - start_l7
            ) * 1_000_000

            # Convert logits -> probability
            logits = l7_outputs[0][0]

            # Numerically stable softmax
            logits = logits - np.max(logits)

            exp_logits = np.exp(logits)

            probs = exp_logits / np.sum(exp_logits)

            pred = probs[1]

            return pred, exec_time


        # ============================================================
        # 6. EXECUTION STRATEGY
        # ============================================================

        l4_pred = None
        l7_pred = None

        l4_time = None
        l7_time = None


        # ------------------------------------------------------------
        # BOTH AVAILABLE
        # ------------------------------------------------------------

        if l4_available and l7_available:

            print("\n[*] Both L4 and L7 features available.")
            print("[*] Firing L4 + L7 concurrently...")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=2
            ) as executor:

                future_l4 = executor.submit(
                    infer_l4
                )

                future_l7 = executor.submit(
                    infer_l7
                )

                l4_pred, l4_time = future_l4.result()

                l7_pred, l7_time = future_l7.result()


        # ------------------------------------------------------------
        # ONLY L4 AVAILABLE
        # ------------------------------------------------------------

        elif l4_available:

            print("\n[*] Only L4 features available.")
            print("[*] Firing L4 model only...")

            l4_pred, l4_time = infer_l4()


        # ------------------------------------------------------------
        # ONLY L7 AVAILABLE
        # ------------------------------------------------------------

        elif l7_available:

            print("\n[*] Only L7 features available.")
            print("[*] Firing L7 model only...")

            l7_pred, l7_time = infer_l7()


        # ============================================================
        # 7. TOTAL PIPELINE TIME
        # ============================================================

        total_dispatch_time = (
            time.perf_counter() - dispatch_start
        ) * 1_000_000


        # ============================================================
        # 8. TRACEABLE LOGGING
        # ============================================================

        print("\n" + "=" * 50)
        print(" CONDITIONAL ASYNCHRONOUS HOT PATH COMPLETE ")
        print("=" * 50)

        print(
            f"[*] Traceability ID : {req_id}"
        )


        # ---------------- L4 ----------------

        if l4_pred is not None:

            print(
                f"[*] L4 Prediction    : "
                f"{'Attack' if l4_pred > 0.5 else 'Benign'} "
                f"({l4_pred:.4f})"
            )

            print(
                f"[*] L4 Inference    : "
                f"{l4_time:.2f} µs"
            )

        else:

            print(
                "[*] L4 Prediction    : NOT EXECUTED"
            )


        # ---------------- L7 ----------------

        if l7_pred is not None:

            print(
                f"[*] L7 Prediction    : "
                f"{'Attack' if l7_pred > 0.5 else 'Benign'} "
                f"({l7_pred:.4f})"
            )

            print(
                f"[*] L7 Inference    : "
                f"{l7_time:.2f} µs"
            )

        else:

            print(
                "[*] L7 Prediction    : NOT EXECUTED"
            )


        print(
            f"[*] Total Pipeline  : "
            f"{total_dispatch_time:.2f} µs"
        )


        # ============================================================
        # 9. RETURN STRUCTURED RESULT
        # ============================================================

        return {
            "req_id": req_id,

            "l4_available": l4_available,
            "l7_available": l7_available,

            "l4_prediction": l4_pred,
            "l7_prediction": l7_pred,

            "l4_time_us": l4_time,
            "l7_time_us": l7_time,

            "total_time_us": total_dispatch_time
        }