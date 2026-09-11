import uuid 
import hashlib 
import production_dispatcher 
import warnings

# Mute the scikit-learn feature name warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from xai_explainer import ExplainabilityModule 
from llm_agent import SOCReasoningAgent 

def run_master_system(traffic_event):
    """
    Accepts a traffic_event dictionary that may contain 'l4', 'l7', or both.
    """
    # 1. Initialize Hot Path (Microseconds)
    dispatcher = production_dispatcher.ProductionDispatcher()
    
    # 2. Dynamically extract and verify available features
    l4_packet = traffic_event.get("l4", {})
    l7_packet = traffic_event.get("l7", {})
    
    has_l4 = bool(l4_packet)
    has_l7 = bool(l7_packet)
    
    if not has_l4 and not has_l7:
        print("[-] Dropping empty event: No L4 or L7 features available.")
        return

    # ---------------------------------------------------------
    # CRYPTOGRAPHIC PROVENANCE GENERATION
    # ---------------------------------------------------------
    req_id = str(uuid.uuid4())
    
    # Only hash L7 if L7 data actually exists, otherwise flag it
    if has_l7 and "content" in l7_packet:
        payload_hash = hashlib.sha256(l7_packet["content"].encode('utf-8')).hexdigest()
    else:
        payload_hash = "NO_L7_PAYLOAD"
    
    print("\n" + "="*50)
    print(f" INCOMING PACKET DETECTED | ID: {req_id}")
    print(f" Features Present -> L4: {has_l4} | L7: {has_l7}")
    print("="*50)
    
    # Execute Hot Path - Pass None if a layer is missing so the dispatcher doesn't choke
    result = dispatcher.run_hot_path(
        l4_packet if has_l4 else None, 
        l7_packet if has_l7 else None, 
        req_id
    )
    
    # Check what features were actually returned dynamically
    l4_score = result.get("l4_prediction", None)
    l7_score = result.get("l7_prediction", None)
    total_time = result.get("total_time_us", 0.0)
    
    print(f"[*] Total Pipeline Time : {float(total_time):.2f} µs")
    
    # Evaluate threat dynamically depending on which layer fired and returned a score
    l4_threat = (l4_score is not None) and (l4_score > 0.5)
    l7_threat = (l7_score is not None) and (l7_score > 0.5)
    
    # 3. Dynamic Cold Path Trigger
    if l4_threat or l7_threat:
        print("\n[!] THREAT DETECTED BY NATIVE PROXY LAYER.")
        trigger = input("[?] Trigger Heavy XAI & LLM Forensic Analysis? (y/n): ")
        
        if trigger.lower() == 'y':
            print("\n[*] Initializing Cold Path...")
            xai = ExplainabilityModule()
            llm = SOCReasoningAgent()
            
            # Conditionally fire XAI only for models that flagged a threat AND have data
            l4_shap = None
            if l4_threat and has_l4:
                print("[*] Generating L4 SHAP Explanations...")
                l4_shap = xai.explain_l4_attack(l4_packet)
                
            l7_lime = None
            if l7_threat and has_l7:
                print("[*] Generating L7 LIME Explanations...")
                l7_lime = xai.explain_l7_tokens(l7_packet)
            
            # Generate Report 
            report = llm.generate_incident_report(
                l4_score if l4_score is not None else 0.0, 
                l7_score if l7_score is not None else 0.0, 
                l4_shap, 
                l7_lime, 
                req_id, 
                payload_hash
            )
            
            print("\n" + "="*50)
            print(" FINAL SOC INCIDENT REPORT ")
            print("="*50)
            print(report)
        else:
            print("[*] Forensics bypassed. Packet dropped. Returning to high-speed ingestion.")
    else:
        print("[*] Traffic Benign or No Target Layer Fired. Allowing packet through proxy.")

if __name__ == "__main__":
    # Mock Scenario 1: A volumetric network attack (pure L4, no L7 data)
    volumetric_event = {
        "l4": {
            "Destination Port": 80,
            "Flow Duration": 54020,
            "Total Length of Fwd Packets": 999999,
            "Flow Bytes/s": 8500000.0,
            "Flow Packets/s": 4500000.0,
            "ACK Flag Count": 850,
            "Fwd Packet Length Max": 1460.0,
            "Bwd Packet Length Max": 1460.0,
            "Flow IAT Mean": 1.2,
            "SYN Flag Count": 1,
            "Subflow Fwd Bytes": 999999
        },
        "l7": {} # Empty L7 dict simulates missing application payload
    }
    
    # Mock Scenario 2: An application layer attack (SQLi/XSS)
    app_layer_event = {
        "l4": {}, # Empty L4 dict
        "l7": {
            "Method": "POST",
            "URL": "/login.php?admin=' OR 1=1--",
            "User-Agent": "sqlmap/1.5",
            "content": "user=admin&pass=<script>alert(XSS)</script>"
        }
    }

    print("--- RUNNING VOLUMETRIC MOCK ---")
    run_master_system(volumetric_event)
    
    print("\n--- RUNNING APP LAYER MOCK ---")
    run_master_system(app_layer_event)