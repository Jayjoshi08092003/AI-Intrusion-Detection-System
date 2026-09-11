import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class SOCReasoningAgent:
    def __init__(self):
        self.client = Groq()

    # CRITICAL: Added req_id and payload_hash here
    def generate_incident_report(self, l4_score, l7_score, l4_shap, l7_context, req_id, payload_hash):
        print("\n[🤖] LLM Agent transmitting forensic data to Groq LPU...")
        
        top_drivers = list(l4_shap.items())[:3]
        
        prompt = f"""
        You are an elite Tier-3 SOC Analyst responding to an automated proxy alert.
        
        System logged an anomalous network event:
        - Packet Traceability ID: {req_id}
        - Payload SHA-256 Hash: {payload_hash}
        - Layer 4 Network Classifier Confidence: {l4_score:.2f} (Attack Threshold: 0.5)
        - Layer 7 Payload Classifier Confidence: {l7_score:.2f} (Attack Threshold: 0.5)
        - Top Network Features Driving the Attack (TreeSHAP values): {top_drivers}
        - Suspicious Layer 7 Payload Context (Extracted Tokens): {l7_context['critical_tokens']}
        
        Generate a brutal, strict, 2-sentence incident summary and a recommended firewall action. Do not hallucinate. Do not add conversational filler.
        
        CRITICAL INSTRUCTION: You MUST start your response exactly with:
        "INCIDENT REPORT FOR ID: {req_id} [Hash: {payload_hash[:8]}]"
        """
        
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1, 
            max_completion_tokens=2048,
            top_p=1,
            stream=True,
            stop=None
        )
        
        full_report = ""
        
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            full_report += content
            
        return full_report.strip()