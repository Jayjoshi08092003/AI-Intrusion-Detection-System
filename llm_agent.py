import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = "openai/gpt-oss-120b"


class SOCReasoningAgent:
    """
    Evidence-grounded LLM Analyst + Critic (paper Sec. 5.11, Eq. 29-31).

    generate_analyst_report()  -> R_LLM    = G_theta(E)          (Eq. 29)
    generate_critic_review()   -> R_critic = C_psi(R_LLM, E)     (Eq. 30)
    generate_incident_report() -> R_final  = Psi(R_LLM, R_critic, E)  (Eq. 31)

    This replaces the previous single-prompt implementation: the analyst
    drafts a report from evidence alone, and a SEPARATE critic call
    cross-checks that draft against the same structured evidence and can
    reject/rewrite claims that aren't actually supported by it (this is
    the mechanism the paper's Section 6.4 example describes).
    """

    def __init__(self):
        self.client = Groq()

    # ------------------------------------------------------------------
    # Shared evidence structure E (Eq. 28)
    # ------------------------------------------------------------------
    def _build_evidence_block(self, l4_score, l7_score, l4_shap, l7_context, req_id, payload_hash):
        top_l4_drivers = list(l4_shap.items())[:3] if l4_shap else []
        l7_tokens = (l7_context or {}).get("critical_tokens", [])

        return {
            "request_id": req_id,
            "payload_sha256": payload_hash,
            "l4_confidence": round(float(l4_score), 4),
            "l7_confidence": round(float(l7_score), 4),
            "attack_threshold": 0.5,
            "top_l4_shap_features": [
                {"feature": f, "shap_value": round(float(v), 4)} for f, v in top_l4_drivers
            ],
            "l7_critical_tokens": l7_tokens,
        }

    def _call_groq(self, prompt, temperature=0.2, max_tokens=1024, json_mode=False):
        kwargs = dict(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        if json_mode:
            # Groq's JSON mode (OpenAI-compatible): forces the response body
            # itself to be valid JSON. Still requires the prompt to describe
            # the desired keys (done in generate_critic_review), but removes
            # the class of failures where the model wraps JSON in prose or
            # markdown fences.
            kwargs["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content.strip()

    @staticmethod
    def _extract_json_object(raw: str):
        """Best-effort JSON extraction: strips markdown fences, then falls
        back to grabbing the outermost {...} span, since some models still
        add a sentence before/after the object even in JSON mode."""
        text = raw.strip()

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    # ------------------------------------------------------------------
    # Analyst: R_LLM = G_theta(E)
    # ------------------------------------------------------------------
    def generate_analyst_report(self, evidence: dict) -> str:
        prompt = f"""
        You are a Tier-3 SOC analyst. You may use ONLY the structured evidence
        below. Do not assume any fact, actor, or campaign that is not present
        in it.

        Evidence (JSON):
        {json.dumps(evidence, indent=2)}

        Write a brutal, strict 2-3 sentence incident summary and a recommended
        firewall action. Ground every claim strictly in the evidence fields
        above. Do not add conversational filler.

        Start your response exactly with:
        "INCIDENT REPORT FOR ID: {evidence['request_id']} [Hash: {evidence['payload_sha256'][:8]}]"
        """
        return self._call_groq(prompt, temperature=0.3)

    # ------------------------------------------------------------------
    # Critic: R_critic = C_psi(R_LLM, E)
    # ------------------------------------------------------------------
    def generate_critic_review(self, analyst_report: str, evidence: dict) -> dict:
        prompt = f"""
        You are a skeptical SOC Critic. Compare the DRAFT REPORT to the
        EVIDENCE. Flag any claim in the draft that is NOT directly supported
        by the evidence — e.g. named threat actors, campaign attribution,
        malware families, or severity language stronger than the confidence
        scores/features/tokens actually justify.

        EVIDENCE (JSON):
        {json.dumps(evidence, indent=2)}

        DRAFT REPORT:
        {analyst_report}

        Respond with ONLY a JSON object, no other text, in this exact shape:
        {{
          "verdict": "APPROVED" or "REVISE",
          "unsupported_claims": ["..."],
          "corrected_summary": "a corrected 2-3 sentence summary using only grounded evidence, or null if APPROVED"
        }}
        """
        raw = self._call_groq(prompt, temperature=0.0, json_mode=True)
        parsed = self._extract_json_object(raw)

        if parsed is None:
            print(f"[Critic] WARNING: could not parse critic JSON. Raw output:\n{raw}\n")
            return {
                "verdict": "REVIEW_FAILED",
                "unsupported_claims": [],
                "corrected_summary": None,
                "raw_critic_output": raw,
            }

        # Normalize: tolerate missing keys from an otherwise-valid object.
        parsed.setdefault("verdict", "REVIEW_FAILED")
        parsed.setdefault("unsupported_claims", [])
        parsed.setdefault("corrected_summary", None)
        return parsed

    # ------------------------------------------------------------------
    # Orchestration: R_final = Psi(R_LLM, R_critic, E)
    # ------------------------------------------------------------------
    def generate_incident_report(self, l4_score, l7_score, l4_shap, l7_context, req_id, payload_hash) -> str:
        evidence = self._build_evidence_block(l4_score, l7_score, l4_shap, l7_context, req_id, payload_hash)

        print("\n[Analyst] Drafting incident report from structured evidence...")
        analyst_report = self.generate_analyst_report(evidence)

        print("[Critic] Validating draft against SHAP/token evidence...")
        critic_result = self.generate_critic_review(analyst_report, evidence)
        verdict = critic_result.get("verdict", "REVIEW_FAILED")

        if verdict == "APPROVED":
            final_summary = analyst_report
            critic_note = "Critic verdict: APPROVED — all claims grounded in evidence."

        elif verdict == "REVISE" and critic_result.get("corrected_summary"):
            final_summary = (
                f"INCIDENT REPORT FOR ID: {req_id} [Hash: {payload_hash[:8]}]\n"
                f"{critic_result['corrected_summary']}"
            )
            critic_note = (
                "Critic verdict: REVISE — analyst draft contained unsupported claims "
                f"{critic_result.get('unsupported_claims')}; rewritten to match evidence only."
            )

        else:
            final_summary = analyst_report
            critic_note = (
                "Critic verdict: REVIEW_FAILED — critic response could not be parsed; "
                "returning unreviewed analyst draft. Treat with reduced confidence."
            )

        return f"{final_summary}\n\n--- Critic Validation ---\n{critic_note}"
