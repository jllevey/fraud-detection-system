"""
Calibrated Multi-Signal Risk Fusion & Decision Engine
Combines Supervised ML, Behavioral Anomaly, Graph/Ring Intelligence,
and Rule Heuristics into a calibrated 0-100 score, decision tier,
and rich explainability rationale.
"""

import pandas as pd
import numpy as np

class RiskFusionEngine:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                "ml": 0.40,
                "anomaly": 0.25,
                "graph": 0.25,
                "rule": 0.10
            }
        else:
            self.weights = weights
            
    def compute_rule_score(self, tx_row, acc_row):
        """
        Calculates rule-based heuristic penalty (0 to 100).
        """
        score = 0.0
        
        # Rule 1: High-risk voucher merchants
        if tx_row["merchant_id"] in ["M91_CRYPTO_CARD", "M77_DIGITAL_GIFT"]:
            score += 35.0
            
        # Rule 2: Brand new account (<10 days old)
        if acc_row.get("account_age_days", 100) < 10:
            score += 30.0
            
        # Rule 3: Known mule destination
        if tx_row["destination_id"] in ["P99_MULE_HUB", "P88_MULE_HUB"]:
            score += 30.0
            
        # Rule 4: Device mismatch
        if tx_row["device_id"] != acc_row.get("primary_device_id", ""):
            score += 15.0
            
        return min(score, 100.0)
        
    def fuse_risk(self, ml_score, anomaly_score, graph_score, rule_score, is_vip=False, amt_ratio=1.0):
        """
        Calibrated fusion logic with protection for legitimate VIP transactions.
        """
        # Baseline weighted sum
        raw_risk = (
            self.weights["ml"] * ml_score +
            self.weights["anomaly"] * anomaly_score +
            self.weights["graph"] * graph_score +
            self.weights["rule"] * rule_score
        )
        
        # VIP Protection: Legitimate expensive transactions shouldn't be penalized
        if is_vip and graph_score <= 5.0 and amt_ratio <= 2.2:
            raw_risk = min(raw_risk * 0.25, 18.0)
            
        # Reinforcement: If member of confirmed coordinated ring, boost to critical
        if graph_score >= 80.0 and (ml_score >= 50.0 or rule_score >= 50.0):
            raw_risk = max(raw_risk, 88.0)
            
        final_score = round(min(max(raw_risk, 1.0), 99.0), 1)
        
        # Decision Tiers
        if final_score < 30.0:
            action = "ALLOW"
            tier = "Low"
            color = "#10b981" # Green
            badge = "🟢 ALLOW"
        elif final_score < 70.0:
            action = "MONITOR"
            tier = "Medium"
            color = "#f59e0b" # Amber
            badge = "🟡 MONITOR"
        elif final_score < 85.0:
            action = "STEP-UP VERIFICATION"
            tier = "High"
            color = "#f97316" # Orange
            badge = "🟠 STEP-UP VERIFICATION"
        elif final_score < 94.0:
            action = "MANUAL REVIEW"
            tier = "Critical"
            color = "#ef4444" # Red
            badge = "🔴 MANUAL REVIEW"
        else:
            action = "TEMPORARY BLOCK + INVESTIGATION"
            tier = "Severe Critical"
            color = "#7f1d1d" # Dark Red
            badge = "⛔ TEMPORARY BLOCK"
            
        return {
            "final_risk_score": final_score,
            "action": action,
            "tier": tier,
            "color": color,
            "badge": badge,
            "components": {
                "ml_score": ml_score,
                "anomaly_score": anomaly_score,
                "graph_score": graph_score,
                "rule_score": rule_score
            }
        }
        
    def generate_narrative_explanation(self, tx_row, acc_row, risk_result, graph_evidence, behavioral_evidence):
        """
        Generates full human-readable explanation with bulleted evidence.
        """
        score = risk_result["final_risk_score"]
        action = risk_result["action"]
        comps = risk_result["components"]
        
        title = f"Transaction {tx_row['transaction_id']} Risk Evaluation: {score}/100 ({risk_result['tier']})"
        
        reasons = []
        
        # 1. Behavioral points
        reasons.extend(behavioral_evidence)
        
        # 2. Graph evidence
        reasons.extend(graph_evidence)
        
        # 3. Action rationale
        if action == "ALLOW":
            summary = (
                f"Transaction approved. Customer activity matches verified historical baselines "
                f"with negligible network risk ({comps['graph_score']:.1f}/100) and zero entity ring associations."
            )
        elif action in ["MANUAL REVIEW", "TEMPORARY BLOCK + INVESTIGATION"]:
            summary = (
                f"High severity alert. Transaction displays severe coordinated ring signals "
                f"(Graph Risk: {comps['graph_score']:.1f}/100, ML Risk: {comps['ml_score']:.1f}/100). "
                f"Funneled toward suspicious cashout endpoints alongside correlated syndicates."
            )
        elif action == "STEP-UP VERIFICATION":
            summary = (
                f"Elevated risk flagged ({score}/100). Discrepancy observed between transaction context "
                f"and historical account patterns, but customer tenure warrants step-up 2FA rather than immediate denial."
            )
        else:
            summary = (
                f"Moderate risk observed ({score}/100). Transaction routed to continuous monitoring "
                f"for subsequent co-occurring patterns."
            )
            
        return {
            "title": title,
            "score": score,
            "action": action,
            "summary": summary,
            "reasons": reasons
        }
