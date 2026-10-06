"""
Integrated Pipeline Orchestrator
Coordinates data generation, graph intelligence, behavioral anomaly detection,
supervised classification, and risk score fusion.
Caches processed state for lightning-fast Streamlit interactivity.
"""

import pandas as pd
import numpy as np
from src.data_generator import generate_synthetic_fraud_dataset
from src.graph_engine import GraphIntelligenceEngine
from src.anomaly_engine import BehavioralAnomalyEngine
from src.supervised_model import SupervisedFraudClassifier
from src.risk_fusion import RiskFusionEngine

class FraudDetectionPipeline:
    def __init__(self):
        self.graph_engine = GraphIntelligenceEngine()
        self.anomaly_engine = BehavioralAnomalyEngine()
        self.supervised_model = SupervisedFraudClassifier()
        self.fusion_engine = RiskFusionEngine()
        
        self.data = None
        self.df_scored = None
        self.metrics = None
        self.is_initialized = False
        
    def run_full_pipeline(self, n_accounts=120, n_transactions=1600):
        print(" [1/5] Generating synthetic dataset with injected fraud rings...")
        self.data = generate_synthetic_fraud_dataset(
            n_normal_accounts=n_accounts, 
            n_normal_transactions=n_transactions
        )
        df_accounts = self.data["accounts"]
        df_txs = self.data["transactions"]
        
        print(" [2/5] Building heterogeneous network & running community detection...")
        self.graph_engine.build_graph(df_txs, df_accounts)
        
        print(" [3/5] Fitting Isolation Forest for behavioral anomaly detection...")
        self.anomaly_engine.fit(df_txs, df_accounts)
        anomaly_df = self.anomaly_engine.score_transactions(df_txs, df_accounts)
        
        print(" [4/5] Training XGBoost supervised fraud classifier (chronological split)...")
        ml_feat_df, self.metrics = self.supervised_model.train_and_evaluate(
            df_txs, df_accounts, self.graph_engine
        )
        
        print(" [5/5] Calibrating risk fusion and assigning decision engine tiers...")
        acc_dict = df_accounts.set_index("account_id").to_dict(orient="index")
        
        scored_records = []
        for idx, tx in df_txs.iterrows():
            tx_id = tx["transaction_id"]
            acc_id = tx["account_id"]
            acc = acc_dict.get(acc_id, {})
            
            # Scores from subsystems
            ml_score = float(ml_feat_df.loc[ml_feat_df["transaction_id"] == tx_id, "ml_fraud_score"].values[0])
            anomaly_score = float(anomaly_df.loc[anomaly_df["transaction_id"] == tx_id, "behavioral_anomaly_score"].values[0])
            g_feats = self.graph_engine.get_account_features(acc_id)
            graph_score = float(g_feats["graph_ring_score"])
            rule_score = float(self.fusion_engine.compute_rule_score(tx, acc))
            
            is_vip = (acc.get("account_type") == "VIP_HIGH_VALUE")
            amt_ratio = tx["amount"] / max(acc.get("baseline_avg_amount", 1000.0), 1.0)
            
            fusion_result = self.fusion_engine.fuse_risk(
                ml_score=ml_score,
                anomaly_score=anomaly_score,
                graph_score=graph_score,
                rule_score=rule_score,
                is_vip=is_vip,
                amt_ratio=amt_ratio
            )
            
            # Evidence
            b_ev = self.anomaly_engine.explain_transaction(tx, acc)
            g_ev = self.graph_engine.get_evidence(acc_id)
            narrative = self.fusion_engine.generate_narrative_explanation(
                tx, acc, fusion_result, g_ev, b_ev
            )
            
            scored_records.append({
                "transaction_id": tx_id,
                "account_id": acc_id,
                "account_name": acc.get("name", acc_id),
                "account_type": acc.get("account_type", "STANDARD"),
                "account_age_days": acc.get("account_age_days", 100),
                "timestamp": tx["timestamp"],
                "amount": tx["amount"],
                "baseline_avg": acc.get("baseline_avg_amount", 1000.0),
                "amount_ratio": round(amt_ratio, 2),
                "merchant_id": tx["merchant_id"],
                "merchant_name": tx["merchant_name"],
                "item_name": tx["item_name"],
                "device_id": tx["device_id"],
                "destination_id": tx["destination_id"],
                "payment_method": tx["payment_method"],
                "is_fraud_ground_truth": tx["is_fraud"],
                "fraud_type": tx["fraud_type"],
                "ring_id": tx["ring_id"],
                # Subsystem scores
                "ml_score": ml_score,
                "anomaly_score": anomaly_score,
                "graph_score": graph_score,
                "rule_score": rule_score,
                # Final fused risk & decision
                "final_risk_score": fusion_result["final_risk_score"],
                "action": fusion_result["action"],
                "tier": fusion_result["tier"],
                "color": fusion_result["color"],
                "badge": fusion_result["badge"],
                "narrative_title": narrative["title"],
                "narrative_summary": narrative["summary"],
                "narrative_reasons": narrative["reasons"]
            })
            
        self.df_scored = pd.DataFrame(scored_records)
        self.is_initialized = True
        print(f" Pipeline execution complete! Scored {len(self.df_scored)} transactions.")
        return self

if __name__ == "__main__":
    pipeline = FraudDetectionPipeline()
    pipeline.run_full_pipeline()
    print("Sample scored transactions:")
    print(pipeline.df_scored[["transaction_id", "account_id", "amount", "final_risk_score", "action", "tier"]].head(10))
