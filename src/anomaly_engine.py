"""
Behavioral Anomaly Detection Engine
Applies Isolation Forest to account behavioral deviation features
and extracts calibrated behavioral anomaly scores (0-100) and rationale.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class BehavioralAnomalyEngine:
    def __init__(self, contamination=0.03, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_estimators=100
        )
        self.feature_cols = [
            "amount_to_baseline_ratio",
            "z_score_amount",
            "is_night_txn",
            "account_age_days",
            "is_new_device",
            "is_new_destination"
        ]
        self.is_fitted = False
        
    def prepare_features(self, df_transactions, df_accounts):
        """
        Calculates deviation between the transaction and account's historical baseline.
        """
        acc_dict = df_accounts.set_index("account_id").to_dict(orient="index")
        
        feats = []
        for _, tx in df_transactions.iterrows():
            acc_id = tx["account_id"]
            acc = acc_dict.get(acc_id, {
                "baseline_avg_amount": 1000.0,
                "baseline_std_amount": 200.0,
                "account_age_days": 180,
                "primary_device_id": tx["device_id"],
                "primary_destination_id": tx["destination_id"]
            })
            
            baseline_avg = acc["baseline_avg_amount"]
            baseline_std = max(acc["baseline_std_amount"], 1.0)
            
            amt_ratio = tx["amount"] / max(baseline_avg, 1.0)
            z_score = abs(tx["amount"] - baseline_avg) / baseline_std
            
            # Night transaction (00:00 - 05:59)
            tx_hour = pd.to_datetime(tx["timestamp"]).hour
            is_night = 1 if 0 <= tx_hour < 6 else 0
            
            is_new_device = 1 if tx["device_id"] != acc["primary_device_id"] else 0
            is_new_dest = 1 if tx["destination_id"] != acc["primary_destination_id"] else 0
            
            feats.append({
                "transaction_id": tx["transaction_id"],
                "amount_to_baseline_ratio": amt_ratio,
                "z_score_amount": z_score,
                "is_night_txn": is_night,
                "account_age_days": acc["account_age_days"],
                "is_new_device": is_new_device,
                "is_new_destination": is_new_dest
            })
            
        return pd.DataFrame(feats)
        
    def fit(self, df_transactions, df_accounts):
        features_df = self.prepare_features(df_transactions, df_accounts)
        X = features_df[self.feature_cols].values
        self.model.fit(X)
        self.is_fitted = True
        return self
        
    def score_transactions(self, df_transactions, df_accounts):
        """
        Outputs calibrated anomaly scores (0 to 100) and explanations.
        """
        features_df = self.prepare_features(df_transactions, df_accounts)
        X = features_df[self.feature_cols].values
        
        # Raw decision function: higher = more normal, lower/negative = anomalous
        decision_raw = self.model.decision_function(X)
        
        # Calibrate to 0-100 scale where 100 is most anomalous
        # Typical range is [-0.3, 0.2]
        calibrated_scores = []
        for raw, (_, row) in zip(decision_raw, features_df.iterrows()):
            # Base inversion: normal (~0.15) -> low risk, anomalous (< -0.1) -> high risk
            norm_val = 1.0 / (1.0 + np.exp(raw * 18.0)) # Sigmoid transformation
            score = norm_val * 100.0
            
            # Additional heuristic adjustments for extreme deviations
            if row["amount_to_baseline_ratio"] > 8.0:
                score = max(score, 88.0)
            if row["is_night_txn"] == 1 and row["amount_to_baseline_ratio"] > 4.0:
                score = max(score, 92.0)
                
            calibrated_scores.append(round(min(max(score, 1.0), 99.0), 1))
            
        features_df["behavioral_anomaly_score"] = calibrated_scores
        return features_df
        
    def explain_transaction(self, tx_row, acc_row):
        """
        Produces natural language evidence regarding behavioral deviation.
        """
        baseline_avg = acc_row["baseline_avg_amount"]
        amount = tx_row["amount"]
        ratio = amount / max(baseline_avg, 1.0)
        hour = pd.to_datetime(tx_row["timestamp"]).hour
        
        bullets = []
        if ratio > 3.0:
            bullets.append(f"Transaction amount ₹{amount:,.2f} is {ratio:.1f}× higher than account baseline average (₹{baseline_avg:,.2f}).")
        elif ratio < 0.2 and amount > 50:
            bullets.append(f"Transaction amount is significantly below typical spending range ({ratio:.2f}×).")
        else:
            bullets.append(f"Transaction amount ₹{amount:,.2f} is well within normal baseline limits ({ratio:.2f}× of ₹{baseline_avg:,.2f}).")
            
        if 0 <= hour < 6:
            bullets.append(f"Executed during abnormal late-night window ({hour:02d}:00 hours).")
            
        if tx_row["device_id"] != acc_row["primary_device_id"]:
            bullets.append(f"Unfamiliar hardware signature: transaction originated from new device {tx_row['device_id']}.")
        else:
            bullets.append("Originated from registered primary device.")
            
        if acc_row["account_age_days"] < 15:
            bullets.append(f"High-risk account age profile: Account created only {acc_row['account_age_days']} days ago.")
            
        return bullets
