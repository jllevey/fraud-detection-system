"""
Supervised Fraud Classifier Engine (XGBoost)
Trains an XGBoost model combining transaction, behavioral, and graph features.
Computes evaluation metrics (PR-AUC, Precision, Recall, F1, Confusion Matrix)
and calibrated ML fraud probabilities (0-100).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    precision_recall_curve, auc, confusion_matrix
)

class SupervisedFraudClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=15.0,  # Handle class imbalance
            random_state=random_state,
            eval_metric="logloss"
        )
        self.feature_cols = [
            "amount",
            "amount_to_baseline_ratio",
            "z_score_amount",
            "account_age_days",
            "is_night_txn",
            "is_new_device",
            "is_new_destination",
            "is_digital_voucher_merchant",
            "graph_degree",
            "community_size",
            "graph_ring_score"
        ]
        self.is_fitted = False
        self.metrics = {}
        self.feature_importances_ = {}
        
    def build_feature_matrix(self, df_transactions, df_accounts, graph_engine, anomaly_df=None):
        """
        Assembles all multi-modal features for each transaction.
        """
        acc_dict = df_accounts.set_index("account_id").to_dict(orient="index")
        
        records = []
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
            
            hour = pd.to_datetime(tx["timestamp"]).hour
            is_night = 1 if 0 <= hour < 6 else 0
            is_new_dev = 1 if tx["device_id"] != acc["primary_device_id"] else 0
            is_new_dest = 1 if tx["destination_id"] != acc["primary_destination_id"] else 0
            
            is_digital = 1 if tx["merchant_id"] in ["M91_CRYPTO_CARD", "M77_DIGITAL_GIFT"] else 0
            
            # Graph features
            g_feats = graph_engine.get_account_features(acc_id)
            
            rec = {
                "transaction_id": tx["transaction_id"],
                "account_id": acc_id,
                "amount": float(tx["amount"]),
                "amount_to_baseline_ratio": float(amt_ratio),
                "z_score_amount": float(z_score),
                "account_age_days": float(acc["account_age_days"]),
                "is_night_txn": int(is_night),
                "is_new_device": int(is_new_dev),
                "is_new_destination": int(is_new_dest),
                "is_digital_voucher_merchant": int(is_digital),
                "graph_degree": float(g_feats["graph_degree"]),
                "community_size": float(g_feats["community_size"]),
                "graph_ring_score": float(g_feats["graph_ring_score"]),
                "is_fraud": int(tx.get("is_fraud", 0))
            }
            records.append(rec)
            
        return pd.DataFrame(records)
        
    def train_and_evaluate(self, df_transactions, df_accounts, graph_engine):
        """
        Temporal-aware train/test split: uses chronological split to avoid leakage.
        """
        feat_df = self.build_feature_matrix(df_transactions, df_accounts, graph_engine)
        
        # Chronological split: 70% train, 30% test
        split_idx = int(len(feat_df) * 0.70)
        train_df = feat_df.iloc[:split_idx]
        test_df = feat_df.iloc[split_idx:]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df["is_fraud"]
        
        X_test = test_df[self.feature_cols]
        y_test = test_df["is_fraud"]
        
        # Fit XGBoost model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Predictions on test set
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.50).astype(int)
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(rec_curve, prec_curve)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
        
        self.metrics = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm,
            "test_sample_size": len(test_df),
            "test_fraud_count": int(y_test.sum())
        }
        
        # Feature importances
        imp = self.model.feature_importances_
        self.feature_importances_ = {
            col: round(float(val), 4) for col, val in zip(self.feature_cols, imp)
        }
        
        # Now score all transactions to return dataframe with ml_scores
        X_all = feat_df[self.feature_cols]
        all_proba = self.model.predict_proba(X_all)[:, 1]
        feat_df["ml_fraud_score"] = np.round(all_proba * 100.0, 1)
        
        return feat_df, self.metrics
        
    def predict_single(self, row_dict):
        """
        Scores a single feature dict.
        """
        row_df = pd.DataFrame([row_dict])[self.feature_cols]
        proba = self.model.predict_proba(row_df)[0, 1]
        return round(proba * 100.0, 1)
