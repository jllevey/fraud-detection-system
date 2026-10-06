"""
Heterogeneous Graph Intelligence & Fraud-Ring Detection Engine
Constructs entity graphs, computes account similarity projections,
runs community detection, and calculates calibrated graph risk scores and evidence.
"""

import networkx as nx
import pandas as pd
import numpy as np
from datetime import timedelta
from networkx.algorithms.community import greedy_modularity_communities

class GraphIntelligenceEngine:
    def __init__(self):
        self.bipartite_graph = nx.Graph()
        self.account_graph = nx.Graph()
        self.community_map = {}
        self.account_graph_scores = {}
        self.account_graph_evidence = {}
        
    def build_graph(self, df_transactions, df_accounts):
        """
        Builds the heterogeneous network and account projection network.
        """
        self.bipartite_graph = nx.Graph()
        self.account_graph = nx.Graph()
        
        # 1. Add all accounts as nodes
        for _, acc in df_accounts.iterrows():
            acc_id = acc["account_id"]
            self.account_graph.add_node(
                acc_id, 
                node_type="account",
                name=acc["name"],
                is_ring_member=acc.get("is_ring_member", 0),
                account_type=acc.get("account_type", "STANDARD")
            )
            self.bipartite_graph.add_node(
                acc_id,
                node_type="account",
                label=acc_id,
                name=acc["name"]
            )
            
        # 2. Add heterogeneous relationships from transactions
        for _, tx in df_transactions.iterrows():
            acc_id = tx["account_id"]
            dev_id = tx["device_id"]
            dest_id = tx["destination_id"]
            merch_id = tx["merchant_id"]
            
            # Nodes
            self.bipartite_graph.add_node(dev_id, node_type="device", label=dev_id)
            self.bipartite_graph.add_node(dest_id, node_type="destination", label=dest_id)
            self.bipartite_graph.add_node(merch_id, node_type="merchant", label=merch_id)
            
            # Edges
            self.bipartite_graph.add_edge(acc_id, dev_id, relation="USES_DEVICE")
            self.bipartite_graph.add_edge(acc_id, dest_id, relation="SENDS_TO_DESTINATION")
            self.bipartite_graph.add_edge(acc_id, merch_id, relation="TRANSACTS_AT")
            
        # 3. Construct Account-Account Projection Graph
        # Group accounts by shared devices and shared destinations
        device_to_accounts = {}
        dest_to_accounts = {}
        
        for _, tx in df_transactions.iterrows():
            device_to_accounts.setdefault(tx["device_id"], set()).add(tx["account_id"])
            dest_to_accounts.setdefault(tx["destination_id"], set()).add(tx["account_id"])
            
        # Add edges for shared devices
        for dev, accs in device_to_accounts.items():
            acc_list = list(accs)
            if len(acc_list) > 1:
                for i in range(len(acc_list)):
                    for j in range(i + 1, len(acc_list)):
                        u, v = acc_list[i], acc_list[j]
                        if self.account_graph.has_edge(u, v):
                            self.account_graph[u][v]["weight"] += 3.5
                            self.account_graph[u][v]["shared_devices"].append(dev)
                        else:
                            self.account_graph.add_edge(
                                u, v, 
                                weight=3.5, 
                                shared_devices=[dev], 
                                shared_destinations=[],
                                shared_merchants=[]
                            )

        # Add edges for shared destinations
        for dest, accs in dest_to_accounts.items():
            acc_list = list(accs)
            if len(acc_list) > 1:
                for i in range(len(acc_list)):
                    for j in range(i + 1, len(acc_list)):
                        u, v = acc_list[i], acc_list[j]
                        if self.account_graph.has_edge(u, v):
                            self.account_graph[u][v]["weight"] += 3.0
                            self.account_graph[u][v]["shared_destinations"].append(dest)
                        else:
                            self.account_graph.add_edge(
                                u, v, 
                                weight=3.0, 
                                shared_devices=[], 
                                shared_destinations=[dest],
                                shared_merchants=[]
                            )
                            
        # Detect temporal co-transacting at same high-risk merchant (<30 mins)
        tx_sorted = df_transactions.sort_values("timestamp")
        digital_txs = tx_sorted[tx_sorted["merchant_id"].isin(["M91_CRYPTO_CARD", "M77_DIGITAL_GIFT"])]
        
        dt_list = digital_txs.to_dict(orient="records")
        for i in range(len(dt_list)):
            for j in range(i + 1, min(i + 20, len(dt_list))):
                tx1 = dt_list[i]
                tx2 = dt_list[j]
                if tx1["account_id"] == tx2["account_id"]:
                    continue
                time_diff = abs((tx2["timestamp"] - tx1["timestamp"]).total_seconds())
                if time_diff <= 1800: # within 30 minutes
                    u, v = tx1["account_id"], tx2["account_id"]
                    if self.account_graph.has_edge(u, v):
                        self.account_graph[u][v]["weight"] += 2.0
                        self.account_graph[u][v]["shared_merchants"].append(tx1["merchant_id"])
                    else:
                        self.account_graph.add_edge(
                            u, v,
                            weight=2.0,
                            shared_devices=[],
                            shared_destinations=[],
                            shared_merchants=[tx1["merchant_id"]]
                        )
                        
        self._compute_communities_and_risk(df_transactions)
        return self
        
    def _compute_communities_and_risk(self, df_transactions):
        """
        Runs community detection on account graph and computes ring risk scores.
        """
        # Run greedy modularity community detection on connected components
        subgraphs = [self.account_graph.subgraph(c).copy() for c in nx.connected_components(self.account_graph) if len(c) > 1]
        
        community_id = 1
        self.community_map = {}
        self.community_stats = {}
        
        for subg in subgraphs:
            comms = list(greedy_modularity_communities(subg))
            for comm in comms:
                c_nodes = list(comm)
                if len(c_nodes) >= 3: # Ring candidate threshold
                    c_sub = self.account_graph.subgraph(c_nodes)
                    density = nx.density(c_sub)
                    
                    # Gather shared destinations and devices
                    c_txs = df_transactions[df_transactions["account_id"].isin(c_nodes)]
                    unique_dests = c_txs["destination_id"].unique()
                    unique_devices = c_txs["device_id"].unique()
                    
                    # Mule concentration ratio
                    mule_ratio = 1.0 if ("P99_MULE_HUB" in unique_dests or "P88_MULE_HUB" in unique_dests) else 0.1
                    
                    self.community_stats[community_id] = {
                        "community_id": community_id,
                        "members": c_nodes,
                        "size": len(c_nodes),
                        "density": round(density, 3),
                        "unique_destinations": list(unique_dests),
                        "unique_devices": list(unique_devices),
                        "has_known_mule": "P99_MULE_HUB" in unique_dests or "P88_MULE_HUB" in unique_dests
                    }
                    
                    for node in c_nodes:
                        self.community_map[node] = community_id
                        
                    community_id += 1

        # Calculate Account-Level Graph Risk Score (0 to 100) and human-readable evidence
        for node in self.account_graph.nodes():
            degree = self.account_graph.degree(node)
            comm_id = self.community_map.get(node, None)
            
            evidence = []
            score = 0.0
            
            if comm_id is not None:
                c_stat = self.community_stats[comm_id]
                size = c_stat["size"]
                density = c_stat["density"]
                has_mule = c_stat["has_known_mule"]
                
                # Base ring score for community membership
                base = 50.0 + min(size * 4.0, 25.0) + (density * 15.0)
                if has_mule:
                    base += 15.0
                    
                score = min(base, 99.0)
                
                # Extract specific evidence bullets
                neighbors = list(self.account_graph.neighbors(node))
                shared_dests = set()
                shared_devs = set()
                for nbr in neighbors:
                    edge_data = self.account_graph.get_edge_data(node, nbr, {})
                    shared_dests.update(edge_data.get("shared_destinations", []))
                    shared_devs.update(edge_data.get("shared_devices", []))
                    
                if "P99_MULE_HUB" in shared_dests or "P88_MULE_HUB" in shared_dests:
                    mule_hub = "P99_MULE_HUB" if "P99_MULE_HUB" in shared_dests else "P88_MULE_HUB"
                    evidence.append(f"Shares flagged cashout destination ({mule_hub}) with {len(neighbors)} correlated accounts.")
                    
                if len(shared_devs) > 0:
                    evidence.append(f"Shares hardware fingerprint / emulator ({list(shared_devs)[0]}) with {len(neighbors)} accounts.")
                    
                evidence.append(f"Member of high-density Fraud Syndicate Ring #{comm_id} ({size} accounts, network density: {density:.2f}).")
                evidence.append(f"Graph degree: connected to {degree} accounts within suspicious co-transacting window.")
            else:
                if degree > 0:
                    score = min(degree * 8.0, 25.0)
                    evidence.append(f"Loosely connected to {degree} other accounts via shared attributes.")
                else:
                    score = 2.0
                    evidence.append("Isolated entity node: zero graph connections to known clusters or shared devices.")
                    
            self.account_graph_scores[node] = round(score, 1)
            self.account_graph_evidence[node] = evidence
            
    def get_account_features(self, account_id):
        """
        Returns graph features for an account to feed tabular models.
        """
        degree = self.account_graph.degree(account_id) if self.account_graph.has_node(account_id) else 0
        comm_id = self.community_map.get(account_id, 0)
        comm_size = self.community_stats[comm_id]["size"] if comm_id in self.community_stats else 1
        ring_score = self.account_graph_scores.get(account_id, 2.0)
        
        return {
            "graph_degree": degree,
            "community_id": comm_id,
            "community_size": comm_size,
            "graph_ring_score": ring_score
        }
        
    def get_evidence(self, account_id):
        return self.account_graph_evidence.get(account_id, ["No suspicious graph associations detected."])
