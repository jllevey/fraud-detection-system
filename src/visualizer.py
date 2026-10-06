"""
Interactive Visualization Helper using Plotly & NetworkX
Generates heterogeneous entity network graphs, risk gauges, and radar charts.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np

def create_network_figure(graph_engine, highlight_community=None):
    """
    Builds an interactive Plotly graph showing accounts, devices, destinations, and merchants.
    """
    G = graph_engine.bipartite_graph
    
    # Generate spring layout positions for all nodes
    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=50)
    
    # 1. Edge Traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.0, color="#334155"),
        hoverinfo="none",
        mode="lines"
    )
    
    # 2. Node Traces by Type
    node_types = {
        "account": {"color": "#3b82f6", "symbol": "circle", "size": 12, "name": "Accounts"},
        "device": {"color": "#06b6d4", "symbol": "square", "size": 11, "name": "Devices"},
        "destination": {"color": "#ec4899", "symbol": "diamond", "size": 13, "name": "Destinations (Mules/VPAs)"},
        "merchant": {"color": "#f59e0b", "symbol": "triangle-up", "size": 12, "name": "Merchants"}
    }
    
    traces = [edge_trace]
    
    for n_type, style in node_types.items():
        xs, ys, texts, colors, sizes, hovertexts = [], [], [], [], [], []
        
        for node, data in G.nodes(data=True):
            if data.get("node_type") == n_type:
                x, y = pos[node]
                xs.append(x)
                ys.append(y)
                texts.append(node)
                
                # Dynamic coloring for accounts based on risk
                if n_type == "account":
                    risk = graph_engine.account_graph_scores.get(node, 0.0)
                    comm = graph_engine.community_map.get(node, 0)
                    
                    if risk > 70.0:
                        c = "#ef4444" # Red for high risk
                        s = 15
                    elif risk > 30.0:
                        c = "#f97316" # Orange
                        s = 13
                    else:
                        c = "#10b981" # Green for normal
                        s = 10
                        
                    if highlight_community and comm == highlight_community:
                        c = "#e11d48" # Highlighted ring
                        s = 18
                        
                    colors.append(c)
                    sizes.append(s)
                    hovertexts.append(f"Account: <b>{node}</b><br>Graph Risk: <b>{risk:.1f}/100</b><br>Community: #{comm}")
                else:
                    colors.append(style["color"])
                    sizes.append(style["size"])
                    hovertexts.append(f"{n_type.capitalize()}: <b>{node}</b>")
                    
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if n_type in ["destination", "device"] and len(xs) < 25 else "markers",
            text=[t if len(xs) < 25 else "" for t in texts],
            textposition="top center",
            hovertext=hovertexts,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color=colors,
                symbol=style["symbol"],
                line=dict(width=1.5, color="#0f172a")
            ),
            name=style["name"]
        ))
        
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text="<b>Heterogeneous Fraud Network Topology</b>",
                font=dict(size=18, color="#f8fafc")
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0b0f19",
            paper_bgcolor="#0b0f19",
            legend=dict(
                font=dict(color="#94a3b8"),
                bgcolor="rgba(15, 23, 42, 0.7)",
                bordercolor="#334155",
                borderwidth=1
            ),
            height=580
        )
    )
    return fig

def create_risk_gauge(score, action_name, action_color):
    """
    Renders an eye-catching half-donut gauge chart for the 0-100 risk score.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>RISK SCORE: {score:.1f}</b><br><span style='color:{action_color};font-size:16px;'>{action_name}</span>", 'font': {'color': '#f8fafc', 'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': "#94a3b8"}},
            'bar': {'color': action_color, 'thickness': 0.3},
            'bgcolor': "#1e293b",
            'borderwidth': 1,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.25)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [70, 85], 'color': 'rgba(249, 115, 22, 0.25)'},
                {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font={'color': "#f8fafc"},
        height=260,
        margin=dict(t=50, b=10, l=25, r=25)
    )
    return fig

def create_radar_breakdown(ml_score, anomaly_score, graph_score, rule_score):
    """
    Radar chart showing the 4 pipeline signals.
    """
    categories = ['Supervised ML', 'Behavioral Anomaly', 'Graph / Ring Intelligence', 'Rule Heuristics']
    values = [ml_score, anomaly_score, graph_score, rule_score]
    # close the loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.35)',
        line=dict(color='#3b82f6', width=2),
        name='Signal Strength'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#64748b", size=10),
                gridcolor="#334155"
            ),
            angularaxis=dict(
                tickfont=dict(color="#cbd5e1", size=11),
                gridcolor="#334155"
            ),
            bgcolor="#131b2e"
        ),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        font=dict(color="#f8fafc"),
        showlegend=False,
        height=260,
        margin=dict(t=30, b=30, l=45, r=45)
    )
    return fig
