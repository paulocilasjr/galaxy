import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict

def build_classification_plots(stats_path: str) -> List[Dict[str, str]]:
    """
    Read Ludwig’s test_statistics.json and build four interactive Plotly panels:
      - Confusion Matrix
      - ROC Curve
      - Precision–Recall Curve
      - Classification Report Heatmap

    Returns a list of dicts, each with:
      {
        "title": <plot title>,
        "html":  <HTML fragment for embedding>
      }
    """

    # --- Load stats ---
    with open(stats_path, "r") as f:
        stats = json.load(f)
    label_stats = stats["label"]

    # common sizing
    cell = 40
    n_classes = len(label_stats["confusion_matrix"])
    side_px = max(cell * n_classes + 200, 600)
    common_cfg = {"displayModeBar": True, "scrollZoom": True}

    plots: List[Dict[str,str]] = []

    # 1) Confusion Matrix
    cm = np.array(label_stats["confusion_matrix"], dtype=int)
    labels = label_stats.get("labels", [str(i) for i in range(cm.shape[0])])
    total = cm.sum()
    fig_cm = go.Figure(
        go.Heatmap(z=cm, x=labels, y=labels,
                   colorscale="Blues", showscale=True,
                   colorbar=dict(title="Count"))
    )
    fig_cm.update_traces(xgap=2, ygap=2)
    fig_cm.update_layout(
        title=dict(text="Confusion Matrix", x=0.5),
        xaxis_title="Predicted", yaxis_title="Observed",
        yaxis_autorange="reversed",
        width=side_px, height=side_px,
        margin=dict(t=100,l=80,r=80,b=80)
    )
    mval = cm.max() if cm.size else 0
    thresh = mval/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i,j]
            pct = (v/total*100) if total>0 else 0
            col = "white" if v>thresh else "black"
            fig_cm.add_annotation(
                x=labels[j], y=labels[i],
                text=f"<b>{v}</b>",
                showarrow=False,
                font=dict(color=col,size=14),
                xanchor="center", yanchor="bottom", yshift=2
            )
            fig_cm.add_annotation(
                x=labels[j], y=labels[i],
                text=f"{pct:.1f}%",
                showarrow=False,
                font=dict(color=col,size=13),
                xanchor="center", yanchor="top", yshift=-2
            )
    plots.append({
        "title": "Confusion Matrix",
        "html": pio.to_html(fig_cm, full_html=False, include_plotlyjs="cdn", config=common_cfg)
    })

    # 2) ROC Curve
    roc = label_stats.get("roc_curve", {})
    if roc:
        fig_roc = go.Figure(go.Scatter(x=roc.get("fpr",[]), y=roc.get("tpr",[]), mode="lines"))
        auc = label_stats.get("roc_auc",0)
        fig_roc.update_layout(
            title=f"ROC Curve (AUC={auc:.2f})",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            width=side_px, height=side_px,
            margin=dict(t=80,l=80,r=80,b=80)
        )
        plots.append({
            "title": "ROC Curve",
            "html": pio.to_html(fig_roc, full_html=False, include_plotlyjs=False, config=common_cfg)
        })

    # 3) Precision–Recall Curve
    prc = label_stats.get("precision_recall_curve", {})
    if prc:
        fig_pr = go.Figure(go.Scatter(x=prc.get("recall",[]), y=prc.get("precision",[]), mode="lines"))
        ap = label_stats.get("average_precision",0)
        fig_pr.update_layout(
            title=f"Precision–Recall Curve (AP={ap:.2f})",
            xaxis_title="Recall", yaxis_title="Precision",
            width=side_px, height=side_px,
            margin=dict(t=80,l=80,r=80,b=80)
        )
        plots.append({
            "title": "Precision–Recall Curve",
            "html": pio.to_html(fig_pr, full_html=False, include_plotlyjs=False, config=common_cfg)
        })

    # 4) Classification Report Heatmap
    pcs = label_stats.get("per_class_stats", {})
    if pcs:
        classes = list(pcs.keys())
        metrics = ["precision","recall","f1_score","support"]
        z, txt = [], []
        for c in classes:
            row, trow = [], []
            for m in metrics:
                val = pcs[c].get(m,0)
                row.append(val)
                trow.append(f"{int(val)}" if m=="support" else f"{val:.2f}")
            z.append(row); txt.append(trow)
        fig_cr = go.Figure(
            go.Heatmap(z=z, x=metrics, y=[str(c) for c in classes],
                       text=txt, texttemplate="%{text}",
                       colorscale="Reds", showscale=True,
                       colorbar=dict(title="Value"))
        )
        fig_cr.update_layout(
            title="Classification Report",
            xaxis_title="", yaxis_title="Class",
            width=side_px, height=side_px,
            margin=dict(t=80,l=80,r=80,b=80)
        )
        plots.append({
            "title": "Classification Report",
            "html": pio.to_html(fig_cr, full_html=False, include_plotlyjs=False, config=common_cfg)
        })

    return plots
