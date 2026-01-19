"""HTML report generator using Jinja2 templates."""

import json
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


def generate_html_report(results: dict, output_path: str, checkpoint_path: str = ""):
    """Generate a self-contained HTML report with interactive charts."""
    context = _prepare_context(results, checkpoint_path)
    env = _create_jinja_env()
    template = env.get_template("report.html.jinja")
    html = template.render(context)
    Path(output_path).write_text(html, encoding="utf-8")


def _create_jinja_env():
    """Create Jinja2 environment with custom filters."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Register threshold functions as filters
    env.filters["f1_class"] = _f1_class
    env.filters["chatter_class"] = _chatter_class
    env.filters["ece_class"] = _ece_class
    env.filters["stability_class"] = _stability_class
    env.filters["boundary_diff_class"] = _boundary_diff_class
    env.filters["value_class"] = _value_class
    # Register formatting helpers
    env.filters["format_support"] = lambda s: f"Support: {s}"
    env.filters["marker_size"] = lambda c: max(5, min(20, c / 1000))
    env.filters["format_count"] = lambda c: f"n={c:,}"
    return env


def _prepare_context(results: dict, checkpoint_path: str) -> dict:
    """Prepare template context with processed data."""
    # Extract sections
    keys = results.get("keys", {})
    intent = results.get("intent", {})
    value = results.get("value", {})
    inv_dyn = results.get("inv_dyn")
    temporal = results.get("temporal")
    calibration = results.get("calibration")
    conditional = results.get("conditional")
    intents = intent.get("intents", [])

    # Prepare chart data (Python handles sorting/slicing)
    per_key = keys.get("per_key", {})
    per_key_sorted = sorted(per_key.items(), key=lambda x: x[1]["support"], reverse=True)
    top_keys = per_key_sorted[:20]

    keys_chart = {
        "names": [k for k, _ in top_keys],
        "f1_scores": [v["f1"] for _, v in top_keys],
        "supports": [v["support"] for _, v in top_keys],
    }

    # Confusion matrix data
    cm_norm = intent.get("confusion_matrix_normalized", [[]])
    cm_text = [[f"{v:.2f}" for v in row] for row in cm_norm]

    intent_chart = {
        "confusion_matrix": cm_norm,
        "confusion_matrix_text": cm_text,
        "intents": intents,
    }

    # Calibration chart data
    calibration_chart = None
    if calibration and calibration.get("bins"):
        bins = calibration["bins"]
        calibration_chart = {
            "bin_centers": [b["bin_center"] for b in bins],
            "mean_preds": [b["mean_pred"] for b in bins],
            "actual_freqs": [b["actual_freq"] for b in bins],
            "counts": [b["count"] for b in bins],
        }

    # Prepare tables (sorted/sliced in Python)
    per_key_table = per_key_sorted[:30]

    top_chatterers = []
    if temporal and "key_chatter_per_key" in temporal:
        chatter_per_key = temporal["key_chatter_per_key"]
        top_chatterers = sorted(chatter_per_key.items(), key=lambda x: x[1], reverse=True)[:10]

    ece_table = []
    if calibration and "per_key_ece" in calibration:
        per_key_ece = calibration["per_key_ece"]
        ece_table = sorted(per_key_ece.items(), key=lambda x: x[1], reverse=True)[:10]

    # Build nav sections
    nav_sections = [
        ("summary", "Summary"),
        ("keys", "Keys Head"),
        ("intent", "Intent Head"),
        ("value", "Value Head"),
    ]
    if inv_dyn:
        nav_sections.append(("invdyn", "Inverse Dynamics"))
    if temporal:
        nav_sections.append(("temporal", "Temporal Coherence"))
    if calibration:
        nav_sections.append(("calibration", "Calibration"))
    if conditional:
        nav_sections.append(("conditional", "Conditional Perf"))

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_name": Path(checkpoint_path).name if checkpoint_path else "",
        "nav_sections": nav_sections,
        "keys": keys,
        "intent": intent,
        "value": value,
        "inv_dyn": inv_dyn,
        "temporal": temporal,
        "calibration": calibration,
        "conditional": conditional,
        "intents": intents,
        "keys_chart": keys_chart,
        "intent_chart": intent_chart,
        "calibration_chart": calibration_chart,
        "per_key_table": per_key_table,
        "top_chatterers": top_chatterers,
        "ece_table": ece_table,
    }


# Threshold functions as custom filters
def _f1_class(value):
    """CSS class for F1 score thresholds."""
    if value > 0.7:
        return "good"
    if value > 0.4:
        return "warning"
    return "bad"


def _chatter_class(value):
    """CSS class for chatter rate thresholds."""
    if value < 0.1:
        return "good"
    if value < 0.3:
        return "warning"
    return "bad"


def _ece_class(value):
    """CSS class for calibration error thresholds."""
    if value < 0.05:
        return "good"
    if value < 0.1:
        return "warning"
    return "bad"


def _stability_class(value):
    """CSS class for stability ratio thresholds."""
    if 0.8 <= value <= 1.2:
        return "good"
    if 0.5 <= value <= 1.5:
        return "warning"
    return "bad"


def _boundary_diff_class(value):
    """CSS class for episode boundary difference thresholds."""
    if value < 0.05:
        return "good"
    if value < 0.1:
        return "warning"
    return "bad"


def _value_class(r: float) -> str:
    """CSS class for value head correlation thresholds."""
    if r > 0.5:
        return "good"
    if r > 0.2:
        return "warning"
    return "bad"
