"""HTML report generator for evaluation results."""

import json
from datetime import datetime
from pathlib import Path


def generate_html_report(results: dict, output_path: str, checkpoint_path: str = ""):
    """Generate a self-contained HTML report with interactive charts."""
    html = _build_html(results, checkpoint_path)
    Path(output_path).write_text(html, encoding="utf-8")


def _build_html(results: dict, checkpoint_path: str) -> str:
    keys = results.get("keys", {})
    intent = results.get("intent", {})
    value = results.get("value", {})
    inv_dyn = results.get("inv_dyn")
    intents = intent.get("intents", [])

    # Prepare data for charts
    per_key = keys.get("per_key", {})
    per_key_sorted = sorted(per_key.items(), key=lambda x: x[1]["support"], reverse=True)

    # Top 20 keys by support
    top_keys = per_key_sorted[:20]
    top_key_names = [k for k, _ in top_keys]
    top_key_f1 = [v["f1"] for _, v in top_keys]
    top_key_support = [v["support"] for _, v in top_keys]

    # Confusion matrix data
    cm = intent.get("confusion_matrix", [[]])
    cm_norm = intent.get("confusion_matrix_normalized", [[]])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reflex Model Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #1f2940;
            --text-primary: #e8e8e8;
            --text-secondary: #a8a8a8;
            --accent: #4ecca3;
            --accent-dim: #3ba888;
            --warning: #f0ad4e;
            --danger: #d9534f;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        .container {{
            display: flex;
            min-height: 100vh;
        }}
        .sidebar {{
            width: 240px;
            background: var(--bg-secondary);
            padding: 24px 16px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }}
        .sidebar h1 {{
            font-size: 1.2rem;
            margin-bottom: 8px;
            color: var(--accent);
        }}
        .sidebar .meta {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 24px;
        }}
        .sidebar nav a {{
            display: block;
            color: var(--text-secondary);
            text-decoration: none;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 4px;
            transition: all 0.2s;
        }}
        .sidebar nav a:hover, .sidebar nav a.active {{
            background: var(--bg-card);
            color: var(--text-primary);
        }}
        .main {{
            margin-left: 240px;
            flex: 1;
            padding: 32px;
            max-width: 1200px;
        }}
        section {{
            margin-bottom: 48px;
        }}
        h2 {{
            color: var(--accent);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--bg-card);
        }}
        .card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }}
        .metric {{
            text-align: center;
            padding: 16px;
            background: var(--bg-secondary);
            border-radius: 8px;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .explanation {{
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 0.9rem;
            color: var(--text-secondary);
            border-left: 3px solid var(--accent);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--bg-secondary);
        }}
        th {{
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
        }}
        .chart {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
        }}
        .good {{ color: var(--accent); }}
        .warning {{ color: var(--warning); }}
        .bad {{ color: var(--danger); }}
        @media (max-width: 768px) {{
            .sidebar {{ display: none; }}
            .main {{ margin-left: 0; padding: 16px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <aside class="sidebar">
            <h1>Reflex Eval</h1>
            <div class="meta">
                Generated: {timestamp}<br>
                {f"Checkpoint: {Path(checkpoint_path).name}" if checkpoint_path else ""}
            </div>
            <nav>
                <a href="#summary">Summary</a>
                <a href="#keys">Keys Head</a>
                <a href="#intent">Intent Head</a>
                <a href="#value">Value Head</a>
                {"<a href='#invdyn'>Inverse Dynamics</a>" if inv_dyn else ""}
            </nav>
        </aside>
        <main class="main">
            <section id="summary">
                <h2>Summary</h2>
                <div class="card">
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{keys.get('f1_micro', 0):.3f}</div>
                            <div class="metric-label">Keys F1 (micro)</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{intent.get('accuracy', 0):.3f}</div>
                            <div class="metric-label">Intent Accuracy</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{value.get('pearson_r', 0):.3f}</div>
                            <div class="metric-label">Value Correlation</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{keys.get('total_samples', 0):,}</div>
                            <div class="metric-label">Test Samples</div>
                        </div>
                    </div>
                </div>
            </section>

            <section id="keys">
                <h2>Keys Head (Multi-Label)</h2>
                <div class="explanation">
                    <strong>What this measures:</strong> The model predicts which keys should be pressed simultaneously (multi-label classification).
                    F1 micro treats all key predictions equally, while F1 macro averages per-key F1 scores (better for rare keys).
                    High precision means few false positives (unwanted key presses), high recall means few missed key presses.
                </div>
                <div class="card">
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{keys.get('f1_micro', 0):.3f}</div>
                            <div class="metric-label">F1 Micro</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{keys.get('f1_macro', 0):.3f}</div>
                            <div class="metric-label">F1 Macro</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{keys.get('precision_micro', 0):.3f}</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{keys.get('recall_micro', 0):.3f}</div>
                            <div class="metric-label">Recall</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h3>Top Keys by Support (F1 Score)</h3>
                    <div class="chart" id="keys-chart"></div>
                </div>
                <div class="card">
                    <h3>Per-Key Breakdown</h3>
                    <table>
                        <thead>
                            <tr><th>Key</th><th>F1</th><th>Precision</th><th>Recall</th><th>Support</th><th>%</th></tr>
                        </thead>
                        <tbody>
                            {"".join(_key_row(k, v) for k, v in per_key_sorted[:30])}
                        </tbody>
                    </table>
                </div>
            </section>

            <section id="intent">
                <h2>Intent Head (Multi-Class)</h2>
                <div class="explanation">
                    <strong>What this measures:</strong> The model classifies the player's current intent from {len(intents)} categories: {', '.join(intents)}.
                    The confusion matrix shows what the model predicts (columns) for each true intent (rows).
                    Diagonal values are correct predictions. Look for patterns in misclassification (e.g., WAIT often confused with RUN).
                </div>
                <div class="card">
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value">{intent.get('accuracy', 0):.3f}</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{intent.get('f1_macro', 0):.3f}</div>
                            <div class="metric-label">F1 Macro</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{intent.get('f1_weighted', 0):.3f}</div>
                            <div class="metric-label">F1 Weighted</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <h3>Confusion Matrix (Row-Normalized)</h3>
                    <div class="chart" id="intent-chart"></div>
                </div>
                <div class="card">
                    <h3>Per-Intent Breakdown</h3>
                    <table>
                        <thead>
                            <tr><th>Intent</th><th>F1</th><th>Precision</th><th>Recall</th><th>Support</th></tr>
                        </thead>
                        <tbody>
                            {"".join(_intent_row(name, intent.get('per_class', {}).get(name, {})) for name in intents)}
                        </tbody>
                    </table>
                </div>
            </section>

            <section id="value">
                <h2>Value Head (Regression)</h2>
                <div class="explanation">
                    <strong>What this measures:</strong> The value head predicts expected discounted returns (how good a state is).
                    Pearson correlation shows linear relationship; Spearman shows rank correlation (useful if relationship is monotonic but not linear).
                    Values near 1.0 indicate strong positive correlation. MSE/MAE show average prediction error magnitude.
                </div>
                <div class="card">
                    <div class="metric-grid">
                        <div class="metric">
                            <div class="metric-value class="{_value_class(value.get('pearson_r', 0))}">{value.get('pearson_r', 0):.3f}</div>
                            <div class="metric-label">Pearson r</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{value.get('spearman_r', 0):.3f}</div>
                            <div class="metric-label">Spearman rho</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{value.get('mse', 0):.4f}</div>
                            <div class="metric-label">MSE</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{value.get('mae', 0):.4f}</div>
                            <div class="metric-label">MAE</div>
                        </div>
                    </div>
                </div>
            </section>

            {_inv_dyn_section(inv_dyn) if inv_dyn else ""}
        </main>
    </div>

    <script>
        // Keys bar chart
        Plotly.newPlot('keys-chart', [{{
            x: {json.dumps(top_key_names)},
            y: {json.dumps(top_key_f1)},
            type: 'bar',
            marker: {{ color: '#4ecca3' }},
            text: {json.dumps([f"Support: {s}" for s in top_key_support])},
            hoverinfo: 'text+y'
        }}], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#e8e8e8' }},
            xaxis: {{ tickangle: -45 }},
            yaxis: {{ title: 'F1 Score', range: [0, 1] }},
            margin: {{ t: 20, b: 100 }}
        }}, {{ responsive: true }});

        // Intent confusion matrix heatmap
        Plotly.newPlot('intent-chart', [{{
            z: {json.dumps(cm_norm)},
            x: {json.dumps(intents)},
            y: {json.dumps(intents)},
            type: 'heatmap',
            colorscale: [[0, '#1a1a2e'], [0.5, '#3ba888'], [1, '#4ecca3']],
            showscale: true,
            text: {json.dumps([[f"{v:.2f}" for v in row] for row in cm_norm])},
            texttemplate: '%{{text}}',
            hovertemplate: 'True: %{{y}}<br>Pred: %{{x}}<br>Rate: %{{z:.2f}}<extra></extra>'
        }}], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#e8e8e8' }},
            xaxis: {{ title: 'Predicted', side: 'bottom' }},
            yaxis: {{ title: 'True', autorange: 'reversed' }},
            margin: {{ t: 20 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>"""


def _key_row(name: str, metrics: dict) -> str:
    f1 = metrics.get("f1", 0)
    cls = "good" if f1 > 0.7 else "warning" if f1 > 0.4 else "bad"
    return f"""<tr>
        <td>{name}</td>
        <td class="{cls}">{f1:.3f}</td>
        <td>{metrics.get('precision', 0):.3f}</td>
        <td>{metrics.get('recall', 0):.3f}</td>
        <td>{metrics.get('support', 0):,}</td>
        <td>{metrics.get('support_pct', 0):.2f}%</td>
    </tr>"""


def _intent_row(name: str, metrics: dict) -> str:
    f1 = metrics.get("f1", 0)
    cls = "good" if f1 > 0.7 else "warning" if f1 > 0.4 else "bad"
    return f"""<tr>
        <td>{name}</td>
        <td class="{cls}">{f1:.3f}</td>
        <td>{metrics.get('precision', 0):.3f}</td>
        <td>{metrics.get('recall', 0):.3f}</td>
        <td>{metrics.get('support', 0):,}</td>
    </tr>"""


def _value_class(r: float) -> str:
    if r > 0.5:
        return "good"
    if r > 0.2:
        return "warning"
    return "bad"


def _inv_dyn_section(inv_dyn: dict) -> str:
    return f"""
    <section id="invdyn">
        <h2>Inverse Dynamics Head</h2>
        <div class="explanation">
            <strong>What this measures:</strong> The inverse dynamics head predicts which keys were held given two consecutive frames.
            Exact match accuracy requires all keys to be correctly predicted for a sample. This is a harder metric than F1.
        </div>
        <div class="card">
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{inv_dyn.get('f1_micro', 0):.3f}</div>
                    <div class="metric-label">F1 Micro</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{inv_dyn.get('f1_macro', 0):.3f}</div>
                    <div class="metric-label">F1 Macro</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{inv_dyn.get('exact_match_accuracy', 0):.3f}</div>
                    <div class="metric-label">Exact Match</div>
                </div>
            </div>
        </div>
    </section>"""
