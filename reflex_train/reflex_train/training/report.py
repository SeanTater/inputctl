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
    temporal = results.get("temporal")
    calibration = results.get("calibration")
    conditional = results.get("conditional")
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
                {"<a href='#temporal'>Temporal Coherence</a>" if temporal else ""}
                {"<a href='#calibration'>Calibration</a>" if calibration else ""}
                {"<a href='#conditional'>Conditional Perf</a>" if conditional else ""}
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
            {_temporal_section(temporal) if temporal else ""}
            {_calibration_section(calibration) if calibration else ""}
            {_conditional_section(conditional) if conditional else ""}
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

        // Calibration reliability diagram
        {_calibration_chart_js(calibration)}
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


def _temporal_section(temporal: dict) -> str:
    chatter_rate = temporal.get("key_chatter_rate", 0)
    chatter_class = "good" if chatter_rate < 0.1 else "warning" if chatter_rate < 0.3 else "bad"
    stability = temporal.get("intent_stability_ratio", 0)
    stability_class = "good" if 0.8 <= stability <= 1.2 else "warning" if 0.5 <= stability <= 1.5 else "bad"

    # Top chatterers
    chatter_per_key = temporal.get("key_chatter_per_key", {})
    top_chatterers = sorted(chatter_per_key.items(), key=lambda x: x[1], reverse=True)[:10]
    chatter_rows = "".join(
        f'<tr><td>{k}</td><td class="{"bad" if v > 0.3 else "warning" if v > 0.1 else "good"}">{v:.3f}</td></tr>'
        for k, v in top_chatterers
    )

    return f"""
    <section id="temporal">
        <h2>Temporal Coherence</h2>
        <div class="explanation">
            <strong>What this measures:</strong> How stable are predictions over time? Lower chatter means fewer flip-flops between adjacent frames.
            Stability ratio compares predicted intent run lengths to ground truth (1.0 = perfect match).
            Chatter &lt;0.1 is good; &gt;0.3 suggests jittery predictions.
        </div>
        <div class="card">
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value {chatter_class}">{chatter_rate:.3f}</div>
                    <div class="metric-label">Key Chatter Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{temporal.get('intent_mean_run_pred', 0):.1f}</div>
                    <div class="metric-label">Pred Run Length</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{temporal.get('intent_mean_run_gt', 0):.1f}</div>
                    <div class="metric-label">GT Run Length</div>
                </div>
                <div class="metric">
                    <div class="metric-value {stability_class}">{stability:.2f}</div>
                    <div class="metric-label">Stability Ratio</div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>Top 10 Chattiest Keys</h3>
            <table>
                <thead><tr><th>Key</th><th>Chatter Rate</th></tr></thead>
                <tbody>{chatter_rows if chatter_rows else "<tr><td colspan='2'>No data</td></tr>"}</tbody>
            </table>
        </div>
    </section>"""


def _calibration_section(calibration: dict) -> str:
    ece = calibration.get("ece", 0)
    ece_class = "good" if ece < 0.05 else "warning" if ece < 0.1 else "bad"

    # Per-key ECE table
    per_key_ece = calibration.get("per_key_ece", {})
    ece_rows = "".join(
        f'<tr><td>{k}</td><td class="{"good" if v < 0.05 else "warning" if v < 0.1 else "bad"}">{v:.4f}</td></tr>'
        for k, v in sorted(per_key_ece.items(), key=lambda x: x[1], reverse=True)
    )

    return f"""
    <section id="calibration">
        <h2>Calibration</h2>
        <div class="explanation">
            <strong>What this measures:</strong> Do predicted probabilities match actual frequencies?
            ECE (Expected Calibration Error) &lt;0.05 is well-calibrated. The reliability diagram shows predicted probability vs actual positive rate.
            Perfect calibration = points on diagonal.
        </div>
        <div class="card">
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value {ece_class}">{ece:.4f}</div>
                    <div class="metric-label">ECE (lower is better)</div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>Reliability Diagram</h3>
            <div class="chart" id="calibration-chart"></div>
        </div>
        <div class="card">
            <h3>Per-Key ECE (Top 10)</h3>
            <table>
                <thead><tr><th>Key</th><th>ECE</th></tr></thead>
                <tbody>{ece_rows if ece_rows else "<tr><td colspan='2'>No data</td></tr>"}</tbody>
            </table>
        </div>
    </section>"""


def _calibration_chart_js(calibration: dict | None) -> str:
    if not calibration:
        return ""

    bins = calibration.get("bins", [])
    if not bins:
        return ""

    bin_centers = [b["bin_center"] for b in bins]
    mean_preds = [b["mean_pred"] for b in bins]
    actual_freqs = [b["actual_freq"] for b in bins]
    counts = [b["count"] for b in bins]

    return f"""
        if (document.getElementById('calibration-chart')) {{
            Plotly.newPlot('calibration-chart', [
                {{
                    x: [0, 1],
                    y: [0, 1],
                    mode: 'lines',
                    name: 'Perfect',
                    line: {{ color: '#a8a8a8', dash: 'dash' }}
                }},
                {{
                    x: {json.dumps(mean_preds)},
                    y: {json.dumps(actual_freqs)},
                    mode: 'markers+lines',
                    name: 'Model',
                    marker: {{ color: '#4ecca3', size: {json.dumps([max(5, min(20, c/1000)) for c in counts])} }},
                    text: {json.dumps([f"n={c:,}" for c in counts])},
                    hovertemplate: 'Pred: %{{x:.2f}}<br>Actual: %{{y:.2f}}<br>%{{text}}<extra></extra>'
                }}
            ], {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {{ color: '#e8e8e8' }},
                xaxis: {{ title: 'Mean Predicted Probability', range: [0, 1] }},
                yaxis: {{ title: 'Actual Positive Rate', range: [0, 1] }},
                showlegend: true,
                legend: {{ x: 0.02, y: 0.98 }},
                margin: {{ t: 20 }}
            }}, {{ responsive: true }});
        }}
    """


def _conditional_section(conditional: dict) -> str:
    # F1 by intent table
    f1_by_intent = conditional.get("key_f1_by_intent", {})
    intent_rows = "".join(
        f'<tr><td>{k}</td><td class="{"good" if v > 0.7 else "warning" if v > 0.4 else "bad"}">{v:.3f}</td></tr>'
        for k, v in sorted(f1_by_intent.items(), key=lambda x: x[1], reverse=True)
    )

    # F1 by rarity table
    f1_by_rarity = conditional.get("key_f1_by_rarity", {})
    rarity_rows = "".join(
        f'<tr><td>{k}</td><td class="{"good" if v > 0.7 else "warning" if v > 0.4 else "bad"}">{v:.3f}</td></tr>'
        for k, v in f1_by_rarity.items()
    )

    # Episode boundary metrics
    f1_near = conditional.get("f1_near_episode_boundary", 0)
    f1_away = conditional.get("f1_away_from_boundary", 0)
    boundary_diff = f1_away - f1_near if f1_near > 0 and f1_away > 0 else 0

    return f"""
    <section id="conditional">
        <h2>Conditional Performance</h2>
        <div class="explanation">
            <strong>What this measures:</strong> How does performance vary by context?
            F1 by intent shows model strength per player state. F1 by key rarity shows performance on rare vs common keys.
            Episode boundary performance compares accuracy near game-over events vs normal play.
        </div>
        <div class="card">
            <h3>Episode Boundary Performance</h3>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{f1_near:.3f}</div>
                    <div class="metric-label">F1 Near Boundary (±5 frames)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{f1_away:.3f}</div>
                    <div class="metric-label">F1 Away from Boundary</div>
                </div>
                <div class="metric">
                    <div class="metric-value {"good" if boundary_diff < 0.05 else "warning" if boundary_diff < 0.1 else "bad"}">{boundary_diff:+.3f}</div>
                    <div class="metric-label">Difference</div>
                </div>
            </div>
        </div>
        <div class="card">
            <h3>Key F1 by Intent</h3>
            <table>
                <thead><tr><th>Intent</th><th>Key F1 (micro)</th></tr></thead>
                <tbody>{intent_rows if intent_rows else "<tr><td colspan='2'>No data</td></tr>"}</tbody>
            </table>
        </div>
        <div class="card">
            <h3>Key F1 by Rarity</h3>
            <table>
                <thead><tr><th>Bucket</th><th>Key F1 (micro)</th></tr></thead>
                <tbody>{rarity_rows if rarity_rows else "<tr><td colspan='2'>No data</td></tr>"}</tbody>
            </table>
        </div>
    </section>"""
