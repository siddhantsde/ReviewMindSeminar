"""Portable, print-friendly HTML and Markdown report exports."""
from datetime import datetime
from html import escape


def _priority_lines(evidence):
    rows = evidence.get("pain_points", [])
    if not rows:
        return ["- No recurring taxonomy issue met the threshold."]
    return [
        f"- **{r.get('issue')}** — {r.get('review_count')} of {r.get('eligible_reviews')} eligible reviews ({r.get('share_pct')}%); negative {r.get('negative_pct')}%; priority {r.get('priority_score')}/100 ({r.get('priority_tier')})."
        for r in rows[:10]
    ]


def make_markdown_report(dataset_name, metrics, findings, topics, recommendations, report, generation_method, evidence=None):
    evidence = evidence or {}
    avg = metrics.get("average_rating")
    avg_text = f"{avg}/5" if avg is not None else "Not provided"
    lines = [
        "# ReviewMind | Product Review Intelligence", "",
        f"**Dataset:** {dataset_name}  ",
        f"**Generated:** {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}  ",
        f"**Method:** {generation_method}", "",
        "## Executive metrics", "",
        f"- Reviews analyzed: {metrics.get('review_count', 0)}",
        f"- Average rating: {avg_text}",
        f"- Positive sentiment: {metrics.get('positive_pct', 0)}%",
        f"- Neutral sentiment: {metrics.get('neutral_pct', 0)}%",
        f"- Negative sentiment: {metrics.get('negative_pct', 0)}%",
        f"- Products: {metrics.get('product_count', 0)}", "", "## Key findings",
    ]
    lines += [f"- {x}" for x in findings] or ["- No findings available."]
    lines += ["", "## Discovered topics", "", "| Topic | Keywords | Reviews |", "|---|---|---:|"]
    lines += [f"| {r.get('topic', '')} | {r.get('keywords', '')} | {r.get('review_count', '')} |" for r in topics] or ["| No topics | Add more varied reviews | — |"]
    lines += ["", "## Customer pain points and heuristic priority", "", *_priority_lines(evidence)]
    lines += ["", "## Potential hidden problems"]
    hidden = evidence.get("hidden_problems", [])
    lines += [f"- {r.get('reason')}" for r in hidden[:10]] or ["- No product matched the configured investigation rule."]
    lines += ["", "## Emerging issue changes"]
    trends = evidence.get("trends", [])
    lines += [f"- {r.get('issue')}: mention share {r.get('issue_share_pp_change'):+.1f} pp; negative share {r.get('negative_pp_change'):+.1f} pp (matching reviews n={r.get('previous_reviews')} → {r.get('current_reviews')})." for r in trends[:10]] or ["- No issue changes met minimum sample thresholds or dates were unavailable."]
    lines += ["", "## Recommendations"]
    for r in recommendations:
        lines += [f"- **{r.get('finding', '')}** {r.get('action', '')} _(Evidence strength: {r.get('confidence', '')}; priority: {r.get('priority_tier', 'Monitor')})_"]
        lines += [f"  - Evidence: “{q}”" for q in r.get("evidence", [])]
    if not recommendations:
        lines.append("- No rule-based recommendations were triggered.")
    lines += ["", "## Insight report", "", report, "", "## Representative review evidence"]
    for label, key in [("Positive", "positive"), ("Negative", "negative")]:
        lines += ["", f"### {label}"]
        lines += [f"> {q}" for q in evidence.get(key, [])] or ["No excerpts available."]
    lines += ["", "## Data quality snapshot"]
    quality = evidence.get("data_quality", {})
    lines += [f"- {k}: {v}" for k, v in quality.items()] or ["- No quality snapshot available."]
    lines += ["", "## Methodology and limitations", "Sentiment and topics are approximate text-analysis results. The supplied reviews may not represent all customers. Priority is a configurable heuristic, not a validated risk model. Recommendations flag areas for investigation; they do not prove causes or predict business outcomes. The bundled demonstration data, when used, is synthetic."]
    return "\n".join(lines)


def make_html_report(dataset_name, metrics, findings, topics, recommendations, report, generation_method, evidence=None):
    esc = lambda value: escape(str(value))
    evidence = evidence or {}
    avg = metrics.get("average_rating")
    avg_text = f"{avg}/5" if avg is not None else "Not provided"
    cards = [("Reviews", metrics.get("review_count", 0)), ("Average rating", avg_text), ("Positive", f"{metrics.get('positive_pct', 0)}%"), ("Negative", f"{metrics.get('negative_pct', 0)}%"), ("Products", metrics.get("product_count", 0))]
    card_html = "".join(f"<div class='metric'><span>{esc(k)}</span><strong>{esc(v)}</strong></div>" for k, v in cards)
    finding_html = "".join(f"<li>{esc(x)}</li>" for x in findings) or "<li>No findings available.</li>"
    topic_html = "".join(f"<tr><td>{esc(r.get('topic', ''))}</td><td>{esc(r.get('keywords', ''))}</td><td>{esc(r.get('review_count', ''))}</td></tr>" for r in topics) or "<tr><td colspan='3'>No topics were available.</td></tr>"
    rec_html = "".join(f"<article><b>{esc(r.get('finding', ''))}</b><p>{esc(r.get('action', ''))}</p><small>Evidence: {esc(r.get('confidence', ''))} · Priority: {esc(r.get('priority_tier', 'Monitor'))}</small>{''.join('<blockquote>'+esc(q)+'</blockquote>' for q in r.get('evidence', []))}</article>" for r in recommendations) or "<p>No rule-based recommendations were triggered.</p>"
    issue_html = "".join(f"<tr><td>{esc(r.get('issue'))}</td><td>{esc(r.get('review_count'))}/{esc(r.get('eligible_reviews'))} ({esc(r.get('share_pct'))}%)</td><td>{esc(r.get('negative_pct'))}%</td><td>{esc(r.get('priority_score'))}/100 · {esc(r.get('priority_tier'))}</td></tr>" for r in evidence.get('pain_points', [])[:10]) or "<tr><td colspan='4'>No recurring issue met the threshold.</td></tr>"
    hidden_html = "".join(f"<li>{esc(r.get('reason'))}</li>" for r in evidence.get('hidden_problems', [])[:10]) or "<li>No product matched the configured investigation rule.</li>"
    trend_html = "".join(f"<li>{esc(r.get('issue'))}: mention share {esc(r.get('issue_share_pp_change'))} percentage points; negative share {esc(r.get('negative_pp_change'))} percentage points (n={esc(r.get('previous_reviews'))} → {esc(r.get('current_reviews'))}).</li>" for r in evidence.get('trends', [])[:10]) or "<li>No issue changes met thresholds or dates were unavailable.</li>"
    evidence_html = "".join(f"<h3>{name}</h3>" + ("".join(f"<blockquote>{esc(q)}</blockquote>" for q in evidence.get(key, [])) or "<p class='muted'>No excerpts available.</p>") for name, key in [("Positive evidence", "positive"), ("Negative evidence", "negative")])
    quality_html = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in evidence.get('data_quality', {}).items()) or "<tr><td colspan='2'>No quality metrics available.</td></tr>"
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    synthetic_note = "This bundled demonstration dataset is synthetic and is not real-world consumer research." if "synthetic" in str(dataset_name).lower() else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="ReviewMind evidence-backed product intelligence report"><title>ReviewMind | {esc(dataset_name)}</title><style>
</style></head><body><a class="skip-link" href="#main-content">Skip to main content</a><main id="main-content" tabindex="-1"><header><div class="brand">ReviewMind · Product Intelligence</div><h1>Customer Review Analysis</h1><p><b>Dataset:</b> {esc(dataset_name)} · <b>Generated:</b> {esc(now)}<br><b>Generation method:</b> {esc(generation_method)}</p><div class="metrics">{card_html}</div><p>{esc(synthetic_note)}</p></header><section class="panel"><h2>Executive findings</h2><ul>{finding_html}</ul></section><section class="panel"><h2>Discovered review topics</h2><table><thead><tr><th>Topic</th><th>Keywords</th><th>Reviews</th></tr></thead><tbody>{topic_html}</tbody></table></section><section class="panel"><h2>Customer pain points &amp; heuristic priority</h2><p>Priority scores are configurable heuristics, not validated risk estimates.</p><table><thead><tr><th>Issue</th><th>Frequency</th><th>Negative</th><th>Priority</th></tr></thead><tbody>{issue_html}</tbody></table></section><section class="panel"><h2>Potential hidden problems</h2><ul>{hidden_html}</ul></section><section class="panel"><h2>Emerging issue changes</h2><ul>{trend_html}</ul></section><section class="panel"><h2>Recommendations</h2>{rec_html}</section><section class="panel"><h2>Insight report</h2><div class="report">{esc(report)}</div></section><section class="panel"><h2>Representative evidence</h2>{evidence_html}</section><section class="panel"><h2>Data quality snapshot</h2><table>{quality_html}</table></section><section class="panel"><h2>Method and limitations</h2><p>Sentiment and topics are approximate text-analysis results. Supplied reviews may not represent all customers. Keyword matches and priority scores are heuristics. Recommendations flag investigation areas; they do not prove causes or predict business outcomes. {esc(synthetic_note)}</p></section><footer>Prepared with ReviewMind · {esc(generation_method)}</footer></main></body></html>"""
