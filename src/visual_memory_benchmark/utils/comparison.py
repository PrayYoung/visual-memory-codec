from __future__ import annotations

from pathlib import Path


def write_comparison_html(rows: list[dict], output_path: Path, max_examples: int = 20) -> None:
    selected = rows[:max_examples]
    html = [
        "<html><head><meta charset='utf-8'><title>Comparison Grid</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;} .card{border:1px solid #ddd;padding:10px;} img{width:100%;height:auto;} .meta{font-size:12px;color:#333;line-height:1.5;}</style>",
        "</head><body>",
        "<h1>Natural-image comparison grid</h1>",
    ]
    for row in selected:
        html.append(f"<h2>{row['sample_id']}</h2><div class='grid'>")
        for column in ("original", "text_only_real", "visual_latent_real"):
            item = row[column]
            html.append("<div class='card'>")
            html.append(f"<div><strong>{column}</strong></div>")
            html.append(f"<img src='{item['image_path']}' alt='{column}'>")
            html.append("<div class='meta'>")
            html.append(f"stored_bytes: {item['stored_bytes']}<br>")
            html.append(f"semantic_similarity: {item['semantic_similarity']:.4f}<br>")
            html.append(f"dino_similarity: {item['dino_similarity']:.4f}<br>")
            html.append(f"scene_qa_accuracy: {item['scene_qa_accuracy']:.4f}<br>")
            html.append("</div></div>")
        html.append("</div>")
    html.append("</body></html>")
    output_path.write_text("".join(html))
