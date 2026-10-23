#!/usr/bin/env python3
"""Export compact CSV and SVG diagnostics from one or more HF Trainer states."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]
FIELDS = [
    "run", "step", "reward", "reward_std", "frac_reward_zero_std",
    "completions/clipped_ratio", "completions/mean_length", "grad_norm", "step_time",
]


def load_series(spec: str) -> tuple[str, list[dict]]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("run must be LABEL=TRAINER_STATE_JSON")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in state.get("log_history", []) if "step" in row and "reward" in row]
    if not rows:
        raise ValueError(f"no reward log rows found in {path}")
    return label, rows


def svg_plot(series: list[tuple[str, list[dict]]], metric: str, title: str,
             output: Path, *, width: int = 1100, height: int = 430) -> None:
    margin = {"left": 82, "right": 230, "top": 58, "bottom": 58}
    x0, y0 = margin["left"], margin["top"]
    pw, ph = width - x0 - margin["right"], height - y0 - margin["bottom"]
    vals = [float(row[metric]) for _, rows in series for row in rows if row.get(metric) is not None]
    if not vals:
        raise ValueError(f"no values for metric {metric}")
    ymin, ymax = min(0.0, min(vals)), max(vals)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    max_step = max(float(row["step"]) for _, rows in series for row in rows)

    def xmap(value: float) -> float:
        return x0 + pw * value / max_step if max_step else x0

    def ymap(value: float) -> float:
        return y0 + ph * (ymax - value) / (ymax - ymin)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{x0}" y="32" font-family="sans-serif" font-size="21" font-weight="bold">{html.escape(title)}</text>',
    ]
    ticks = 5
    for i in range(ticks + 1):
        value = ymin + (ymax - ymin) * i / ticks
        y = ymap(value)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#374151">{value:.3g}</text>')
        xvalue = max_step * i / ticks
        x = xmap(xvalue)
        out.append(f'<line x1="{x:.1f}" y1="{y0 + ph}" x2="{x:.1f}" y2="{y0 + ph + 5}" stroke="#111827"/>')
        out.append(f'<text x="{x:.1f}" y="{y0 + ph + 21}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">{xvalue:g}</text>')
    out.extend([
        f'<line x1="{x0}" y1="{y0 + ph}" x2="{x0 + pw}" y2="{y0 + ph}" stroke="#111827"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + ph}" stroke="#111827"/>',
        f'<text x="{x0 + pw / 2:.1f}" y="{height - 14}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">Optimizer update</text>',
    ])
    for idx, (label, rows) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        points = " ".join(f'{xmap(float(row["step"])):.1f},{ymap(float(row[metric])):.1f}' for row in rows if row.get(metric) is not None)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
        for row in rows:
            if row.get(metric) is not None:
                out.append(f'<circle cx="{xmap(float(row["step"])):.1f}" cy="{ymap(float(row[metric])):.1f}" r="2.4" fill="{color}"/>')
        ly = y0 + 18 + idx * 25
        out.append(f'<line x1="{x0 + pw + 18}" y1="{ly - 4}" x2="{x0 + pw + 42}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{x0 + pw + 49}" y="{ly}" font-family="sans-serif" font-size="12" fill="#111827">{html.escape(label)}</text>')
    out.append('</svg>')
    output.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=trainer_state.json (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    series = [load_series(item) for item in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for label, rows in series:
            for row in rows:
                writer.writerow({"run": label, **{key: row.get(key, "") for key in FIELDS if key != "run"}})
    svg_plot(series, "reward", "Training reward per optimizer update (diagnostic only)", args.output_dir / "training_reward.svg")
    svg_plot(series, "completions/clipped_ratio", "Completion clipping ratio per optimizer update", args.output_dir / "training_clipping.svg")


if __name__ == "__main__":
    main()
