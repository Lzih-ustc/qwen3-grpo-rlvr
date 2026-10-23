#!/usr/bin/env python3
"""Render GitHub-friendly SVG summaries from frozen Phase C JSON artifacts.

Uses only Python's standard library; this is a deterministic report renderer,
not a model evaluation or a source of new experiment numbers.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
COLORS = {
    "ink": "#152238",
    "muted": "#617089",
    "grid": "#dbe3ef",
    "paper": "#ffffff",
    "panel": "#f6f8fc",
    "base": "#64748b",
    "g4": "#0f766e",
    "g8": "#d97706",
    "good": "#16803c",
    "bad": "#b42318",
    "both": "#8995a8",
    "selected": "#5b4bdb",
}


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif} .mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}</style>',
    ]


def text(x: float, y: float, value: Any, size: int = 18, color: str | None = None, weight: int = 400, anchor: str = "start") -> str:
    fill = color or COLORS["ink"]
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(value)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, rx: int = 8, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{fill}"{stroke_attr}/>'


def line(x1: float, y1: float, x2: float, y2: float, color: str | None = None, width: int = 1) -> str:
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color or COLORS["grid"]}" stroke-width="{width}"/>'


def write_svg(filename: str, content: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    path.write_text("\n".join(content + ["</svg>"]) + "\n", encoding="utf-8")


def phasec_validation_figure() -> None:
    files = {
        "seed1": {
            "base": "reports/gsm8k_phasec_base_validation.json",
            "g4": "reports/gsm8k_phasec_g4_validation.json",
            "g8": "reports/gsm8k_phasec_g8_validation.json",
        },
        "seed2": {
            "base": "reports/gsm8k_phasec_seed2_base_validation.json",
            "g4": "reports/gsm8k_phasec_seed2_g4_validation.json",
            "g8": "reports/gsm8k_phasec_seed2_g8_validation.json",
        },
    }
    values = {seed: {arm: load(path) for arm, path in arms.items()} for seed, arms in files.items()}
    out = svg_start(1320, 670, "Phase C group-size validation results", "Two fixed-token-budget seeds with Base, Group4, and Group8 validation accuracy. The winner reverses across seeds.")
    out += [text(48, 52, "Phase C  |  Group-size validation", 30, weight=700), text(48, 83, "500 GSM8K validation questions per seed  ·  exact_format reward  ·  approximately 16k generated training tokens per arm", 16, COLORS["muted"])]
    for i, seed in enumerate(("seed1", "seed2")):
        x = 44 + i * 634
        out.append(rect(x, 118, 596, 430, COLORS["panel"], 16, COLORS["grid"]))
        g4 = values[seed]["g4"]["correct_count"]
        g8 = values[seed]["g8"]["correct_count"]
        winner = "Group8" if g8 > g4 else "Group4"
        delta = 100 * (values[seed]["g4"]["accuracy"] - values[seed]["g8"]["accuracy"])
        out.append(text(x + 26, 157, f"{seed.title()}  ·  validation winner: {winner}", 21, weight=700))
        out.append(text(x + 26, 185, f"Group4 − Group8: {delta:+.1f} percentage points", 15, COLORS["muted"]))
        bar_x, bar_width = x + 128, 378
        for j, arm in enumerate(("base", "g4", "g8")):
            payload = values[seed][arm]
            y = 245 + 82 * j
            count = int(payload["correct_count"])
            accuracy = float(payload["accuracy"])
            color = COLORS[arm]
            label = {"base": "Base", "g4": "Group4", "g8": "Group8"}[arm]
            out.append(text(x + 28, y + 6, label, 17, weight=600))
            out.append(rect(bar_x, y - 14, bar_width, 26, "#e5eaf2", 6))
            out.append(rect(bar_x, y - 14, bar_width * accuracy / 0.45, 26, color, 6))
            out.append(text(bar_x + bar_width * accuracy / 0.45 + 12, y + 6, f"{count}/500  ({100*accuracy:.1f}%)", 16, color, 700))
        out.append(text(x + 28, 512, "Validation is for candidate selection; test does not compare G4 with G8.", 13, COLORS["muted"]))
    out.append(rect(44, 574, 1230, 52, "#fff7ed", 10))
    out.append(text(64, 606, "Exit verdict: MIXED  ·  Seed 1 selects Group8; Seed 2 selects Group4. Two seeds do not establish a stable group-size winner.", 17, "#9a4d00", 700))
    write_svg("phasec_validation.svg", out)


def reward_signal_figure() -> None:
    sources = [
        ("Seed 1 · G4", "reports/gsm8k_phasec_g4_rollout_audit.json", "g4"),
        ("Seed 1 · G8", "reports/gsm8k_phasec_g8_rollout_audit.json", "g8"),
        ("Seed 2 · G4", "reports/gsm8k_phasec_seed2_g4_rollout_audit.json", "g4"),
        ("Seed 2 · G8", "reports/gsm8k_phasec_seed2_g8_rollout_audit.json", "g8"),
    ]
    out = svg_start(1320, 720, "GRPO reward-group diagnostics", "Reward-variable groups, correctness-mixed groups, and correctness-mixed groups per thousand generated tokens across both seeds.")
    out += [text(48, 52, "GRPO  |  What signal appeared inside each sampled group?", 28, weight=700), text(48, 83, "All counts are reconstructed from saved rollout completions; a nonzero reward variance can come from format-only reward differences.", 15, COLORS["muted"])]
    headers = [(48, "Candidate", 240), (340, "Nonzero reward variance", 260), (680, "Correctness-mixed groups", 270), (1035, "Mixed groups / 1k tokens", 250)]
    for x, label, _w in headers:
        out.append(text(x, 148, label, 16, weight=700))
    out.append(line(48, 168, 1270, 168, COLORS["grid"], 2))
    for i, (label, path, arm) in enumerate(sources):
        data = load(path)
        a = data["rollout_analysis"]
        total = int(a["generation_groups"])
        variance = int(a["nonzero_reward_variance_groups"])
        mixed = int(a["groups_with_mixed_correct_and_incorrect"])
        density = float(a["correctness_mixed_groups_per_1000_tokens"])
        y = 224 + i * 100
        color = COLORS[arm]
        out.append(rect(48, y - 26, 18, 18, color, 4))
        out.append(text(80, y - 10, label, 17, weight=600))
        out.append(rect(340, y - 27, 220, 20, "#e5eaf2", 5))
        out.append(rect(340, y - 27, 220 * variance / total, 20, color, 5))
        out.append(text(574, y - 10, f"{variance}/{total}  ({100*variance/total:.1f}%)", 15, color, 700))
        out.append(rect(680, y - 27, 220, 20, "#e5eaf2", 5))
        out.append(rect(680, y - 27, 220 * mixed / total, 20, color, 5))
        out.append(text(914, y - 10, f"{mixed}/{total}  ({100*mixed/total:.1f}%)", 15, color, 700))
        out.append(rect(1035, y - 27, 165, 20, "#e5eaf2", 5))
        out.append(rect(1035, y - 27, 165 * density, 20, color, 5))
        out.append(text(1212, y - 10, f"{density:.3f}", 15, color, 700))
    out.append(line(48, 598, 1270, 598, COLORS["grid"], 2))
    out.append(text(48, 633, "Mixed = at least one correct and one incorrect completion in the same prompt group.", 15, weight=600))
    out.append(text(48, 660, "Group8 has a higher mixed-group share, but fewer absolute mixed groups and lower mixed groups per 1k tokens in both seeds.", 14, COLORS["muted"]))
    out.append(text(48, 686, "This is a verifier-signal-density proxy, not gradient efficacy, causal evidence, or model quality.", 14, COLORS["muted"], 600))
    write_svg("phasec_reward_signal.svg", out)


def svamp_cost_figure() -> None:
    audit = load("reports/svamp_phasec_seed2_final_test_audit.json")
    base = audit["base"]
    selected = audit["selected"]
    paired = audit["paired_outcomes"]
    if audit["eval_count"] != 1000:
        raise ValueError("Expected the locked 1,000-item SVAMP test audit")
    out = svg_start(1440, 850, "SVAMP selected-adapter result and inference costs", "One-time external paired test for seed 2: base versus validation-selected Group4 adapter, with paired correctness outcomes and measured inference costs.")
    out += [text(48, 52, "Seed 2  |  SVAMP external paired test and inference cost", 28, weight=700), text(48, 83, "1,000 fixed test items  ·  selected candidate: Group4  ·  paired McNemar p = 2.07e−85", 16, COLORS["muted"])]
    # Accuracy and paired outcomes card.
    out.append(rect(42, 116, 470, 666, COLORS["panel"], 16, COLORS["grid"]))
    out.append(text(70, 158, "Exact match", 21, weight=700))
    for i, (label, payload, color) in enumerate((("Base", base, COLORS["base"]), ("Selected G4", selected, COLORS["selected"]))):
        y = 222 + i * 78
        acc = float(payload["accuracy"])
        out.append(text(70, y + 5, label, 16, weight=600))
        out.append(rect(180, y - 16, 250, 24, "#e5eaf2", 6))
        out.append(rect(180, y - 16, 250 * acc, 24, color, 6))
        out.append(text(442, y + 5, f"{int(payload['correct_count'])}/1,000  ({100*acc:.1f}%)", 15, color, 700))
    out.append(rect(70, 305, 400, 48, "#f0edff", 9))
    out.append(text(88, 335, f"Paired accuracy change: {float(audit['accuracy_delta_percentage_points']):+.1f} percentage points", 15, "#4433b6", 700))
    out.append(text(70, 403, "Question-level correctness transitions", 17, weight=700))
    outcomes = [
        ("Both correct", int(paired["both_correct"]), COLORS["both"]),
        ("Base only", int(paired["base_only_correct"]), COLORS["base"]),
        ("Adapter only", int(paired["selected_only_correct"]), COLORS["good"]),
        ("Both wrong", int(paired["both_wrong"]), COLORS["bad"]),
    ]
    for i, (label, count, color) in enumerate(outcomes):
        y = 452 + i * 67
        out.append(text(70, y + 5, label, 15, weight=600))
        out.append(rect(185, y - 14, 225, 22, "#e5eaf2", 5))
        out.append(rect(185, y - 14, 225 * count / 500, 22, color, 5))
        out.append(text(426, y + 5, count, 15, color, 700))
    out.append(text(70, 742, "Test is consumed; this compares selected G4 against Base, not G4 against G8.", 13, COLORS["muted"]))

    # Inference cost table card.
    out.append(rect(540, 116, 860, 666, COLORS["paper"], 16, COLORS["grid"]))
    out.append(text(575, 158, "Measured cost for the same 1,000 questions", 21, weight=700))
    out.append(text(575, 190, "Greedy decoding · max_new_tokens=192 · thinking disabled", 14, COLORS["muted"]))
    cols = [575, 895, 1068, 1236]
    for x, label in zip(cols, ("Metric", "Base", "Selected G4", "Adapter / Base")):
        out.append(text(x, 242, label, 14, COLORS["muted"], 700))
    out.append(line(575, 258, 1365, 258, COLORS["grid"], 2))
    elapsed_ratio = selected["elapsed_seconds"] / base["elapsed_seconds"]
    tokens_ratio = selected["generated_tokens"] / base["generated_tokens"]
    metric_rows = [
        ("Generated tokens", f"{base['generated_tokens']:,}", f"{selected['generated_tokens']:,}", f"{tokens_ratio:.2f}×"),
        ("Wall time", f"{base['elapsed_seconds']/60:.1f} min", f"{selected['elapsed_seconds']/60:.1f} min", f"{elapsed_ratio:.2f}×"),
        ("Generation throughput", f"{base['tokens_per_second']:.2f} tok/s", f"{selected['tokens_per_second']:.2f} tok/s", f"{selected['tokens_per_second']/base['tokens_per_second']:.2f}×"),
        ("Whole-GPU peak memory", f"{base['whole_gpu_peak_memory_used_mib']:,} MiB", f"{selected['whole_gpu_peak_memory_used_mib']:,} MiB", f"{selected['whole_gpu_peak_memory_used_mib']/base['whole_gpu_peak_memory_used_mib']:.2f}×"),
    ]
    for i, (label, base_value, selected_value, ratio) in enumerate(metric_rows):
        y = 310 + i * 105
        out.append(text(cols[0], y, label, 15, weight=600))
        out.append(text(cols[1], y, base_value, 15, COLORS["base"], 700))
        out.append(text(cols[2], y, selected_value, 15, COLORS["selected"], 700))
        out.append(text(cols[3], y, ratio, 15, COLORS["ink"], 700))
        track_x, track_y, track_w = 575, y + 16, 720
        out.append(rect(track_x, track_y, track_w, 12, "#e5eaf2", 4))
        out.append(rect(track_x, track_y, track_w * metric_value(base, i) / metric_numeric(base, selected, i), 12, COLORS["base"], 4))
        out.append(rect(track_x, track_y + 15, track_w, 12, "#e5eaf2", 4))
        out.append(rect(track_x, track_y + 15, track_w * metric_value(selected, i) / metric_numeric(base, selected, i), 12, COLORS["selected"], 4))
    out.append(rect(575, 746, 790, 18, "#fff7ed", 6))
    out.append(text(575, 724, "The adapter answers more questions but uses 2.94× tokens and 5.06× wall time; this is not an efficiency gain.", 14, "#9a4d00", 600))
    out.append(text(575, 769, "Source: reports/svamp_phasec_seed2_final_test_audit.json  ·  bars are scaled independently within each metric.", 12, COLORS["muted"]))
    write_svg("phasec_svamp_cost.svg", out)


def metric_numeric(base: dict[str, Any], selected: dict[str, Any], index: int) -> float:
    if index == 0:
        return float(max(base["generated_tokens"], selected["generated_tokens"]))
    if index == 1:
        return float(max(base["elapsed_seconds"], selected["elapsed_seconds"]))
    if index == 2:
        return float(max(base["tokens_per_second"], selected["tokens_per_second"]))
    return float(max(base["whole_gpu_peak_memory_used_mib"], selected["whole_gpu_peak_memory_used_mib"]))


def metric_value(payload: dict[str, Any], index: int) -> float:
    if index == 0:
        return float(payload["generated_tokens"])
    if index == 1:
        return float(payload["elapsed_seconds"])
    if index == 2:
        return float(payload["tokens_per_second"])
    return float(payload["whole_gpu_peak_memory_used_mib"])


def main() -> None:
    phasec_validation_figure()
    reward_signal_figure()
    svamp_cost_figure()
    print(json.dumps({"status": "written", "figures": ["phasec_validation.svg", "phasec_reward_signal.svg", "phasec_svamp_cost.svg"]}))


if __name__ == "__main__":
    main()
