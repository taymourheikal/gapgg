#!/usr/bin/env python3
"""
visualize_match_diagnosis_v5.py

- Reads a diagnose_match_style_*.json report (produced by diagnose_match_style_v6_demo_fixed_v3.py)
- Produces a dumbbell-style plot on a per-feature ROBUST Z scale so small-magnitude
  features (e.g., dragons/cs) are readable next to big ones (e.g., gold/xp).

Changes vs v4:
1) X-axis is in "IQR units" (robust z) centered on the TARGET median:
      z = (value - target_p50) / pooled_iqr
   pooled_iqr defaults to report["iqr"] if present, else avg(target_iqr, loss_iqr).
2) Adds IQR bars (p25–p75) for both LOSS and TARGET distributions.
3) Output filename remains gaps_dumbbell.png (dashboard-safe).

Usage:
  python visualize_match_diagnosis_v5.py --run-dir runs/... --match-id ... --puuid ... --minute 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _pick_report_path(run_dir: Path, match_id: str, minute: int, puuid: Optional[str]) -> Path:
    reports = run_dir / "reports"
    if not reports.exists():
        raise FileNotFoundError(f"Missing reports dir: {reports}")

    patterns: List[str] = []
    if puuid:
        patterns.append(f"diagnose_match_style_{match_id}_*m{minute}*.json")
    patterns.append(f"diagnose_match_style_{match_id}_*m{minute}*.json")

    for pat in patterns:
        hits = sorted(reports.glob(pat))
        if hits:
            hits = sorted(hits, key=lambda p: p.stat().st_mtime)
            return hits[-1]
    raise FileNotFoundError(f"No report found for match={match_id} minute={minute} under {reports}")


def _extract_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gaps = report.get("gaps")
    if not isinstance(gaps, list):
        return rows

    for g in gaps:
        if not isinstance(g, dict):
            continue
        feat = g.get("feature")
        if not feat:
            continue

        you = _safe_float(g.get("you"))

        # Prefer explicit quantiles if present; fall back to legacy keys.
        target_p25 = _safe_float(g.get("target_p25"))
        target_p50 = _safe_float(g.get("target_p50") if "target_p50" in g else g.get("target_median"))
        target_p75 = _safe_float(g.get("target_p75"))

        loss_p25 = _safe_float(g.get("loss_p25"))
        loss_p50 = _safe_float(g.get("loss_p50") if "loss_p50" in g else g.get("loss_median"))
        loss_p75 = _safe_float(g.get("loss_p75"))

        # IQRs
        target_iqr = _safe_float(g.get("target_iqr"))
        if not (target_iqr == target_iqr):  # nan
            if target_p25 == target_p25 and target_p75 == target_p75:
                target_iqr = target_p75 - target_p25

        loss_iqr = _safe_float(g.get("loss_iqr"))
        if not (loss_iqr == loss_iqr):  # nan
            if loss_p25 == loss_p25 and loss_p75 == loss_p75:
                loss_iqr = loss_p75 - loss_p25

        pooled_iqr = _safe_float(g.get("iqr"))
        if not (pooled_iqr == pooled_iqr):  # nan
            vals = []
            if target_iqr == target_iqr:
                vals.append(target_iqr)
            if loss_iqr == loss_iqr:
                vals.append(loss_iqr)
            pooled_iqr = sum(vals) / len(vals) if vals else float("nan")

        rows.append(
            dict(
                feature=str(feat),
                you=you,
                target_p25=target_p25,
                target_p50=target_p50,
                target_p75=target_p75,
                loss_p25=loss_p25,
                loss_p50=loss_p50,
                loss_p75=loss_p75,
                pooled_iqr=pooled_iqr,
            )
        )
    return rows


def _robust_z(val: float, center: float, scale: float) -> float:
    if not (val == val) or not (center == center) or not (scale == scale):
        return float("nan")
    if abs(scale) < 1e-12:
        return float("nan")
    return (val - center) / scale


def _pretty_num(x: float) -> str:
    if not (x == x):
        return "NaN"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:,.1f}"
    return f"{x:.2f}"


def plot_iqr_z(out_path: Path, rows: List[Dict[str, Any]], title: str, subtitle: str, top_k: int = 10) -> None:
    if not rows:
        raise ValueError("No rows to plot.")

    rows = rows[: max(1, top_k)]
    feats = [r["feature"] for r in rows]
    ys = list(range(len(rows)))

    # Z-transform relative to target median using pooled IQR.
    you_z, loss50_z, target50_z = [], [], []
    loss25_z, loss75_z, target25_z, target75_z = [], [], [], []

    for r in rows:
        center = _safe_float(r.get("target_p50"))
        scale = _safe_float(r.get("pooled_iqr"))

        # If pooled_iqr isn't usable, fall back to target IQR, then loss IQR.
        if not (scale == scale) or abs(scale) < 1e-12:
            ti = _safe_float(r.get("target_p75")) - _safe_float(r.get("target_p25"))
            li = _safe_float(r.get("loss_p75")) - _safe_float(r.get("loss_p25"))
            scale = ti if (ti == ti and abs(ti) > 1e-12) else li

        you_z.append(_robust_z(_safe_float(r.get("you")), center, scale))
        loss50_z.append(_robust_z(_safe_float(r.get("loss_p50")), center, scale))
        target50_z.append(_robust_z(_safe_float(r.get("target_p50")), center, scale))

        loss25_z.append(_robust_z(_safe_float(r.get("loss_p25")), center, scale))
        loss75_z.append(_robust_z(_safe_float(r.get("loss_p75")), center, scale))
        target25_z.append(_robust_z(_safe_float(r.get("target_p25")), center, scale))
        target75_z.append(_robust_z(_safe_float(r.get("target_p75")), center, scale))

    fig_h = max(4.8, 0.55 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    # Slight vertical jitter so LOSS/TARGET IQR bars are distinguishable.
    y_loss = [y - 0.14 for y in ys]
    y_tgt = [y + 0.14 for y in ys]

    # IQR bars
    for i in range(len(rows)):
        if loss25_z[i] == loss25_z[i] and loss75_z[i] == loss75_z[i]:
            ax.hlines(y_loss[i], loss25_z[i], loss75_z[i], linewidth=5, alpha=0.35)
        if target25_z[i] == target25_z[i] and target75_z[i] == target75_z[i]:
            ax.hlines(y_tgt[i], target25_z[i], target75_z[i], linewidth=5, alpha=0.35)

    # Medians and "you"
    ax.scatter(loss50_z, y_loss, marker="o", s=55, alpha=0.95, label="Loss median")
    ax.scatter(target50_z, y_tgt, marker="o", s=55, alpha=0.95, label="Target median")
    ax.scatter(you_z, ys, marker="x", s=110, linewidths=2.2, alpha=0.95, label="You")

    # Reference line at 0 (target median)
    ax.axvline(0.0, linestyle="--", linewidth=1)

    ax.set_yticks(ys)
    ax.set_yticklabels(feats)
    ax.invert_yaxis()

    ax.set_xlabel("Robust z (IQR units) relative to TARGET median (0 = target p50)")
    ax.set_title(title, fontsize=14, pad=10)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, va="bottom")

    # Annotate raw "you" values next to the x marker for context.
    for i, r in enumerate(rows):
        raw_you = _safe_float(r.get("you"))
        if not (you_z[i] == you_z[i]) or not (raw_you == raw_you):
            continue
        ax.annotate(_pretty_num(raw_you), (you_z[i], ys[i]), textcoords="offset points", xytext=(8, -2), fontsize=9)

    # Limits
    all_x = [x for x in (you_z + loss25_z + loss75_z + target25_z + target75_z) if x == x]
    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        pad = 0.15 * (xmax - xmin + 1e-6)
        ax.set_xlim(xmin - pad, xmax + pad)

    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--puuid", default=None)
    ap.add_argument("--minute", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    report_path = _pick_report_path(run_dir, args.match_id, args.minute, args.puuid)
    report = json.load(open(report_path, "r", encoding="utf-8"))
    rows = _extract_rows(report)
    if not rows:
        raise SystemExit(f"No rows in report: {report_path}")

    # Keep output path format consistent with v4.
    role = str(report.get("role", report.get("meta", {}).get("role", "")) or "").upper() or "UNKNOWN"
    champ = str(report.get("champion_id", report.get("meta", {}).get("champion_id", "")) or "champ")

    out_dir = run_dir / "viz" / "match_diagnosis" / f"m{args.minute}" / f"{args.match_id}_{role}_champ{champ}"
    out_path = out_dir / "gaps_dumbbell.png"

    title = f"Match diagnosis gaps @ {args.minute}m — {args.match_id} — {role} — champ {champ}"
    subtitle = "Markers = medians; bars = IQR (p25–p75). X-axis is robust z (IQR units) vs target median."
    plot_iqr_z(out_path, rows, title, subtitle, top_k=args.top_k)
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
