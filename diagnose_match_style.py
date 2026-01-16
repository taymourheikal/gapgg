#!/usr/bin/env python3

"""diagnose_match_style_v6_demo_fixed_v3.py

Goal (demo-friendly, dependency-light):
  Given a (match_id, puuid, minute) row in phase_features, compare the player's early-game
  feature values to two baselines within the *same* (role, champion) cohort:
    - TARGET: WIN (or WIN&CARRY if requested and available)
    - LOSS:   LOSS

This script is designed to be robust across the two phase_features schemas we have used:
  A) long format: explicit columns [match_id, puuid, role, champion_id, minute, ...features]
  B) wide format: one row per participant, and minute-suffixed columns like gold_10, cs_10, ...

Inputs:
  - runs/<run_id>/features/phase_features.csv.gz
  - runs/<run_id>/features/outcomes/outcome_labels_m{minute}.csv.gz

Key fixes vs earlier demo versions:
  - If WIN&CARRY has too few samples for stats, automatically fall back to WIN.
  - Feature selection no longer removes "team"-named columns (e.g., plates_for_team_10).
  - If MWU p-values are unavailable / low-n, we still compute effect sizes and produce
    a "near_misses" section so the UI never looks "empty".

Output:
  - prints a short text report (stdout)
  - writes JSON to runs/<run_id>/reports/diagnose_match_style_<match>_<role>_champ<id>_m<minute>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


# ---------------------------- IO helpers ----------------------------

def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pretty(x) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return "NA"
    if abs(xf) >= 1000:
        return f"{xf:.3g}"
    if abs(xf) >= 10:
        return f"{xf:.2f}"
    return f"{xf:.3f}"


# ---------------------------- Column inference ----------------------------

def _infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


def infer_phase_id_cols(pf: pd.DataFrame) -> Tuple[str, Optional[str], str, str, str, str]:
    """Returns (match_col, minute_col_or_None, puuid_col, role_col, champ_col, schema)."""
    match_col = _infer_col(pf, ["match_id", "matchId", "metadata.matchId"])
    puuid_col = _infer_col(pf, ["puuid"])
    role_col = _infer_col(pf, ["role", "teamPosition", "position"])
    champ_col = _infer_col(pf, ["champion_id", "championId", "champion"])
    minute_col = _infer_col(pf, ["minute", "phase_minute", "t", "phase"])
    schema = "long" if minute_col is not None else "wide"
    missing = [n for n, c in [("match_id", match_col), ("puuid", puuid_col), ("role", role_col), ("champion_id", champ_col)] if c is None]
    if missing:
        raise KeyError(
            f"phase_features missing required columns: {missing}. Have: {list(pf.columns)[:30]} ..."
        )
    return match_col, minute_col, puuid_col, role_col, champ_col, schema


def infer_outcome_cols(out: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
    out_match = _infer_col(out, ["match_id", "matchId", "metadata.matchId"])
    out_puuid = _infer_col(out, ["puuid"])
    out_outcome = _infer_col(out, ["outcome", "win", "result"])
    out_bucket = _infer_col(out, ["agency_bucket", "agencyBucket"])
    missing = [n for n, c in [("match_id", out_match), ("puuid", out_puuid), ("outcome", out_outcome)] if c is None]
    if missing:
        raise KeyError(
            f"outcome_labels missing required columns: {missing}. Have: {list(out.columns)[:30]} ..."
        )
    return out_match, out_puuid, out_outcome, out_bucket


# ---------------------------- Stats helpers ----------------------------

def safe_numeric(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    x = x[np.isfinite(x)]
    return x


def q25_50_75(x: np.ndarray) -> Tuple[float, float, float]:
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    q25, q50, q75 = np.nanpercentile(x, [25, 50, 75])
    return float(q25), float(q50), float(q75)


def iqr(x: np.ndarray) -> float:
    q25, _, q75 = q25_50_75(x)
    if not np.isfinite(q25) or not np.isfinite(q75):
        return float("nan")
    return float(q75 - q25)


def safe_mwu_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    if mannwhitneyu is None:
        return float("nan")
    if x.size < 5 or y.size < 5:
        return float("nan")
    try:
        res = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Approx Cliff's delta with subsampling guard."""
    if x.size == 0 or y.size == 0:
        return float("nan")
    max_n = 1500
    rng = np.random.default_rng(0)
    if x.size > max_n:
        x = rng.choice(x, size=max_n, replace=False)
    if y.size > max_n:
        y = rng.choice(y, size=max_n, replace=False)
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    n = x.size * y.size
    return float((gt - lt) / n) if n > 0 else float("nan")


# ---------------------------- Feature selection ----------------------------

EXCLUDE_EXACT = {
    "match_id",
    "matchid",
    "puuid",
    "summonername",
    "summoner_id",
    "participant",
    "participantid",
}

EXCLUDE_SUBSTR = [
    "accountid",
    "gameid",
    "platform",
    "queue",
    "tier",
    "division",
    "cluster",
    "outcome",
    "win",
    "result",
    "agency",
]


def infer_feature_cols(pf: pd.DataFrame, schema: str, minute: int) -> List[str]:
    """Return numeric feature columns for the requested minute."""
    if schema == "long":
        # Long schema: features are not minute-suffixed; we will just take numeric columns
        # (minus id-like and outcome-like fields).
        feats: List[str] = []
        for c in pf.columns:
            lc = str(c).lower()
            if lc in EXCLUDE_EXACT:
                continue
            if any(s in lc for s in EXCLUDE_SUBSTR):
                continue
            # avoid obvious ids
            if lc.endswith("_id"):
                continue
            feats.append(c)
        return sorted(set(feats))

    suf = f"_{minute}"
    feats = []
    for c in pf.columns:
        sc = str(c)
        if not sc.endswith(suf):
            continue
        lc = sc.lower()
        if any(s in lc for s in EXCLUDE_SUBSTR):
            continue
        # drop pure advantages by heuristic (keep deltas)
        if "adv" in lc and "delta" not in lc:
            continue
        feats.append(sc)
    return sorted(set(feats))


# ---------------------------- Main logic ----------------------------

def find_participant_row(
    pf: pd.DataFrame,
    schema: str,
    match_col: str,
    minute_col: Optional[str],
    puuid_col: str,
    minute: int,
    match_id: str,
    puuid: str,
) -> pd.Series:
    m = (pf[match_col].astype(str) == str(match_id)) & (pf[puuid_col].astype(str) == str(puuid))
    if schema == "long" and minute_col is not None:
        mm = pf[minute_col]
        if pd.api.types.is_numeric_dtype(mm):
            m = m & (mm.astype(int) == int(minute))
        else:
            m = m & (mm.astype(str) == str(minute))
    sub = pf[m]
    if len(sub) == 0:
        raise RuntimeError("Participant row not found for match + puuid (and minute if long schema)")
    return sub.iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--minute", type=int, required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--puuid", required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--use-win-carrier", action="store_true")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--min-abs-cliff", type=float, default=0.05)
    ap.add_argument("--min-iqr", type=float, default=1e-9)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--near-k", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    minute = int(args.minute)

    pf_path = run_dir / "features" / "phase_features.csv.gz"
    out_path = run_dir / "features" / "outcomes" / f"outcome_labels_m{minute}.csv.gz"
    report_dir = run_dir / "reports"
    ensure_dir(report_dir)

    pf = read_csv_any(pf_path)
    out = read_csv_any(out_path)

    match_col, minute_col, puuid_col, role_col, champ_col, schema = infer_phase_id_cols(pf)
    out_match_col, out_puuid_col, out_outcome_col, out_bucket_col = infer_outcome_cols(out)

    prow = find_participant_row(pf, schema, match_col, minute_col, puuid_col, minute, args.match_id, args.puuid)
    role = str(prow[role_col])
    champ = int(prow[champ_col])

    # Join outcomes onto phase_features keys (for wide schema there is no minute column)
    out2 = out.copy()
    if out_match_col != match_col:
        out2 = out2.rename(columns={out_match_col: match_col})
    if out_puuid_col != puuid_col:
        out2 = out2.rename(columns={out_puuid_col: puuid_col})
    if out_outcome_col != "outcome":
        out2 = out2.rename(columns={out_outcome_col: "outcome"})
    if out_bucket_col is not None and out_bucket_col != "agency_bucket":
        out2 = out2.rename(columns={out_bucket_col: "agency_bucket"})

    # Keep only rows in the same champ+role cohort
    cohort = pf[(pf[role_col].astype(str) == role) & (pd.to_numeric(pf[champ_col], errors="coerce") == champ)].copy()
    cohort = cohort.merge(out2[[match_col, puuid_col, "outcome"] + (["agency_bucket"] if "agency_bucket" in out2.columns else [])],
                          on=[match_col, puuid_col], how="left")

    # Player outcome
    oo = out2[(out2[match_col].astype(str) == str(args.match_id)) & (out2[puuid_col].astype(str) == str(args.puuid))]
    outcome_val = int(pd.to_numeric(oo.iloc[0]["outcome"], errors="coerce")) if len(oo) else np.nan
    outcome = "WIN" if outcome_val == 1 else "LOSS"
    agency_bucket = str(oo.iloc[0]["agency_bucket"]) if len(oo) and "agency_bucket" in oo.columns else "unknown"

    # Target group selection with fallback
    want_win_carrier = bool(args.use_win_carrier)
    target_label = "WIN&CARRIER" if want_win_carrier else "WIN"

    loss_df = cohort[cohort["outcome"].astype(float) == 0.0]
    win_df = cohort[cohort["outcome"].astype(float) == 1.0]

    if want_win_carrier and "agency_bucket" in cohort.columns:
        win_car_df = win_df[win_df["agency_bucket"].astype(str).str.lower().isin(["carrying", "carrier", "carry"])]
        # If too small, fall back to WIN.
        if len(win_car_df) >= max(args.min_samples, 5):
            target_df = win_car_df
        else:
            target_df = win_df
            target_label = "WIN (fallback; WIN&CARRIER too small)"
    else:
        target_df = win_df

    # Features
    feat_cols = infer_feature_cols(pf, schema, minute)

    # For wide schema: build player's minute feature values from suffix columns.
    if schema == "wide":
        you_vals: Dict[str, float] = {}
        for c in feat_cols:
            you_vals[c] = float(pd.to_numeric(prow.get(c, np.nan), errors="coerce"))
    else:
        you_vals = {c: float(pd.to_numeric(prow.get(c, np.nan), errors="coerce")) for c in feat_cols}

    gaps: List[Dict] = []
    near: List[Dict] = []

    drop_counts = {
        "total_features": len(feat_cols),
        "you_missing": 0,
        "low_samples": 0,
        "degenerate_iqr": 0,
        "nan_effect": 0,
        "nan_p": 0,
        "filtered_alpha": 0,
        "filtered_cliff": 0,
    }

    for f in feat_cols:
        you = you_vals.get(f, float("nan"))
        if not np.isfinite(you):
            drop_counts["you_missing"] += 1
            continue

        x = safe_numeric(target_df[f])
        y = safe_numeric(loss_df[f])
        if x.size < 2 or y.size < 2:
            drop_counts["low_samples"] += 1
            continue

        tq25, tp50, tq75 = q25_50_75(x)
        lq25, lp50, lq75 = q25_50_75(y)
        tiqr = tq75 - tq25 if np.isfinite(tq25) and np.isfinite(tq75) else float("nan")
        liqr = lq75 - lq25 if np.isfinite(lq25) and np.isfinite(lq75) else float("nan")

        if (not np.isfinite(tiqr)) or (not np.isfinite(liqr)) or (max(tiqr, liqr) < args.min_iqr):
            drop_counts["degenerate_iqr"] += 1
            continue

        cliff = cliffs_delta(x, y)
        if not np.isfinite(cliff):
            drop_counts["nan_effect"] += 1
            continue

        p = safe_mwu_pvalue(x, y)
        if not np.isfinite(p):
            # We still keep a near-miss candidate, but we won't include it as "supported".
            drop_counts["nan_p"] += 1

        row = {
            "feature": str(f),
            "you": float(you),
            "target_p25": float(tq25),
            "target_p50": float(tp50),
            "target_p75": float(tq75),
            "loss_p25": float(lq25),
            "loss_p50": float(lp50),
            "loss_p75": float(lq75),
            "p_value": float(p) if np.isfinite(p) else None,
            "cliffs_delta": float(cliff),
            "n_target": int(x.size),
            "n_loss": int(y.size),
            "target_iqr": float(tiqr),
            "loss_iqr": float(liqr),
        }

        # Supported gates
        supported = True
        if np.isfinite(p) and p > args.alpha:
            supported = False
            drop_counts["filtered_alpha"] += 1
        if abs(cliff) < args.min_abs_cliff:
            supported = False
            drop_counts["filtered_cliff"] += 1

        if supported:
            gaps.append(row)
        else:
            near.append(row)

    # Rank by |cliff| then by p (if present)
    def _rank_key(r: Dict) -> Tuple[float, float]:
        p = r.get("p_value")
        p_rank = float(p) if (p is not None and np.isfinite(p)) else 1.0
        return (-abs(float(r.get("cliffs_delta", 0.0))), p_rank)

    gaps = sorted(gaps, key=_rank_key)[: args.topk]
    near = sorted(near, key=_rank_key)[: args.near_k]

    report = {
        "match_id": str(args.match_id),
        "puuid": str(args.puuid),
        "minute": int(minute),
        "role": role,
        "champion_id": int(champ),
        "outcome": outcome,
        "agency_bucket": agency_bucket,
        "target_group": target_label,
        "config": {
            "alpha": float(args.alpha),
            "min_samples": int(args.min_samples),
            "min_abs_cliff": float(args.min_abs_cliff),
            "min_iqr": float(args.min_iqr),
            "use_win_carrier": bool(args.use_win_carrier),
        },
        "cohort_sizes": {
            "n_total": int(len(cohort)),
            "n_win": int(len(win_df)),
            "n_loss": int(len(loss_df)),
            "n_target": int(len(target_df)),
        },
        "drop_counts": drop_counts,
        "gaps": gaps,
        "near_misses": near,
    }

    # Text output
    print("\n=== Match Diagnosis (signal-gated; demo) ===")
    print(f"Match: {args.match_id}")
    print(f"Role: {role} | Champion: {champ}")
    print(f"Outcome: {outcome} | Agency bucket: {agency_bucket} | Target group: {target_label}")
    if not gaps:
        # show near misses so the user always sees *something* actionable-looking
        print("\nNo statistically supported gaps for this matchup + configuration.")
        if near:
            print("\nTop near-misses (ranked by effect size; may be low-n or p>alpha):")
            for i, g in enumerate(near[: args.near_k], 1):
                print(
                    f"{i}. {g['feature']}: you={_pretty(g['you'])} | target p50={_pretty(g['target_p50'])} | loss p50={_pretty(g['loss_p50'])} "
                    f"(p={_pretty(g.get('p_value'))}, cliff={_pretty(g.get('cliffs_delta'))}, nT={g['n_target']}, nL={g['n_loss']})"
                )
    else:
        print("\nTop behavior gaps:")
        for i, g in enumerate(gaps, 1):
            print(
                f"{i}. {g['feature']}: you={_pretty(g['you'])} | target p50={_pretty(g['target_p50'])} | loss p50={_pretty(g['loss_p50'])}"
            )
            print(
                f"   evidence: p={_pretty(g.get('p_value'))}, cliff={_pretty(g.get('cliffs_delta'))}, IQR(T)={_pretty(g.get('target_iqr'))}, nT={g['n_target']}, nL={g['n_loss']}"
            )

    out_json = report_dir / f"diagnose_match_style_{args.match_id}_{role}_champ{champ}_m{minute}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ok] Wrote: {out_json}")


if __name__ == "__main__":
    main()
