#!/usr/bin/env python3
"""demo_dashboard_v4.py

Single-file Flask demo dashboard.

Fixes vs v3:
- Builds match dropdown from outcome_labels (has role/champion_id) instead of phase_features.
- Merges outcome_labels with phase_features on (match_id, puuid) for minute-wide features.
- Robustly parses playstyles enrichment JSON to derive demo-able (role, champion_id) pairs.
- Fixes /what_if payload mismatch (accepts frac or pct) and JS now sends frac.
- Avoids hard dependency on phase_features having 'role'/'championId' columns.

Run:
  python demo_dashboard_v4.py --run-dir runs/<RUN_ID> --minute 10
Then open:
  http://127.0.0.1:5055
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template_string, request, send_file, url_for

APP = Flask(__name__)

# ---------- Helpers ----------

ROLE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        s = str(x).strip()
        if s == "":
            return None
        # handle strings like "champ21" or "21"
        m = re.search(r"(-?\d+)", s)
        if not m:
            return None
        return int(m.group(1))
    except Exception:
        return None


def pick(cols: Sequence[str], options: Sequence[str]) -> Optional[str]:
    for o in options:
        if o in cols:
            return o
    return None


def infer_phase_key_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """phase_features only needs (match, puuid)."""
    match_col = pick(df.columns, ["match_id", "matchId", "metadata.matchId"])
    puuid_col = pick(df.columns, ["puuid"])
    if not match_col or not puuid_col:
        raise KeyError(
            f"phase_features missing required key cols. Need match_id+puuid. Have: {list(df.columns)[:30]} ..."
        )
    return match_col, puuid_col


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> str:
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\n\n{r.stdout}")
    return r.stdout


_FULL_RUN_RE = re.compile(r"^\d{8}T\d{6}Z_[A-Z0-9]+_[A-Z]+_[IVX]+_q\d+_\d+d$")


def find_latest_run_dir(root: Path, minute: int) -> Optional[Path]:
    """Pick a sensible default run directory for the demo.

    Critical behavior (per your earlier requirement):
    - Prefer the latest *non-player* full pipeline run (tier/div/queue format).
    - Only pick runs that actually have the required dependency files.
    """
    if not root.exists():
        return None

    required_rel = [
        Path("features/phase_features.csv.gz"),
        Path(f"features/outcomes/outcome_labels_m{minute}.csv.gz"),
        Path(f"features/playstyles/enrichment_m{minute}.json"),
    ]

    def has_required(p: Path) -> bool:
        return all((p / rr).exists() for rr in required_rel)

    # First: full runs (exclude PLAYER) that pass dependency checks
    full_runs = [p for p in root.iterdir() if p.is_dir() and _FULL_RUN_RE.match(p.name) and has_required(p)]
    if full_runs:
        full_runs.sort(key=lambda p: p.stat().st_mtime)
        return full_runs[-1]

    # Fallback: any non-player run that passes dependency checks
    non_player = [p for p in root.iterdir() if p.is_dir() and "_PLAYER_" not in p.name and has_required(p)]
    if non_player:
        non_player.sort(key=lambda p: p.stat().st_mtime)
        return non_player[-1]

    # Last resort: anything at all
    any_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not any_dirs:
        return None
    any_dirs.sort(key=lambda p: p.stat().st_mtime)
    return any_dirs[-1]


# ---------- Cache ----------

@dataclass
class MatchRow:
    match_id: str
    puuid: str
    role: str
    champion_id: int
    win: int
    agency_bucket: str
    outcome_bucket: str


class DemoCache:
    def __init__(self, run_dir: Path, minute: int):
        self.run_dir = run_dir
        self.minute = minute

        # Load phase features (wide)
        self.phase_path = run_dir / "features" / "phase_features.csv.gz"
        if not self.phase_path.exists():
            raise FileNotFoundError(f"Missing: {self.phase_path}")
        self.pf = pd.read_csv(self.phase_path)
        self.pf_match_col, self.pf_puuid_col = infer_phase_key_cols(self.pf)

        # Ensure the requested minute exists (wide columns ending _{minute})
        self.minute_cols = [c for c in self.pf.columns if c.endswith(f"_{minute}")]
        if not self.minute_cols:
            raise RuntimeError(
                f"phase_features has no columns ending with _{minute}."
            )

        # Load outcome labels
        self.ol_path = run_dir / "features" / "outcomes" / f"outcome_labels_m{minute}.csv.gz"
        if not self.ol_path.exists():
            raise FileNotFoundError(f"Missing: {self.ol_path}")
        self.ol = pd.read_csv(self.ol_path)

        # Column names for ol
        self.ol_match_col = pick(self.ol.columns, ["match_id", "matchId"]) or "match_id"
        self.ol_puuid_col = pick(self.ol.columns, ["puuid"]) or "puuid"
        self.ol_role_col = pick(self.ol.columns, ["role", "teamPosition", "position"]) or "role"
        self.ol_champ_col = pick(self.ol.columns, ["champion_id", "championId", "championId"]) or "champion_id"
        self.ol_win_col = pick(self.ol.columns, ["win", "outcome", "label", "y"]) or "win"
        self.ol_agency_col = pick(self.ol.columns, ["agency_bucket", "agency", "agencyBucket"]) or "agency_bucket"
        self.ol_outcome_bucket_col = pick(self.ol.columns, ["outcome_bucket", "bucket", "outcomeBucket"]) or "outcome_bucket"

        for req in [self.ol_match_col, self.ol_puuid_col, self.ol_role_col, self.ol_champ_col, self.ol_win_col]:
            if req not in self.ol.columns:
                raise KeyError(f"outcome_labels missing required col '{req}'. Have: {list(self.ol.columns)[:30]} ...")

        # Build merged frame (ol + pf keys) to ensure we only include rows present in pf.
        pf_keys = self.pf[[self.pf_match_col, self.pf_puuid_col]].dropna().drop_duplicates()
        pf_keys = pf_keys.rename(columns={self.pf_match_col: "match_id", self.pf_puuid_col: "puuid"})

        ol_keys = self.ol[[self.ol_match_col, self.ol_puuid_col, self.ol_role_col, self.ol_champ_col, self.ol_win_col,
                           self.ol_agency_col, self.ol_outcome_bucket_col]].copy()
        ol_keys = ol_keys.rename(columns={
            self.ol_match_col: "match_id",
            self.ol_puuid_col: "puuid",
            self.ol_role_col: "role",
            self.ol_champ_col: "champion_id",
            self.ol_win_col: "win",
            self.ol_agency_col: "agency_bucket",
            self.ol_outcome_bucket_col: "outcome_bucket",
        })

        self.ol_pf = ol_keys.merge(pf_keys, on=["match_id", "puuid"], how="inner")

        # Build an index for role/champ lookup
        self.key_to_role_champ: Dict[Tuple[str, str], Tuple[str, int]] = {}
        for r in self.ol_pf.itertuples(index=False):
            try:
                self.key_to_role_champ[(str(r.match_id), str(r.puuid))] = (str(r.role), int(r.champion_id))
            except Exception:
                continue

        # Load enrichment JSON (optional but used to choose "clustered" champ/role pairs)
        self.enrichment_path = run_dir / "features" / "playstyles" / f"enrichment_m{minute}.json"
        self.enrich: Optional[dict] = None
        if self.enrichment_path.exists():
            self.enrich = json.load(open(self.enrichment_path, "r", encoding="utf-8"))

        self.demo_pairs = self._extract_demo_pairs(self.enrich)

        self.match_rows = self._build_match_rows()
        self.match_options = self._build_match_options(limit=250)

        # requested early-warning features (keep aligned with early_warning_model.py)
        self.ew_features = [
            "gold_10",
            "cs_10",
            "xp_10",
            "gold_delta_10",
            "cs_delta_10",
            "xp_delta_10",
            "kills_10",
            "assists_10",
            "deaths_10",
            "kp_10",
            "first_shop_time_s",
            "first_death_time_s",
            "dragon_kills_10",
            "herald_kills_10",
            "towers_destroyed_10",
            "plates_for_team_10",
            "plates_against_10",
            "plates_diff_10",
        ]

    # ----- demo pair extraction -----
    def _extract_demo_pairs(self, enrich: Optional[dict]) -> List[Tuple[str, int]]:
        """Return list of (ROLE, champion_id) known to be clustered.

        Attempts to be robust to multiple enrichment JSON layouts.
        """
        if not enrich or not isinstance(enrich, dict):
            return []

        # common containers
        groups = None
        for k in ("groups", "by_group", "enrichment", "clusters", "results"):
            if k in enrich and isinstance(enrich[k], list):
                groups = enrich[k]
                break
        if groups is None and "groups" in enrich and isinstance(enrich["groups"], dict):
            groups = list(enrich["groups"].values())
        if groups is None:
            # sometimes it is directly a list
            if isinstance(enrich, list):
                groups = enrich

        out: List[Tuple[str, int]] = []
        if not groups:
            return out

        for g in groups:
            if not isinstance(g, dict):
                continue
            # try explicit fields
            role = g.get("role") or g.get("teamPosition")
            champ = g.get("champion_id") or g.get("championId") or g.get("champ")

            # or parse group id like "BOTTOM_21" or "m10_BOTTOM_21" etc.
            gid = g.get("group") or g.get("group_id") or g.get("id") or g.get("key")
            if (role is None or champ is None) and isinstance(gid, str):
                m = re.search(r"(TOP|JUNGLE|MIDDLE|BOTTOM|UTILITY)[^0-9]*([0-9]+)", gid)
                if m:
                    role = role or m.group(1)
                    champ = champ or m.group(2)

            role_s = str(role).upper() if role is not None else None
            champ_i = safe_int(champ)
            if role_s in ROLE_ORDER and champ_i is not None:
                out.append((role_s, champ_i))

        # de-dup preserving order
        seen = set()
        uniq = []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    # ----- match rows / dropdown -----
    def _build_match_rows(self) -> List[MatchRow]:
        df = self.ol_pf.copy()
        # normalize
        df["role"] = df["role"].astype(str).str.upper()
        df["champion_id"] = df["champion_id"].apply(safe_int)
        df = df.dropna(subset=["match_id", "puuid", "role", "champion_id"])
        df["champion_id"] = df["champion_id"].astype(int)

        if self.demo_pairs:
            allowed = set(self.demo_pairs)
            df = df[df.apply(lambda r: (r["role"], int(r["champion_id"])) in allowed, axis=1)]

        # keep most recent-ish matches first by sorting match_id numeric tail if possible
        def match_sort_key(mid: str) -> int:
            m = re.search(r"_(\d+)$", str(mid))
            return int(m.group(1)) if m else 0

        df = df.copy()
        df["_k"] = df["match_id"].map(match_sort_key)
        df = df.sort_values(["_k"], ascending=False)

        rows: List[MatchRow] = []
        for r in df.itertuples(index=False):
            rows.append(
                MatchRow(
                    match_id=str(r.match_id),
                    puuid=str(r.puuid),
                    role=str(r.role),
                    champion_id=int(r.champion_id),
                    win=int(r.win) if str(r.win).strip() != "" else 0,
                    agency_bucket=str(getattr(r, "agency_bucket", "")),
                    outcome_bucket=str(getattr(r, "outcome_bucket", "")),
                )
            )
        return rows

    def _build_match_options(self, limit: int = 250) -> List[Dict[str, str]]:
        opts: List[Dict[str, str]] = []
        for r in self.match_rows[:limit]:
            label = f"{r.match_id} | {r.role} | champ {r.champion_id} | {'WIN' if r.win==1 else 'LOSS'} | agency {r.agency_bucket}"
            val = f"{r.match_id}::{r.puuid}::{r.role}::{r.champion_id}"
            opts.append({"label": label, "value": val})
        return opts

    # ----- data access -----
    def get_row_context(self, match_id: str, puuid: str) -> Dict[str, Any]:
        """Return role/champ/win/agency/outcome_bucket (from ol_pf)."""
        sub = self.ol_pf[(self.ol_pf["match_id"] == match_id) & (self.ol_pf["puuid"] == puuid)]
        if sub.empty:
            return {}
        r = sub.iloc[0].to_dict()
        return {
            "role": str(r.get("role", "")).upper(),
            "champion_id": safe_int(r.get("champion_id")),
            "win": int(r.get("win", 0)) if str(r.get("win", "")).strip() != "" else 0,
            "agency_bucket": str(r.get("agency_bucket", "")),
            "outcome_bucket": str(r.get("outcome_bucket", "")),
        }

    def get_pf_row(self, match_id: str, puuid: str) -> Optional[pd.Series]:
        sub = self.pf[(self.pf[self.pf_match_col] == match_id) & (self.pf[self.pf_puuid_col] == puuid)]
        if sub.empty:
            return None
        return sub.iloc[0]

    # ----- p(win) model utilities -----
    def model_path_for_role(self, role: str) -> Path:
        return self.run_dir / "models" / f"early_warning_m{self.minute}_{role}.pkl"

    
    def compute_pwin(self, role: str, pf_row: pd.Series):
        """Compute P(win) using the early-warning model for this role.

        Supports both:
          - sklearn Pipeline / estimator
          - dict bundle from early_warning_model.py: {features, scaler, model}
        """
        pkl = self.run_dir / "models" / f"early_warning_m{self.minute}_{role}.pkl"
        if not pkl.exists():
            return None, f"Missing model for role={role}: {pkl}"

        import joblib
        model_obj = joblib.load(pkl)

        # Prefer feature list from bundle if present; otherwise fall back to ew_features
        feats = None
        if isinstance(model_obj, dict):
            feats = model_obj.get("features")
        feats = list(feats) if feats else list(self.ew_features)

        row = {}
        for f in feats:
            row[f] = float(pf_row.get(f, np.nan))
        X_df = pd.DataFrame([row], columns=feats)

        if X_df.isna().any(axis=None):
            missing = [c for c in feats if pd.isna(X_df.at[0, c])]
            return None, f"NaN values for model features: {missing[:10]}{'...' if len(missing)>10 else ''}"

        try:
            if hasattr(model_obj, "predict_proba"):
                p = model_obj.predict_proba(X_df)
                return float(p[0, 1]), None

            if isinstance(model_obj, dict):
                scaler = model_obj.get("scaler")
                mdl = model_obj.get("model") or model_obj.get("clf") or model_obj.get("pipeline")
                if mdl is None:
                    return None, "Model bundle missing 'model'"
                X = X_df.values
                if scaler is not None:
                    X = scaler.transform(X)
                if not hasattr(mdl, "predict_proba"):
                    return None, "Underlying model has no predict_proba"
                p = mdl.predict_proba(X)
                return float(p[0, 1]), None

            return None, f"Unsupported model type: {type(model_obj)}"
        except Exception as e:
            return None, f"predict_proba failed: {e}"

    def _find_diag_json_for_match(self, match_id: str, puuid: str) -> Optional[Path]:
        """Locate a diagnosis JSON for a specific match+puuid at the configured minute.

        The file name encodes role/champ, but match_id is always present.
        """
        pat = str(self.run_dir / "reports" / f"diagnose_match_style_{match_id}_*_m{self.minute}.json")
        cands = sorted(glob.glob(pat))
        if not cands:
            # also support older naming variants if present
            pat2 = str(self.run_dir / "reports" / f"diagnose_match_style_{match_id}_m{self.minute}*.json")
            cands = sorted(glob.glob(pat2))
        return Path(cands[-1]) if cands else None

    def _target_median_for(self, match_id: str, puuid: str, role: str, feature: str, use_win_carrier: bool) -> Tuple[Optional[float], str]:
        """Get a WIN (or WIN&CARRIER) target median for `feature`.

        Prefer the per-match diagnosis JSON (which already computed the target median);
        fallback to computing it directly from outcome_labels + phase_features.
        """
        # 1) Prefer per-match diagnosis JSON (already aligned to the same target group)
        jp = self._find_diag_json_for_match(match_id, puuid)
        if jp and jp.exists():
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                gaps = d.get("gaps") if isinstance(d, dict) else None
                if isinstance(gaps, list):
                    for g in gaps:
                        if isinstance(g, dict) and g.get("feature") == feature and "target_median" in g:
                            tm = g.get("target_median")
                            if tm is not None and np.isfinite(float(tm)):
                                return float(tm), "from_diagnosis_json"
            except Exception:
                pass

        # 2) Fallback: compute from dataset for this role/champ
        if feature not in self.pf.columns:
            return None, f"feature_missing_in_phase_features: {feature}"

        # identify this player's champ_id (we use it to keep medians champion-specific)
        try:
            my = self.ol_pf[(self.ol_pf["match_id"] == match_id) & (self.ol_pf["puuid"] == puuid)].iloc[0]
            champ_id = int(my.get("champion_id")) if "champion_id" in my else int(my.get("championId"))
        except Exception:
            champ_id = None

        ol = self.ol_pf
        mask = (ol["role"].astype(str) == str(role)) & (ol["win"] == 1)
        if champ_id is not None and "champion_id" in ol.columns:
            mask = mask & (ol["champion_id"] == champ_id)
        if use_win_carrier and "agency_bucket" in ol.columns:
            mask = mask & (ol["agency_bucket"].astype(str) == "carrier")

        ids = ol.loc[mask, ["match_id", "puuid"]]
        if ids.empty:
            return None, "target_group_empty"

        pf_sub = self.pf[["match_id", "puuid", feature]].merge(ids.drop_duplicates(), on=["match_id", "puuid"], how="inner")
        s = pd.to_numeric(pf_sub[feature], errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            return None, "target_group_feature_all_nan"
        return float(s.median()), "computed_from_dataset"

    
    def _get_target_median(self, match_id: str, puuid: str, role: str, feature: str, use_win_carrier: bool) -> Tuple[Optional[float], str]:
        """Return the target median used by the what-if slider.

        Preference order:
        1) Use target_median from the most recent diagnose_match_style report JSON for this match (if present).
        2) Compute from this run's cohort data (same role+champ), using WIN or WIN&CARRIER depending on config.
        """
        # 1) Try diagnosis JSON (fast + consistent with the "gaps" card).
        try:
            rc = self.key_to_role_champ.get((str(match_id), str(puuid)))
            if rc:
                r_role, r_champ = rc
                diag = self._find_diag_json_for_match(str(match_id), str(r_role), int(r_champ), int(self.minute))
                if diag and diag.exists():
                    import json
                    d = json.load(open(diag, "r", encoding="utf-8"))
                    gaps = d.get("gaps") or []
                    for g in gaps:
                        if isinstance(g, dict) and g.get("feature") == feature:
                            tm = g.get("target_median", None)
                            if tm is not None:
                                return float(tm), f"from diagnosis report ({diag.name})"
        except Exception:
            pass

        # 2) Compute from cohort data.
        # Find player's champ + agency bucket from outcome_labels.
        prow = None
        try:
            m = (self.ol_pf["match_id"].astype(str) == str(match_id)) & (self.ol_pf["puuid"].astype(str) == str(puuid))
            sub = self.ol_pf[m]
            if len(sub) > 0:
                prow = sub.iloc[0]
        except Exception:
            prow = None
        if prow is None:
            return None, "player row not found in outcome_labels"

        champ_id = int(prow["champion_id"])
        agency_bucket = str(prow.get("agency_bucket", "")).lower().strip()

        # Target cohort filter:
        # - default: WIN
        # - use_win_carrier: if player was not carried, restrict to WIN & carrier cohort; if carried, use WIN cohort.
        cohort = self.ol_pf[(self.ol_pf["role"].astype(str) == str(role)) & (self.ol_pf["champion_id"].astype(int) == champ_id)]
        if use_win_carrier:
            if agency_bucket in ("carried", "carryed", "carried?"):
                cohort = cohort[cohort["win"].astype(int) == 1]
                cohort_name = "WIN"
            else:
                cohort = cohort[(cohort["win"].astype(int) == 1) & (cohort["agency_bucket"].astype(str).str.lower().isin(["carrier", "carry"]))]
                cohort_name = "WIN&CARRIER"
                if len(cohort) == 0:
                    # fallback to plain WIN if carrier cohort is too small / empty
                    cohort = self.ol_pf[(self.ol_pf["role"].astype(str) == str(role)) & (self.ol_pf["champion_id"].astype(int) == champ_id) & (self.ol_pf["win"].astype(int) == 1)]
                    cohort_name = "WIN (fallback)"
        else:
            cohort = cohort[cohort["win"].astype(int) == 1]
            cohort_name = "WIN"

        keys = cohort[["match_id", "puuid"]].drop_duplicates()
        if len(keys) == 0:
            return None, f"no cohort rows for {cohort_name}"

        # Filter phase_features to cohort keys and take median on the requested feature column.
        if feature not in self.pf.columns:
            return None, f"feature not present in phase_features: {feature}"

        pf_small = self.pf[[self.pf_match_col, self.pf_puuid_col, feature]].copy()
        pf_small = pf_small.rename(columns={self.pf_match_col: "match_id", self.pf_puuid_col: "puuid"})
        merged = keys.merge(pf_small, on=["match_id", "puuid"], how="inner")
        if len(merged) == 0:
            return None, f"no phase_features rows for cohort ({cohort_name})"

        vals = merged[feature].dropna()
        if len(vals) == 0:
            return None, f"all NaN for feature in cohort ({cohort_name})"

        return float(vals.median()), f"computed from cohort {cohort_name} (n={len(vals)})"

    def what_if(self, role: str, pf_row: pd.Series, match_id: str, puuid: str, feature: str, frac: float, use_win_carrier: bool):
        """What-if: move one feature toward the WIN (or WIN&CARRIER) median and recompute p(win).

        frac in [0,1] means "how far you move from your current value to the target median".
        """
        pkl = self.run_dir / "models" / f"early_warning_m{self.minute}_{role}.pkl"
        if not pkl.exists():
            return None, f"Missing model for role={role}"

        import joblib
        model_obj = joblib.load(pkl)

        feats = None
        if isinstance(model_obj, dict):
            feats = model_obj.get("features")
        feats = list(feats) if feats else list(self.ew_features)

        target_median, how = self._get_target_median(match_id, puuid, role, feature, use_win_carrier)
        if target_median is None or not np.isfinite(target_median):
            return None, f"No target median for feature={feature}"

        row = {}
        for f in feats:
            row[f] = float(pf_row.get(f, np.nan))

        base = row.get(feature, np.nan)
        if not np.isfinite(base):
            return None, f"Selected feature is NaN: {feature}"

        row[feature] = float(base + frac * (target_median - base))
        X_df = pd.DataFrame([row], columns=feats)

        if X_df.isna().any(axis=None):
            missing = [c for c in feats if pd.isna(X_df.at[0, c])]
            return None, f"NaN values for what-if model features: {missing[:10]}{'...' if len(missing)>10 else ''}"

        try:
            if hasattr(model_obj, "predict_proba"):
                p = model_obj.predict_proba(X_df)
                return float(p[0, 1]), None

            if isinstance(model_obj, dict):
                scaler = model_obj.get("scaler")
                mdl = model_obj.get("model") or model_obj.get("clf") or model_obj.get("pipeline")
                if mdl is None:
                    return None, "Model bundle missing 'model'"
                X = X_df.values
                if scaler is not None:
                    X = scaler.transform(X)
                if not hasattr(mdl, "predict_proba"):
                    return None, "Underlying model has no predict_proba"
                p = mdl.predict_proba(X)
                return float(p[0, 1]), None

            return None, f"Unsupported model type: {type(model_obj)}"
        except Exception as e:
            return None, f"predict_proba failed: {e}"


# ---------- Global cache ----------

CACHE: Optional[DemoCache] = None


def get_cache() -> DemoCache:
    global CACHE

    # defaults
    default_run = Path("runs")
    run_dir = Path(os.environ.get("RUN_DIR", "")) if os.environ.get("RUN_DIR") else None
    minute = int(os.environ.get("MINUTE", "10"))

    # CLI overrides are stored on APP.config
    if "RUN_DIR" in APP.config:
        run_dir = Path(APP.config["RUN_DIR"])
    if "MINUTE" in APP.config:
        minute = int(APP.config["MINUTE"])

    if run_dir is None:
        run_dir = find_latest_run_dir(default_run, minute)

    if run_dir is None:
        raise RuntimeError("Could not locate a run dir under ./runs")

    if CACHE is None or CACHE.run_dir != run_dir or CACHE.minute != minute:
        CACHE = DemoCache(run_dir=run_dir, minute=minute)
    return CACHE


# ---------- UI Template ----------

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LeagueAnalytics Demo Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    .row { display:flex; gap:16px; flex-wrap:wrap; }
    .card { border:1px solid #ddd; border-radius:12px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.06); background:#fff; }
    .card h3 { margin:0 0 8px 0; font-size:16px; }
    .muted { color:#666; font-size:13px; }
    select, button, input[type=range] { font-size:14px; }
    pre { white-space: pre-wrap; word-wrap: break-word; background:#fafafa; border:1px solid #eee; padding:12px; border-radius:10px; }
    .kpi { font-size:34px; font-weight:700; }
    .kpi small { font-size:13px; font-weight:500; color:#666; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 8px; text-align:left; font-size: 13px; }
    th { background:#fafafa; position: sticky; top:0; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #ddd; font-size:12px; }
  </style>
</head>
<body>
  <h2>LeagueAnalytics — Riot API Demo Dashboard</h2>
  <div class="muted">Run: <b>{{ run_dir }}</b> | Minute: <b>{{ minute }}</b> | Matches in dropdown: <b>{{ match_count }}</b></div>

  <div class="card" style="margin-top:12px;">    <h3>Note to Riot</h3>    <div class="small">      <p><b>What we’re building:</b> an explainable, match-by-match coaching layer that answers <i>“why did I win/lose?”</i> using Riot match data, role-aware baselines, and lightweight ML.</p>      <p><b>This demo includes:</b> (1) a match selector, (2) a statistically-gated “gap” diagnosis vs a relevant target cohort, (3) a visual gap plot, and (4) an early-warning <b>P(win)</b> estimate at minute {{ minute }} with a simple “what-if” control.</p>      <p><b>Where we’re going next:</b> richer context (enemy comp + lane matchup + itemization), timelines (minute-by-minute deltas), and actionable recommendations that map each gap to concrete in-game habits (tempo, wave states, objective setup).</p>      <p class="muted">Everything shown here is computed from the demo run directory; no external services are required to view results.</p>    </div>  </div>


  {% if match_count == 0 %}
    <div class="card" style="margin-top:16px;">
      <h3>No demo matches found</h3>
      <div class="muted">This usually means the dashboard couldn't find any (role,champion) pairs from enrichment, or outcome labels are empty.</div>
    </div>
  {% endif %}

  <div class="card" style="margin-top:16px;">
    <h3>Select a match</h3>
    <div class="row">
      <div>
        <select id="matchSelect" style="min-width:520px;">
          {% for o in options %}
            <option value="{{ o.value }}">{{ o.label }}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <button id="btnGenerate">Generate</button>
      </div>
    </div>
    <div class="muted" style="margin-top:8px;">The dropdown is pre-filtered to champions/roles that were clustered in this run (from enrichment_m{{minute}}.json).</div>
  </div>

  <div class="row" style="margin-top:16px;">
    <div class="card" style="flex: 1 1 420px;">
      <h3>Diagnosis (text)</h3>
      <div id="diagMeta" class="muted"></div>
      <pre id="diagText">—</pre>
    </div>

    <div class="card" style="flex: 1 1 420px;">
      <h3>Early warning (P(win) @ minute {{minute}})</h3>
      <div class="muted">Role-specific model trained on this run.</div>
      <div class="kpi" id="pwin">—</div>
      <div class="muted" id="pwinMsg"></div>
      <hr style="border:none;border-top:1px solid #eee;margin:14px 0;"/>
      <h3 style="margin-top:0;">What changes your odds?</h3>
      <div class="muted">Pick one feature and slide toward the WIN target median.</div>
      <div style="margin-top:10px;">
        <select id="featureSelect"></select>
      </div>
      <div style="margin-top:10px;">
        <input id="whatIfSlider" type="range" min="0" max="100" value="0" style="width:100%;" />
        <div class="muted">Toward WIN median: <span id="whatIfPct">0%</span></div>
      </div>
      <div style="margin-top:10px; display:flex; gap:10px; align-items:center;">
        <button id="btnWhatIfApply" class="btn" type="button">Apply</button>
        <div class="muted" id="whatIfPending">(drag slider, then click Apply)</div>
      </div>
      <div class="kpi" id="pwinWhatIf">—</div>
      <div class="muted" id="whatIfMsg"></div>
    </div>

    <div class="card" style="flex: 1 1 520px;">
      <h3>Gaps (dumbbell plot)</h3>
      <div class="muted">You (x) vs Target WIN median (o) vs LOSS median (o)</div>
      <div style="margin-top:10px;">
        <img id="gapsImg" src="" style="max-width:100%; border-radius:10px; border:1px solid #eee;" />
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h3>Match history (within this demo set)</h3>
    <div class="row">
      <div>
        <label class="muted">Filter role</label><br/>
        <select id="filterRole">
          <option value="">All</option>
          {% for r in roles %}<option value="{{r}}">{{r}}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label class="muted">Filter champ</label><br/>
        <input id="filterChamp" placeholder="e.g. 21" style="width:120px;" />
      </div>
      <div style="align-self:flex-end;">
        <button id="btnApply">Apply</button>
      </div>
    </div>
    <div style="max-height:320px; overflow:auto; margin-top:10px;">
      <table>
        <thead><tr><th>Match</th><th>Role</th><th>Champ</th><th>Outcome</th><th>Agency</th></tr></thead>
        <tbody id="historyBody"></tbody>
      </table>
    </div>
  </div>

<script>
let MATCH_ROWS = {{ match_rows_json | safe }};

function qs(id){ return document.getElementById(id); }

function chooseMatch(value){
  const sel = document.getElementById("matchSelect");
  sel.value = value;
  // Trigger any listeners (filters depend on current selection)
  sel.dispatchEvent(new Event("change"));
  // Run diagnosis immediately
  generate();
  window.scrollTo({top:0, behavior:"smooth"});
}

function renderHistory(){
  const role = qs('filterRole').value;
  const champ = (qs('filterChamp').value||'').trim();
  const body = qs('historyBody');
  body.innerHTML = '';
  const rows = MATCH_ROWS.filter(r => {
    if(role && r.role !== role) return false;
    if(champ && String(r.champion_id) !== champ) return false;
    return true;
  }).slice(0, 250);
  for(const r of rows){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a href="#" class="link" onclick="chooseMatch('${r.value}'); return false;">${r.match_id}</a></td><td>${r.role}</td><td>${r.champion_id}</td><td><span class="pill">${r.win===1?'WIN':'LOSS'}</span></td><td>${r.agency_bucket||''}</td>`;
    body.appendChild(tr);
  }
}

function fillFeatureSelect(features){
  const sel = qs('featureSelect');
  sel.innerHTML = '';
  for(const f of features){
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    sel.appendChild(opt);
  }
}



function selectMatch(match_id, puuid, role, champ) {
  // matchSelect uses the same encoding as the <option value>
  const val = `${match_id}::${puuid}::${role}::${champ}`;
  const sel = qs('matchSelect');
  if (sel) {
    sel.value = val;
  }
  // keep filters in sync
  if (qs('roleFilter')) qs('roleFilter').value = role;
  if (qs('champFilter')) qs('champFilter').value = String(champ);
  // run
  generate();
}

async function generate(){
  const val = qs('matchSelect').value;
  if(!val){ return; }
  const parts = val.split('::');
  const match_id = parts[0];
  const puuid = parts[1];
  const role = parts[2];
  const champion_id = parseInt(parts[3],10);

  const res = await fetch('/generate', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({match_id, puuid, role, champion_id})
  });
  const data = await res.json();
  qs('diagMeta').textContent = data.meta || '';
  qs('diagText').textContent = data.text || '—';

  // dumbbell
  if(data.viz_relpath){
    qs('gapsImg').src = `/viz?relpath=${encodeURIComponent(data.viz_relpath)}&t=${Date.now()}`;
  } else {
    qs('gapsImg').src = '';
  }

  // p(win)
  qs('pwin').textContent = (data.pwin == null) ? '—' : (Math.round(data.pwin*1000)/10).toFixed(1) + '%';
  qs('pwinMsg').textContent = data.pwin_msg || '';

  // feature list for what-if
  fillFeatureSelect(data.what_if_features || []);
  qs('whatIfSlider').value = 0;
  qs('whatIfPct').textContent = '0%';
  // Initialize what-if output to the base model probability.
  qs('pwinWhatIf').textContent = (data.pwin == null) ? '—' : (Math.round(data.pwin*1000)/10).toFixed(1) + '%';
  qs('whatIfMsg').textContent = 'Pick a feature, drag the slider toward the WIN median, then click Apply.';
}

async function updateWhatIf(){
  const val = qs('matchSelect').value;
  if(!val){ return; }
  const parts = val.split('::');
  const match_id = parts[0];
  const puuid = parts[1];
  const role = parts[2];

  const feature = qs('featureSelect').value;
  const pct = parseInt(qs('whatIfSlider').value,10);
  qs('whatIfPct').textContent = pct + '%';

  if(!feature){ return; }

  const res = await fetch('/what_if', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({match_id, puuid, role, feature, frac: pct/100.0, use_win_carrier: true})
  });
  const data = await res.json();
  if(!data.ok){
    qs('pwinWhatIf').textContent = '—';
    qs('whatIfMsg').textContent = data.error || 'what-if failed';
    return;
  }
  qs('pwinWhatIf').textContent = (data.pwin == null) ? '—' : (Math.round(data.pwin*1000)/10).toFixed(1) + '%';
  if(data.meta && data.meta.you != null && data.meta.target_median != null && data.meta.new_value != null){
    const you = data.meta.you;
    const tgt = data.meta.target_median;
    const nv = data.meta.new_value;
    // "Toward WIN median: X%" means nv = you + X%*(tgt - you)
    qs('whatIfMsg').textContent = `You: ${fmtNum(you)} → What-if: ${fmtNum(nv)} (WIN median: ${fmtNum(tgt)})`;
  } else {
    qs('whatIfMsg').textContent = '';
  }
}

qs('btnGenerate').addEventListener('click', generate);
qs('btnApply').addEventListener('click', renderHistory);
qs('featureSelect').addEventListener('change', () => {
  // Reset staged shift when changing feature
  qs('whatIfSlider').value = 0;
  qs('whatIfPct').textContent = '0%';
  qs('whatIfMsg').textContent = 'Pick a feature, drag the slider, then click Apply.';
  // Keep the displayed what-if equal to baseline until user applies.
});

// Slider only stages a change; recompute happens on explicit Apply.
qs('whatIfSlider').addEventListener('input', () => {
  const pct = parseInt(qs('whatIfSlider').value, 10);
  qs('whatIfPct').textContent = (isNaN(pct) ? 0 : pct) + '%';
  qs('whatIfMsg').textContent = 'Drag the slider, then click Apply.';
});

qs('btnWhatIfApply').addEventListener('click', () => { updateWhatIf(); });

renderHistory();
</script>

</body>
</html>
"""


# ---------- Routes ----------

@APP.get("/")
def index():
    c = get_cache()
    return render_template_string(
        TEMPLATE,
        run_dir=str(c.run_dir),
        minute=c.minute,
        match_count=len(c.match_options),
        options=c.match_options,
        roles=ROLE_ORDER,
        match_rows_json=json.dumps([r.__dict__ for r in c.match_rows[:500]]),
    )


@APP.get("/health")
def health():
    c = get_cache()
    return jsonify(
        {
            "ok": True,
            "run_dir": str(c.run_dir),
            "minute": c.minute,
            "matches": len(c.match_rows),
            "demo_pairs": len(c.demo_pairs),
        }
    )


@APP.get("/viz")
def viz():
    c = get_cache()
    relpath = request.args.get("relpath", "")
    if not relpath:
        return "missing relpath", 400
    p = (c.run_dir / relpath).resolve()
    # ensure it's under run_dir
    if c.run_dir.resolve() not in p.parents:
        return "invalid path", 400
    if not p.exists():
        return "not found", 404
    return send_file(str(p))


def _run_diagnosis_and_viz(run_dir: Path, minute: int, match_id: str, puuid: str, alpha: float) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (diag_text, diag_json_path, viz_relpath)."""
    # scripts (expected in project root)
    diag_script_candidates = [
        "diagnose_match_style.py"
    ]
    viz_script_candidates = [
        "visualize_match_diagnosis.py"
    ]

    diag_script = next((s for s in diag_script_candidates if Path(s).exists()), None)
    viz_script = next((s for s in viz_script_candidates if Path(s).exists()), None)

    if diag_script is None:
        raise FileNotFoundError("Could not find a diagnose_match_style script in current directory")

    # Run diagnosis
    cmd = [
        "python",
        diag_script,
        "--run-dir",
        str(run_dir),
        "--minute",
        str(minute),
        "--match-id",
        match_id,
        "--puuid",
        puuid,
        "--alpha",
        str(alpha),
        "--use-win-carrier",
    ]
    out = run_cmd(cmd)

    # Best effort: locate the produced json report
    report_dir = run_dir / "reports"
    diag_json = None
    if report_dir.exists():
        # search by match id
        cand = sorted(report_dir.glob(f"diagnose_match_style_{match_id}_*_m{minute}.json"))
        if not cand:
            cand = sorted(report_dir.glob(f"diagnose_match_style_{match_id}_*m{minute}.json"))
        if cand:
            diag_json = str(cand[-1])

    viz_relpath = None
    if viz_script is not None:
        try:
            cmd2 = [
                "python",
                viz_script,
                "--run-dir",
                str(run_dir),
                "--match-id",
                match_id,
                "--puuid",
                puuid,
                "--minute",
                str(minute),
            ]
            run_cmd(cmd2)
            # newest png under viz/match_diagnosis/m{minute}/<match_id*>/gaps_*.png
            base = run_dir / "viz" / "match_diagnosis" / f"m{minute}"
            if base.exists():
                pngs = sorted(base.glob(f"{match_id}_*/*gaps*.png"), key=lambda p: p.stat().st_mtime)
                if pngs:
                    viz_relpath = str(pngs[-1].relative_to(run_dir))
        except Exception:
            viz_relpath = None

    return out, diag_json, viz_relpath


@APP.post("/generate")
def generate():
    c = get_cache()
    data = request.get_json(force=True) or {}
    match_id = str(data.get("match_id", "")).strip()
    puuid = str(data.get("puuid", "")).strip()

    if not match_id or not puuid:
        return jsonify({"ok": False, "error": "missing match_id/puuid"}), 400

    ctx = c.get_row_context(match_id, puuid)
    role = str(data.get("role") or ctx.get("role") or "").upper()
    champ = safe_int(data.get("champion_id") or ctx.get("champion_id"))

    # Fetch pf row
    pf_row = c.get_pf_row(match_id, puuid)
    if pf_row is None:
        return jsonify({"ok": False, "error": "phase_features row not found"}), 400

    # Diagnosis & viz (alpha default moderate)
    alpha = float(data.get("alpha", 0.5))
    try:
        diag_text, diag_json_path, viz_relpath = _run_diagnosis_and_viz(c.run_dir, c.minute, match_id, puuid, alpha=alpha)
    except Exception as e:
        # still show pwin even if diagnosis fails
        diag_text = f"[error] diagnosis failed: {e}"
        diag_json_path = None
        viz_relpath = None

    # Match-specific what-if features + clearer p-value display
    what_if_features: List[str] = []
    if diag_json_path:
        try:
            dj = json.load(open(diag_json_path, "r", encoding="utf-8"))
            gaps = dj.get("gaps", [])
            if isinstance(gaps, list):
                for g in gaps:
                    if isinstance(g, dict) and g.get("feature"):
                        what_if_features.append(str(g["feature"]))

                # If the CLI-style text rounded p-values to 0.00/0.000, append a short,
                # unrounded summary sourced from the JSON.
                def fmt_p(pv: Any) -> str:
                    try:
                        pvv = float(pv)
                    except Exception:
                        return "?"
                    if not np.isfinite(pvv):
                        return "?"
                    if pvv == 0.0:
                        return "0"
                    if pvv < 1e-4:
                        return f"{pvv:.1e}"
                    if pvv < 0.001:
                        return f"{pvv:.3f}"
                    return f"{pvv:.4f}"

                lines = []
                for g in gaps[:10]:
                    if isinstance(g, dict) and g.get("feature") is not None:
                        lines.append(f"- {g.get('feature')}: p={fmt_p(g.get('p_value'))}")
                if lines:
                    diag_text = (diag_text or "") + "\n\n---\nP-values (unrounded from JSON):\n" + "\n".join(lines)
        except Exception:
            pass

    if not what_if_features:
        # fallback: allow moving any early-warning feature even if we can't find a target median
        what_if_features = list(c.ew_features)

    # p(win)
    pwin, pwin_msg = (None, None)
    if role:
        pwin, pwin_msg = c.compute_pwin(role, pf_row)
    else:
        pwin_msg = "Role not available"

    meta = f"Match: {match_id} | Role: {role or '?'} | Champ: {champ if champ is not None else '?'} | Outcome: {'WIN' if int(ctx.get('win',0))==1 else 'LOSS'}"

    return jsonify(
        {
            "ok": True,
            "meta": meta,
            "text": diag_text,
            "diag_json": diag_json_path,
            "viz_relpath": viz_relpath,
            "pwin": pwin,
            "pwin_msg": pwin_msg,
            "what_if_features": what_if_features if what_if_features else c.ew_features,
        }
    )


@APP.post("/what_if")
def what_if():
    c = get_cache()
    data = request.get_json(force=True) or {}

    match_id = str(data.get("match_id", "")).strip()
    puuid = str(data.get("puuid", "")).strip()
    role = str(data.get("role", "")).strip().upper()
    feature = str(data.get("feature", "")).strip()

    frac = data.get("frac")
    if frac is None and data.get("pct") is not None:
        try:
            frac = float(data.get("pct")) / 100.0
        except Exception:
            frac = None

    try:
        frac_f = float(frac)
    except Exception:
        # Don't 400 here; the slider fires rapidly and we want the UI to degrade gracefully.
        return jsonify({"ok": False, "error": "missing/invalid frac"}), 200

    if not (match_id and puuid and role and feature):
        return jsonify({"ok": False, "error": "missing match_id/puuid/role/feature"}), 200

    pf_row = c.get_pf_row(match_id, puuid)
    if pf_row is None:
        return jsonify({"ok": False, "error": "phase_features row not found"}), 200

    use_win_carrier = bool(data.get("use_win_carrier", False))

    base_val = float(pf_row.get(feature, np.nan))
    p, err = c.what_if(role, pf_row, match_id, puuid, feature, frac_f, use_win_carrier)
    if err:
        return jsonify({"ok": False, "error": err}), 200

    # For UI clarity: "frac" is an interpolation from your value to the target median.
    # new = you + frac*(target - you)
    try:
        target_median, how = c._get_target_median(match_id, puuid, role, feature, use_win_carrier)
        new_val = base_val + frac_f * (target_median - base_val)
        meta = {
            "you": base_val,
            "target_median": target_median,
            "new_value": float(new_val),
            "how": how,
        }
    except Exception:
        meta = {"you": base_val}

    return jsonify({"ok": True, "pwin": p, "meta": meta})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="runs/<RUN_ID> (default: latest under ./runs)")
    ap.add_argument("--minute", type=int, default=10)
    ap.add_argument("--port", type=int, default=5055)
    args = ap.parse_args()

    if args.run_dir:
        APP.config["RUN_DIR"] = args.run_dir
    APP.config["MINUTE"] = args.minute

    APP.run(host="127.0.0.1", port=args.port, debug=True)


if __name__ == "__main__":
    main()
