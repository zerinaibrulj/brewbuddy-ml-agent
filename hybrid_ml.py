"""
ML layer: state classification (placeholder for RandomForest/SVM), need/coffee vectors, cosine matching.
Reinforcement / bandit policy stays in BrewBuddyAgent (AI layer) but acts on a candidate set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from subjective_context import SubjectiveContext

def _time_bucket(time_of_day: Optional[str]) -> int:
    if not time_of_day:
        return 0
    m = (time_of_day or "morning").lower()
    order = ("morning", "afternoon", "evening", "night")
    return order.index(m) if m in order else 0


def subjective_features(
    subj: SubjectiveContext,
    time_of_day: Optional[str],
    temp_category: str,
) -> np.ndarray:
    """
    Full feature row for future supervised model + rule engine.
    Order: sleep, fatigue, lactose, social_empty, time_bucket, hot/cold/moderate one-hot[3]
    -> length 4 + 1 + 3 = 8
    """
    tcat = (temp_category or "moderate").lower()
    hot = 1.0 if tcat == "hot" else 0.0
    cold = 1.0 if tcat == "cold" else 0.0
    mod = 1.0 if tcat not in ("hot", "cold") else 0.0
    v = subj.feature_vector()
    return np.array(
        [
            v[0] / 12.0,
            v[1] / 10.0,
            v[2],
            v[3],
            _time_bucket(time_of_day) / 3.0,
            hot,
            cold,
            mod,
        ],
        dtype=float,
    )


def classify_state_rule(
    subj: SubjectiveContext,
    time_of_day: Optional[str],
    temp_category: str,
) -> str:
    """
    Rule-based classifier: mirrors plausible correlations (sleep + fatigue + time).
    Replace / augment with `StateClassifier` (RF) when enough logged labels exist.
    """
    t = (time_of_day or "morning").lower()
    if subj.fatigue >= 8 and subj.sleep_hours < 5.0 and t in ("morning", "afternoon"):
        return "extreme_caffeine"
    if t in ("evening", "night") and (subj.fatigue >= 6 or subj.social_battery == "Empty"):
        return "relaxation"
    if t == "morning" and subj.fatigue <= 4 and subj.sleep_hours >= 6.0:
        return "light_wakeup"
    if subj.social_battery == "Empty" and t in ("afternoon", "evening") and subj.fatigue >= 4:
        return "comfort"
    if (temp_category or "moderate").lower() == "hot" and t == "afternoon":
        return "balanced"
    if subj.fatigue >= 7:
        return "extreme_caffeine" if t in ("morning", "afternoon") else "comfort"
    return "balanced"


def build_need_vector(
    subj: SubjectiveContext,
    profile: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    4D need profile for cosine similarity to coffee feature vectors.
    Dims: need_stimulation, need_comfort, avoid_dairy, prefer_mild
    (all roughly 0–1, comparable to coffee side).
    """
    prof = profile or {}
    w_strong = float(prof.get("pref_strong_caffeine", 0.5))
    w_lf = 1.0 if int(prof.get("pref_lactose_free", 0)) else 0.0
    w_low_b = float(prof.get("pref_low_bitterness", 0.5))

    # Stimulation need rises with fatigue and short sleep; profile adds bias
    need_stim = np.clip(0.55 * (subj.fatigue / 10.0) + 0.45 * (1.0 - subj.sleep_hours / 12.0) + 0.15 * w_strong, 0.0, 1.0)
    # Comfort: low social energy or high fatigue in non-morning
    need_comfort = np.clip(0.5 * (1.0 if subj.social_battery == "Empty" else 0.0) + 0.2 * (subj.fatigue / 10.0), 0.0, 1.0)
    avoid_dairy = 1.0 if (subj.lactose_intolerance or w_lf > 0.5) else 0.0
    prefer_mild = np.clip(0.4 * w_low_b + 0.2 * (1.0 - subj.fatigue / 10.0), 0.0, 1.0)

    return np.array([need_stim, need_comfort, avoid_dairy, prefer_mild], dtype=float)


def coffee_to_vector(coffee: Dict[str, Any]) -> np.ndarray:
    """Maps DB row to same 4D space as build_need_vector."""
    c = float(coffee.get("caffeine_level", 0.5))
    milk = float(coffee.get("milk_level", 0.0))
    dairy = float(coffee.get("dairy_load", 0.0))
    bitter = float(coffee.get("bitterness", 0.5))
    stim = c
    comfort = 0.55 * milk + 0.45 * (1.0 - bitter * 0.2)
    dairy_risk = dairy
    mildness = 1.0 - float(bitter)
    return np.array([stim, comfort, dairy_risk, mildness], dtype=float)


def cosine_scores(
    need: np.ndarray,
    items: List[Tuple[str, np.ndarray]],
) -> Dict[str, float]:
    from numpy.linalg import norm

    if not items:
        return {}
    n = need.astype(float)
    nu = norm(n) or 1.0
    out: Dict[str, float] = {}
    for name, vec in items:
        v = vec.astype(float)
        nv = norm(v) or 1.0
        out[name] = float(np.dot(n, v) / (nu * nv))
    return out


def select_candidates(
    all_names: Sequence[str],
    coffee_dict: Dict[str, Dict[str, Any]],
    need: np.ndarray,
    subj: SubjectiveContext,
    top_k: int = 5,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Lactose: drop high-dairy if user must avoid. Then rank by cosine similarity, take top_k.
    """
    names = list(all_names)
    if subj.lactose_intolerance:
        names = [n for n in names if coffee_dict.get(n, {}).get("dairy_load", 1) <= 0.35]
    if not names:
        names = list(all_names)

    pairs: List[Tuple[str, np.ndarray]] = [
        (n, coffee_to_vector(coffee_dict[n])) for n in names if n in coffee_dict
    ]
    scores = cosine_scores(need, pairs)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[: top_k or len(scores)]
    cands = [k for k, _ in ranked]
    return cands, dict(ranked)


class StateClassifier:
    """
    Wraps a sklearn classifier once enough training data is available; falls back to rules.
    """

    def __init__(self) -> None:
        self._clf: Optional[RandomForestClassifier] = None
        self._enc: Optional[LabelEncoder] = None

    def is_fitted(self) -> bool:
        return self._clf is not None and self._enc is not None

    def fit_from_arrays(self, X: np.ndarray, y: Sequence[str]) -> None:
        if len(X) < 10 or len(y) < 10:
            self._clf = None
            self._enc = None
            return
        self._enc = LabelEncoder()
        y_enc = self._enc.fit_transform(list(y))
        self._clf = RandomForestClassifier(
            n_estimators=80, max_depth=6, class_weight="balanced", random_state=42
        )
        self._clf.fit(X, y_enc)

    def predict(
        self,
        subj: SubjectiveContext,
        time_of_day: Optional[str],
        temp_category: str,
    ) -> str:
        x = subjective_features(subj, time_of_day, temp_category).reshape(1, -1)
        if self._clf is not None and self._enc is not None:
            idx = int(self._clf.predict(x)[0])
            return str(self._enc.inverse_transform([idx])[0])
        return classify_state_rule(subj, time_of_day, temp_category)
