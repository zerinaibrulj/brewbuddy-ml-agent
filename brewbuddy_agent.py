"""
BrewBuddy: hybrid ML (state classification + content match) + RL (Q-learning / bandits).
Implements Sense–Think–Act–Learn; logs interactions to SQLite; optional JSON for agent weights.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from brewbuddy_data.database import get_default_db_path, init_db, log_interaction
from brewbuddy_data.database import get_coffee_dicts, get_coffee_list
from hybrid_ml import StateClassifier, build_need_vector, coffee_to_vector, select_candidates
from subjective_context import SubjectiveContext


class NoWork:
    """Explicit no-work: agent is alive but nothing to process (queue, cooldown, or pending rating)."""

    def __init__(self, reason: str = "No work available"):
        self.reason = reason

    def __repr__(self):
        return f"NoWork(reason='{self.reason}')"

    def __eq__(self, other):
        return isinstance(other, NoWork)


class BrewBuddyAgent:
    """
    AI layer: Q-learning, Thompson, or UCB on a (possibly) restricted candidate set.
    ML layer: state classification + cosine shortlist over SQLite coffee features.
    """

    def __init__(
        self,
        coffees: Optional[Sequence[str]] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.3,
        use_context: bool = True,
        use_subjective: bool = True,
        use_hybrid: bool = True,
        strategy: str = "qlearning",
        db_path: Optional[Path] = None,
    ) -> None:
        self._db_path = db_path or get_default_db_path()
        init_db(self._db_path)

        if coffees is None:
            self.coffees: List[str] = get_coffee_list(self._db_path)
        else:
            self.coffees = list(coffees)
        full = get_coffee_dicts(self._db_path)
        self.coffee_items: Dict[str, Dict[str, Any]] = {n: full[n] for n in self.coffees if n in full}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.use_context = use_context
        self.use_subjective = use_subjective
        self.use_hybrid = use_hybrid
        self.strategy = strategy
        self.state_classifier = StateClassifier()

        self.q_table: Dict[str, Any] = defaultdict(lambda: defaultdict(float))

        if self.strategy == "thompson":
            self.alpha: Dict[str, float] = {c: 1.0 for c in self.coffees}
            self.beta: Dict[str, float] = {c: 1.0 for c in self.coffees}
        elif self.strategy == "ucb":
            self.action_counts: Dict[str, int] = {c: 0 for c in self.coffees}
            self.action_rewards: Dict[str, List[float]] = {c: [] for c in self.coffees}
            self.total_pulls: int = 0

        self.interaction_history: List[Dict[str, Any]] = []
        self.total_interactions: int = 0
        self.best_coffee: Optional[str] = None
        self.best_rating: float = 0

        self.current_context: Optional[str] = None
        self.current_subjective: SubjectiveContext = SubjectiveContext()
        self.current_ml_state: str = "balanced"
        self._candidate_coffees: List[str] = []
        self.last_cosine_scores: Dict[str, float] = {}
        self.last_need_vector: Optional[List[float]] = None

        self.context_queue: Deque[Dict[str, Any]] = deque()
        self.last_recommendation_time: Optional[datetime] = None
        self.min_recommendation_interval: timedelta = timedelta(seconds=5)
        self.pending_recommendation: Optional[str] = None
        self._last_user_profile: Optional[Dict[str, Any]] = None
        self._last_sense: Dict[str, Any] = {}

    def reload_catalog_from_db(self) -> None:
        """Refresh menu items from SQLite after dataset imports."""
        all_names = get_coffee_list(self._db_path)
        self.coffees = list(all_names)
        full = get_coffee_dicts(self._db_path)
        self.coffee_items = {n: full[n] for n in self.coffees if n in full}
        if hasattr(self, "alpha"):
            for c in self.coffees:
                self.alpha.setdefault(c, 1.0)  # type: ignore[attr-defined]
                self.beta.setdefault(c, 1.0)  # type: ignore[attr-defined]
        if hasattr(self, "action_counts"):
            for c in self.coffees:
                self.action_counts.setdefault(c, 0)  # type: ignore[attr-defined]
                self.action_rewards.setdefault(c, [])  # type: ignore[attr-defined]

    def _action_space(self) -> List[str]:
        c = [x for x in (self._candidate_coffees or []) if x in self.coffees]
        if c:
            return c
        return list(self.coffees)

    @staticmethod
    def _temp_category(temperature: Optional[float]) -> str:
        if temperature is None:
            return "moderate"
        if float(temperature) > 25:
            return "hot"
        if float(temperature) < 15:
            return "cold"
        return "moderate"

    def sense(
        self,
        time_of_day: Optional[str] = None,
        weather: Optional[str] = None,
        temperature: Optional[float] = None,
        subjective: Any = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        if time_of_day is None:
            h = datetime.now().hour
            if 5 <= h < 12:
                time_of_day = "morning"
            elif 12 <= h < 17:
                time_of_day = "afternoon"
            elif 17 <= h < 22:
                time_of_day = "evening"
            else:
                time_of_day = "night"

        if isinstance(subjective, SubjectiveContext):
            self.current_subjective = subjective
        else:
            self.current_subjective = SubjectiveContext.from_dict(subjective)

        self._last_user_profile = user_profile
        self._last_sense = {
            "time_of_day": time_of_day,
            "weather": weather,
            "temperature_c": float(temperature) if temperature is not None else None,
        }

        temp_category = self._temp_category(temperature)
        if self.use_context:
            parts = [time_of_day] if time_of_day else ["default"]
            if weather:
                parts.append(weather)
            if temperature is not None:
                parts.append(temp_category)
        else:
            parts = ["default"]
        if self.use_subjective:
            subj_key = (
                f"s{int(self.current_subjective.sleep_hours):02d}f{self.current_subjective.fatigue}"
                f"l{1 if self.current_subjective.lactose_intolerance else 0}"
                f"soc{self.current_subjective.social_battery[:1]}"
            )
        else:
            subj_key = "nosubj"
        if self.use_hybrid:
            self.current_ml_state = self.state_classifier.predict(
                self.current_subjective, time_of_day, temp_category
            )
            in_cat: List[str] = [
                n
                for n in self.coffees
                if (self.coffee_items.get(n) or {}).get("state_category") == self.current_ml_state
            ]
            if not in_cat:
                in_cat = list(self.coffees)
            need = build_need_vector(self.current_subjective, user_profile)
            self.last_need_vector = need.tolist()
            cands, cos_scores = select_candidates(
                in_cat, self.coffee_items, need, self.current_subjective, top_k=5
            )
            self._candidate_coffees = cands
            self.last_cosine_scores = cos_scores
        else:
            self.current_ml_state = "balanced"
            self._candidate_coffees = list(self.coffees)
            n = build_need_vector(self.current_subjective, user_profile)
            self.last_need_vector = n.tolist()
            self.last_cosine_scores = {}

        if self.use_context:
            ext = "_".join(parts)
        else:
            ext = "default" if not self.use_subjective else f"default_{subj_key}"
        if self.use_subjective:
            if self.use_hybrid:
                self.current_context = f"{ext}__{subj_key}__{self.current_ml_state}"
            else:
                self.current_context = f"{ext}__{subj_key}"
        else:
            if self.use_hybrid:
                self.current_context = f"{ext}__{self.current_ml_state}"
            else:
                self.current_context = ext
        return str(self.current_context)

    def think(self) -> str:
        if self.strategy == "qlearning":
            return self._think_qlearning()
        if self.strategy == "thompson":
            return self._think_thompson()
        if self.strategy == "ucb":
            return self._think_ucb()
        return str(np.random.choice(self._action_space()))

    def _think_qlearning(self) -> str:
        space = self._action_space()
        if not space:
            return str(np.random.choice(self.coffees))
        if np.random.random() < self.epsilon:
            return str(np.random.choice(space))
        state = self.current_context or "default"
        qv = {a: float(self.q_table[state][a]) for a in space}
        m = max(qv.values()) if qv else 0.0
        if m == 0:
            return str(np.random.choice(space))
        best = [a for a, v in qv.items() if v == m]
        return str(np.random.choice(best))

    def _think_thompson(self) -> str:
        space = self._action_space()
        if not space:
            return str(np.random.choice(self.coffees))
        s = {c: float(np.random.beta(self.alpha[c], self.beta[c])) for c in space}
        return str(max(s, key=s.get))

    def _think_ucb(self) -> str:
        space = self._action_space()
        if not space:
            return str(np.random.choice(self.coffees))
        if self.total_pulls == 0:
            return str(np.random.choice(space))
        c = 2.0
        ucb: Dict[str, float] = {}
        for a in space:
            if self.action_counts[a] == 0:
                ucb[a] = float("inf")
            else:
                avg = float(np.mean(self.action_rewards[a]))
                conf = c * float(np.sqrt(np.log(self.total_pulls) / self.action_counts[a]))
                ucb[a] = avg + conf
        return str(max(ucb, key=ucb.get))

    def act(self) -> str:
        return self.think()

    def add_context_request(
        self,
        time_of_day: Optional[str] = None,
        weather: Optional[str] = None,
        temperature: Optional[float] = None,
        subjective: Any = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.context_queue.append(
            {
                "time_of_day": time_of_day,
                "weather": weather,
                "temperature": temperature,
                "subjective": subjective,
                "user_profile": user_profile,
                "timestamp": datetime.now(),
            }
        )

    def tick(self) -> Union[str, NoWork]:
        if self.pending_recommendation is not None:
            return NoWork("Waiting for user rating on pending recommendation")
        if self.last_recommendation_time is not None:
            d = datetime.now() - self.last_recommendation_time
            if d < self.min_recommendation_interval:
                rem = (self.min_recommendation_interval - d).total_seconds()
                return NoWork(f"Not enough time since last recommendation ({rem:.1f}s remaining)")
        if len(self.context_queue) == 0:
            return NoWork("No new user context in queue")
        ctx = self.context_queue.popleft()
        self.sense(
            time_of_day=ctx.get("time_of_day"),
            weather=ctx.get("weather"),
            temperature=ctx.get("temperature"),
            subjective=ctx.get("subjective"),
            user_profile=ctx.get("user_profile"),
        )
        rec = self.think()
        self.last_recommendation_time = datetime.now()
        self.pending_recommendation = rec
        return rec

    def learn(self, coffee: str, rating: int) -> None:
        reward = float(rating)
        state = self.current_context or "default"
        s = str(state)

        if self.strategy == "qlearning":
            current_q = float(self.q_table[s][coffee])
            next_s = s
            mx = max([float(self.q_table[next_s][a]) for a in self.coffees], default=0.0)
            self.q_table[s][coffee] = current_q + self.learning_rate * (
                reward + self.discount_factor * mx - current_q
            )
        elif self.strategy == "thompson":
            nrm = (rating - 1) / 4.0
            self.alpha[coffee] = self.alpha.get(coffee, 1.0) + nrm
            self.beta[coffee] = self.beta.get(coffee, 1.0) + (1.0 - nrm)
        elif self.strategy == "ucb":
            self.action_counts[coffee] = self.action_counts.get(coffee, 0) + 1
            self.action_rewards.setdefault(coffee, []).append(float(rating))
            self.total_pulls += 1

        self.interaction_history.append(
            {
                "coffee": coffee,
                "rating": int(rating),
                "reward": float(reward),
                "context": s,
                "ml_state": self.current_ml_state,
                "subjective": self.current_subjective.to_dict(),
                "candidates": list(self._candidate_coffees),
                "cosine_scores": dict(self.last_cosine_scores),
                "need_vector": self.last_need_vector,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.total_interactions += 1
        if rating > self.best_rating:
            self.best_rating = float(rating)
            self.best_coffee = coffee
        if self.pending_recommendation == coffee:
            self.pending_recommendation = None

        rec_v = self.coffee_items.get(coffee)
        cvec: Optional[List[float]] = None
        if rec_v is not None:
            cvec = [float(x) for x in coffee_to_vector(rec_v).tolist()]

        try:
            log_interaction(
                {
                    "time_of_day": (self._last_sense or {}).get("time_of_day"),
                    "weather": (self._last_sense or {}).get("weather"),
                    "temperature_c": (self._last_sense or {}).get("temperature_c"),
                    "sleep_hours": self.current_subjective.sleep_hours,
                    "fatigue": self.current_subjective.fatigue,
                    "lactose_intolerance": self.current_subjective.lactose_intolerance,
                    "social_battery": self.current_subjective.social_battery,
                    "context_key": s,
                    "ml_state_category": self.current_ml_state,
                    "need_vector": self.last_need_vector,
                    "recommended_coffee": coffee,
                    "recommended_coffee_vector": cvec,
                    "candidates": self._candidate_coffees,
                    "cosine_scores": self.last_cosine_scores,
                    "rating": int(rating),
                    "reward": float(reward),
                    "strategy": self.strategy,
                },
                db_path=self._db_path,
            )
        except (sqlite3.OperationalError, OSError) as e:
            print(f"SQLite log failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        av = np.mean([h["rating"] for h in self.interaction_history]) if self.interaction_history else 0
        coffee_stats: Dict[str, Dict[str, Any]] = {}
        for c in self.coffees:
            rts = [h["rating"] for h in self.interaction_history if h.get("coffee") == c]
            if rts:
                coffee_stats[c] = {
                    "count": len(rts),
                    "avg_rating": float(np.mean(rts)),
                    "total_rating": float(sum(rts)),
                }
        return {
            "total_interactions": self.total_interactions,
            "average_rating": float(av) if self.interaction_history else 0.0,
            "best_coffee": self.best_coffee,
            "best_rating": self.best_rating,
            "coffee_stats": coffee_stats,
            "menu_size": len(self.coffees),
            "shortlist_size": len(self._candidate_coffees),
        }

    def get_q_table_df(self) -> pd.DataFrame:
        if not self.q_table:
            return pd.DataFrame()
        states = [k for k in self.q_table]
        if not states:
            return pd.DataFrame()
        data: Dict[str, List[float]] = {}
        for coffee in self.coffees:
            data[coffee] = [float(self.q_table[st].get(coffee, 0.0)) for st in states]  # type: ignore[union-attr, arg-type]
        return pd.DataFrame(data, index=states)

    def save_state(self, filepath: str = "agent_state.json") -> None:
        state: Dict[str, Any] = {
            "q_table": {str(k): dict(v) for k, v in self.q_table.items()},
            "interaction_history": self.interaction_history,
            "total_interactions": self.total_interactions,
            "best_coffee": self.best_coffee,
            "best_rating": self.best_rating,
            "strategy": self.strategy,
            "use_context": self.use_context,
            "use_subjective": self.use_subjective,
            "use_hybrid": self.use_hybrid,
            "coffees": self.coffees,
        }
        if self.strategy == "thompson":
            state["alpha"] = self.alpha
            state["beta"] = self.beta
        elif self.strategy == "ucb":
            state["action_counts"] = self.action_counts
            state["action_rewards"] = {k: v for k, v in self.action_rewards.items()}
            state["total_pulls"] = self.total_pulls
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str = "agent_state.json") -> bool:
        if not os.path.exists(filepath):
            return False
        with open(filepath, encoding="utf-8") as f:
            state = json.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in state.get("q_table", {}).items():
            self.q_table[k] = defaultdict(float, v)
        self.interaction_history = state.get("interaction_history", [])
        self.total_interactions = int(state.get("total_interactions", 0))
        self.best_coffee = state.get("best_coffee")
        self.best_rating = float(state.get("best_rating", 0) or 0)
        if state.get("use_subjective") is not None:
            self.use_subjective = bool(state.get("use_subjective", True))
        if state.get("use_hybrid") is not None:
            self.use_hybrid = bool(state.get("use_hybrid", True))
        if self.strategy == "thompson" and "alpha" in state:
            self.alpha = {k: float(v) for k, v in state["alpha"].items()}  # type: ignore[union-attr]
            self.beta = {k: float(v) for k, v in state["beta"].items()}  # type: ignore[union-attr]
            for c in self.coffees:
                self.alpha.setdefault(c, 1.0)
                self.beta.setdefault(c, 1.0)
        elif self.strategy == "ucb" and "action_counts" in state:
            self.action_counts = {k: int(v) for k, v in state["action_counts"].items()}  # type: ignore[union-attr, arg-type]
            self.action_rewards = {k: [float(x) for x in v] for k, v in state.get("action_rewards", {}).items()}
            self.total_pulls = int(state.get("total_pulls", 0))
            for c in self.coffees:
                self.action_counts.setdefault(c, 0)
                self.action_rewards.setdefault(c, [])
        return True
