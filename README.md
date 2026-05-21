# BrewBuddy AI ☕

Hybrid AI Agent + Machine Learning recommender for personalized coffee decisions.

BrewBuddy is built as a competition-ready prototype where:
- **ML layer** understands user context and preference fit,
- **AI layer** selects actions adaptively using reinforcement/bandit policies,
- **Data layer** persists learning in SQLite, driven by a single curated café menu.

---

## What BrewBuddy Does

BrewBuddy combines:
1. **State understanding** (subjective + external context),
2. **Content-based matching** (need vector vs coffee vectors via cosine similarity),
3. **Policy selection** (Q-learning / Thompson / UCB) over a shortlist,
4. **Continuous learning** from user ratings.

---

## Core Implemented Features

### 1) Subjective + External Feature Engineering
- **Subjective**: `sleep_hours`, `fatigue`, `lactose_intolerance`, `social_battery`
- **External**: `time_of_day`, `weather`, `temperature`

### 2) Hybrid Decision Pipeline
- **Classifier layer** → categories (`extreme_caffeine`, `comfort`, `balanced`, …)
- **Content layer** → cosine similarity vs `coffee_items` feature vectors
- **Policy layer** → `qlearning` | `thompson` | `ucb`

### 3) Persistent Data Layer (SQLite)
Database: `data/brewbuddy.db`
| Table | Purpose |
|-------|---------|
| `coffee_items` | 35 drinks from `cafe_menu.csv` (`source_ref = cafe_menu`) |
| `user_profile` | Taste preferences |
| `interaction_log` | Ratings, vectors, context keys |

### 4) Dataset (single source of truth)
- **`brewbuddy_data/datasets/cafe_menu.csv`** — drink names, descriptions, roast, **image** filename
- **`images/`** — one photo per drink (see `image` column in CSV)
- Legacy Coffee Review CSVs live in `datasets/archive/` (not used by the app)

### 5) Explainability
- Narrative per recommendation (state, cosine match, constraints, policy)

### 6) Evaluation + Ablation
- Validation tab: state rewards, offline ablation (hybrid vs cosine-only vs random, …)

### 7) Modern UI / UX
- **Dashboard:** Browse full menu catalog (card grid + ML snapshot on click)
- **Sidebar:** Control room + engineering panel (need vector & cosine scores)
- Analytics: Q-Table, By coffee, Curve, Context, Catalog, Validation

---

## Tech Stack

- Python 3.8+
- Streamlit · Pandas · NumPy · scikit-learn · Plotly · Pillow · SQLite

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run streamlit_app.py
```

Open: `http://localhost:8501`

---

## Recommended Demo Flow

1. Click **Browse the full menu catalog** on the dashboard.
2. Set context in sidebar **Control room**.
3. **Request recommendation** → read explainability → rate.
4. Sidebar **Engineering** for need vector / cosine JSON.
5. **Validation** tab after several ratings.
6. **Analytics → Catalog** to show DB feature table.

---

## Project Structure

```text
BrewBuddy - ML - Agent/
├── streamlit_app.py          # UI
├── brewbuddy_agent.py        # Agent + RL
├── hybrid_ml.py              # Vectors, cosine, classifier
├── catalog_images.py         # Drink → image path (from CSV)
├── background_worker.py
├── brewbuddy_data/
│   ├── database.py           # SQLite + cafe_menu import
│   └── datasets/
│       ├── cafe_menu.csv     # ★ active catalog
│       └── archive/            # old research CSVs (unused)
├── images/                   # drink photos (names in CSV)
├── data/brewbuddy.db         # generated at runtime
└── agent_state.json          # generated at runtime
```

---

## Competition Readiness Notes

**Strengths:** hybrid architecture, persistence, explainability, ablations, polished UI, clear data story (one menu, one DB).

**Optional next steps:** classification metrics export, run config / seeds in appendix.

---

## License

Educational / academic use.
