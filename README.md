# BrewBuddy AI ☕

Hybrid AI Agent + Machine Learning recommender for personalized coffee decisions.

BrewBuddy is built as a competition-ready prototype where:
- **ML layer** understands user context and preference fit,
- **AI layer** selects actions adaptively using reinforcement/bandit policies,
- **Data layer** persists learning and supports dataset-driven catalog expansion.

---

## What BrewBuddy Does

BrewBuddy combines:
1. **State understanding** (subjective + external context),
2. **Content-based matching** (need vector vs coffee vectors via cosine similarity),
3. **Policy selection** (Q-learning / Thompson / UCB) over a shortlist,
4. **Continuous learning** from user ratings.

It supports both classic RL behavior and a modern hybrid ML workflow.

---

## Core Implemented Features

### 1) Subjective + External Feature Engineering
- **Subjective**: `sleep_hours`, `fatigue`, `lactose_intolerance`, `social_battery`
- **External**: `time_of_day`, `weather`, `temperature`
- These are used to build richer context/state keys and recommendation logic.

### 2) Hybrid Decision Pipeline
- **Classifier layer** maps current state to categories (e.g., `extreme_caffeine`, `comfort`, `balanced`, etc.).
- **Content layer** computes cosine similarity between current need vector and coffee feature vectors.
- **Policy layer** chooses the final recommendation from the shortlisted set using:
  - `qlearning`
  - `thompson`
  - `ucb`

### 3) Persistent Data Layer (SQLite)
Database: `data/brewbuddy.db`
- `coffee_items` (catalog + normalized features)
- `user_profile` (taste preferences)
- `interaction_log` (full learning history)

### 4) Dataset Integration
- Built-in import pipeline for:
  - `brewbuddy_data/datasets/simplified_coffee.csv`
  - `brewbuddy_data/datasets/coffee_analysis.csv`
- New rows are normalized and upserted into `coffee_items`.

### 5) Explainability
- Every recommendation can show a **narrative explanation**:
  - predicted state,
  - cosine match relevance,
  - active constraints (e.g., lactose),
  - final policy decision context.

### 6) Evaluation + Ablation
- Validation tab includes:
  - state-level reward analysis
  - coverage indicators
  - offline ablation comparison
- Ablations compare:
  - Hybrid (logged)
  - Cosine-only
  - Content-only
  - Bandit-mean
  - Random

### 7) Modern UI / UX
- Premium dark interface (gold/coffee accents)
- Card-based recommendation display + alternatives
- Analytics tabs for:
  - Q-Table
  - By coffee
  - Curve
  - Context
  - Catalog
  - Validation

---

## Tech Stack

- Python 3.8+
- Streamlit
- Pandas / NumPy
- scikit-learn
- Plotly
- SQLite (built-in `sqlite3`)
- Pillow

See `requirements.txt` for exact dependencies.

---

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

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

1. Launch app and open sidebar “Control room”.
2. Set context + subjective state.
3. Request recommendation and inspect explainability narrative.
4. Rate recommendation.
5. Open **Validation** tab to view evaluation + ablations.
6. Use **Competition data boost** to import dataset rows and expand catalog.

---

## Project Structure

```text
BrewBuddy - ML - Agent/
├── streamlit_app.py
├── brewbuddy_agent.py
├── hybrid_ml.py
├── subjective_context.py
├── background_worker.py
├── run_background_worker.py
├── brewbuddy_data/
│   ├── __init__.py
│   ├── database.py
│   └── datasets/
│       ├── simplified_coffee.csv
│       └── coffee_analysis.csv
├── data/
│   └── brewbuddy.db               # generated at runtime
├── agent_state.json               # generated at runtime
├── QUICKSTART.md
└── README.md
```

---

## Competition Readiness Notes

What is already strong:
- Hybrid AI + ML architecture
- Real persistence and dataset integration
- Explainability narrative
- Offline evaluation and ablations
- Professional UI and analytics

What to improve further for top-tier judging:
- Add confusion matrix / classification metrics from a trained classifier
- Add exported result snapshots (CSV/figures) for paper appendix
- Add reproducibility checklist (random seeds, run configs, data versioning)

---

## License

Educational / academic use.
