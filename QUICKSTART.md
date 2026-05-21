# BrewBuddy Quick Start

Live demo guide for presentations.

---

## 1) Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

On first launch the app loads **35 drinks** from `cafe_menu.csv` into SQLite.

---

## 2) Demo Flow

1. Click **Browse the full menu catalog** (dashboard).
2. Tap a drink card → see **need vector**, **drink vector**, **cosine** score.
3. Sidebar: set sleep, fatigue, time, weather, taste profile.
4. **Request recommendation** → expand **Why this recommendation?**
5. Rate the drink (5–10 times for meaningful analytics).
6. **Validation** tab + **Engineering** expander for judges who want ML detail.

---

## 3) What to Show Judges

| Topic | Where |
|-------|--------|
| Hybrid ML + RL | Live context, recommendation, Validation |
| Data / DB | Analytics → **Catalog** tab (`coffee_items`) |
| UX / menu | Dashboard catalog + `images/` |
| Transparency | Sidebar → **Engineering** |

---

## 4) Analytics Tabs

- **Q-Table** — Q-learning only
- **By coffee** / **Curve** / **Context** — learning over time
- **Catalog** — feature table for all menu drinks
- **Validation** — ablation comparison

---

## 5) Sidebar

- **Learning engine** — policy, α, γ, ε
- **Environment** / **How you feel** / **Taste profile**
- **Hybrid model** toggle
- **Catalog maintenance** — reload CSV into DB if needed
- **Engineering** — raw vectors & cosine map

---

## 6) Troubleshooting

| Issue | Fix |
|-------|-----|
| Empty charts | Add ratings first |
| Wrong drink names in catalog | Sidebar → **Reload café menu from CSV** |
| Missing image | Check `image` column in `cafe_menu.csv` and file in `images/` |
| Validation empty | Need ≥5 rated interactions |

---

## 7) Paper Checklist

- [ ] 10+ rated interactions logged
- [ ] Validation + ablation visible
- [ ] Menu catalog + images demoed
- [ ] Screenshots of Engineering + Catalog tab

Full details: `README.md`.
