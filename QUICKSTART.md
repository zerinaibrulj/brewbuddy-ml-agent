# BrewBuddy Quick Start

This guide is optimized for live demos and competition presentations.

---

## 1) Run in 2 Commands

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

---

## 2) First Demo Flow (Recommended)

1. In sidebar (**Control room**), keep policy as `qlearning`.
2. Set subjective state:
   - Sleep, Fatigue, Lactose intolerance, Social battery.
3. Click **Request recommendation**.
4. Open **Why this recommendation?** and read the narrative.
5. Rate and submit.
6. Repeat a few times (5–10 interactions gives meaningful analytics).

---

## 3) Competition Features to Show

### A) Hybrid AI + ML
- ML state/category + cosine shortlist
- RL/bandit policy final action selection

### B) Dataset integration
- Sidebar → **Competition data boost**
- Click **Import datasets into catalog**
- Confirm catalog size increases

### C) Validation + Ablation
- Go to **Validation** tab
- Show:
  - state-level reward chart
  - ablation comparison table and chart

### D) Explainability
- In recommendation panel, expand:
  - **Why this recommendation?**

---

## 4) Analytics Tabs (Current UI)

- **Q-Table**: state-action heatmap (Q-learning only)
- **By coffee**: average reward per drink
- **Curve**: learning trajectory over interactions
- **Context**: performance by state/context
- **Catalog**: source composition + feature table
- **Validation**: evaluation + ablation outputs

---

## 5) Key Sidebar Controls

- **Learning engine**: policy + α/γ/ε
- **Environment**: time/weather/temperature
- **How you feel**: subjective features
- **Taste profile**: user preference vector
- **Hybrid model**: turn classifier+shortlist pipeline on/off
- **Competition data boost**: import external dataset rows

---

## 6) Troubleshooting

### App starts but charts look empty
- You need rated interactions first.
- `Q-Table` requires `qlearning`.

### Recommendation is missing image
- Imported coffees may not have a dedicated local image.
- UI uses fallback coffee imagery automatically.

### Validation tab says insufficient data
- Add more ratings (minimum ~5, better at 10+).

### Import button doesn’t change much
- Rows are upserted by coffee name; existing names update.

---

## 7) “Paper-ready” Checklist

Before submitting/reporting:

- [ ] At least 10–20 logged ratings
- [ ] Validation tab populated
- [ ] Ablation table populated
- [ ] Explainability narrative shown in demo
- [ ] Dataset import demonstrated
- [ ] Screenshots exported for report appendix

---

For full architecture and technical details, see `README.md`.

