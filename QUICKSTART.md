# BrewBuddy Quick Start Guide

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Start using BrewBuddy**:
   - The app opens in your browser automatically
   - Click "Get Coffee Recommendation" to get your first suggestion
   - Rate the coffee (1-5 stars)
   - Submit your rating
   - Repeat to help the agent learn!

## 🎯 Key Features to Try

### 1. Basic Recommendation Flow
- Get a recommendation → Rate it → See the agent learn
- Watch statistics update in real-time

### 2. Context-Aware Recommendations
- Enable "Use Context-Aware States" in sidebar
- Set time of day, weather, and temperature
- See how context affects recommendations

### 3. Different Learning Strategies
Try switching between:
- **Q-Learning**: Full reinforcement learning with states
- **Thompson Sampling**: Bayesian Multi-Armed Bandit
- **UCB**: Upper Confidence Bound

### 4. Visualizations
Explore the tabs:
- **Q-Table**: See learned Q-values (Q-Learning only)
- **Coffee Performance**: Compare coffee ratings
- **Learning Curve**: Track improvement over time
- **Context Analysis**: See context-based patterns

### 5. Hyperparameter Tuning
Adjust in sidebar:
- **Learning Rate (α)**: How quickly the agent learns (0.01-0.5)
- **Discount Factor (γ)**: Importance of future rewards (0.1-0.99)
- **Exploration Rate (ε)**: Balance exploration vs exploitation (0.0-1.0)

## 💡 Tips for Best Results

1. **Rate Multiple Coffees**: The more ratings, the better the agent learns
2. **Use Context**: Enable context-aware mode for personalized recommendations
3. **Experiment**: Try different strategies and hyperparameters
4. **Be Consistent**: Rate honestly to help the agent learn your true preferences
5. **Check Analytics**: Review the learning curve to see improvement

## 🔧 Troubleshooting

**App won't start?**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Agent not learning?**
- Make sure you're submitting ratings after getting recommendations
- Check that you're not resetting the agent too frequently
- Verify hyperparameters are set correctly

**Visualizations empty?**
- Some visualizations require interaction history
- Q-Table tab only works with Q-Learning strategy
- Make sure you've rated at least a few coffees

## 📚 Understanding the Agent

### Sense-Think-Act-Learn Cycle

1. **Sense**: Agent gathers context (time, weather, temperature)
2. **Think**: Agent decides which coffee to recommend
3. **Act**: Agent provides the recommendation
4. **Learn**: Agent updates knowledge based on your rating

### How Q-Learning Works

- Agent maintains a Q-table: `Q(state, action) = expected_reward`
- Updates Q-values using: `Q(s,a) = Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]`
- Uses epsilon-greedy: explore randomly ε% of time, exploit best action (1-ε)% of time

### Context States

States are combinations like:
- `morning_sunny_hot`
- `evening_rainy_cold`
- `afternoon_moderate`

The agent learns which coffee works best in each context!

## 🎓 For Your Seminar

This implementation includes:
✅ Q-Learning algorithm
✅ Context-aware recommendations (Sense phase)
✅ Multi-Armed Bandits (Thompson Sampling, UCB)
✅ Sense-Think-Act-Learn architecture
✅ Professional Streamlit UI
✅ Q-table visualization
✅ Learning progress tracking
✅ State persistence

Perfect for demonstrating reinforcement learning in practice!

