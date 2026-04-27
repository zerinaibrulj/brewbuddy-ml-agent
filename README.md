# BrewBuddy AI ☕

**AI Agent for Personalized Coffee Recommendations using Reinforcement Learning**

BrewBuddy is an intelligent AI barista that learns your coffee preferences through reinforcement learning. It uses Q-Learning, Thompson Sampling, and Upper Confidence Bound (UCB) algorithms to provide personalized coffee recommendations based on your ratings and contextual information.

## 🌟 Features

### Core Functionality
* **Intelligent Recommendations**: AI agent recommends coffee using Q-Learning, Thompson Sampling, or UCB algorithms
* **Context-Aware Learning**: Considers time of day, weather, and temperature for better recommendations
* **Interactive Rating System**: Rate coffees (1-5) to help the agent learn your preferences
* **Real-Time Learning**: Agent updates its knowledge immediately after each rating
* **Multiple Learning Strategies**: Switch between Q-Learning, Thompson Sampling, and UCB

### Sense-Think-Act-Learn Architecture
* **Sense**: Gathers context (time of day, weather, temperature)
* **Think**: Decides which coffee to recommend using selected strategy
* **Act**: Provides the recommendation
* **Learn**: Updates internal knowledge based on user feedback

### Advanced Features
* **Q-Table Visualization**: Interactive heatmap showing learned Q-values
* **Performance Analytics**: Track average ratings and popularity of each coffee
* **Learning Curve**: Visualize how the agent improves over time
* **Context Analysis**: See how different contexts affect recommendations
* **State Persistence**: Save and load agent state
* **Professional UI**: Modern, responsive Streamlit interface

## 🛠️ Technologies

* **Backend**: Python 3.8+
* **Frontend**: Streamlit
* **Machine Learning**: 
  - Q-Learning (Reinforcement Learning)
  - Thompson Sampling (Multi-Armed Bandit)
  - Upper Confidence Bound (UCB - Multi-Armed Bandit)
* **Visualization**: Plotly
* **Data Processing**: Pandas, NumPy

## 📋 Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## 🚀 Installation

1. **Clone or download the repository**:
   ```bash
   cd "BrewBuddy Agent"
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser**:
   The app will automatically open at `http://localhost:8501`

3. **Start using BrewBuddy**:
   - Click "Get Coffee Recommendation" to receive a suggestion
   - Rate the coffee (1-5 stars)
   - Watch the agent learn and improve over time!

## 📖 How It Works

### Q-Learning Algorithm
The agent uses Q-Learning to learn the optimal action (coffee) for each state (context). The Q-value update rule is:

```
Q(s,a) = Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate (how quickly the agent learns)
- **γ (gamma)**: Discount factor (importance of future rewards)
- **R**: Reward (user rating 1-5)
- **ε (epsilon)**: Exploration rate (balance between exploration and exploitation)

### Context-Aware States
The agent considers:
- **Time of Day**: Morning, Afternoon, Evening, Night
- **Weather**: Sunny, Rainy, Cloudy, Cold, Hot
- **Temperature**: Hot (>25°C), Cold (<15°C), Moderate

### Multi-Armed Bandit Alternatives
- **Thompson Sampling**: Bayesian approach using Beta distributions
- **UCB**: Upper Confidence Bound for exploration-exploitation balance

## 🎯 Usage Guide

### Getting Recommendations
1. Configure agent settings in the sidebar (learning rate, discount factor, etc.)
2. Set context information (time of day, weather, temperature)
3. Click "Get Coffee Recommendation"
4. Rate the recommended coffee
5. Submit your rating to help the agent learn

### Viewing Analytics
- **Q-Table Tab**: See the learned Q-values in a heatmap
- **Coffee Performance Tab**: Compare average ratings across different coffees
- **Learning Curve Tab**: Track improvement over time
- **Context Analysis Tab**: Understand how context affects recommendations

### Saving Progress
- Agent state is automatically saved after each rating
- Use "Save Agent State" button to manually save
- Use "Reset Agent" to start fresh

## 📊 Project Structure

```
BrewBuddy Agent/
├── streamlit_app.py          # Main Streamlit application
├── brewbuddy_agent.py        # Q-Learning agent implementation
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── agent_state.json         # Saved agent state (auto-generated)

```

## 🔬 Algorithm Details

### Q-Learning
- **States**: Context combinations (e.g., "morning_sunny_hot")
- **Actions**: Coffee selections
- **Rewards**: User ratings (1-5)
- **Update**: Bellman equation with learning rate and discount factor

### Thompson Sampling
- Uses Beta distribution for each coffee
- Samples from distribution to balance exploration/exploitation
- Updates alpha and beta parameters based on ratings

### Upper Confidence Bound (UCB)
- Calculates confidence intervals for each coffee
- Balances exploitation (high average reward) with exploration (uncertainty)
- Formula: `UCB = avg_reward + c·√(ln(total_pulls) / count)`

## 🎓 Educational Value

This project demonstrates:
- Reinforcement Learning in practice
- Multi-Armed Bandit problems
- Context-aware recommendation systems
- Interactive machine learning
- Real-time learning from user feedback

## 📝 Notes

- The agent learns from your ratings and improves over time
- Context-aware mode provides more personalized recommendations
- Different learning strategies may perform better for different users
- Agent state persists between sessions
- For best results, rate multiple coffees to help the agent learn

## 🤝 Contributing

This is a seminar project. Feel free to extend it with:
- More coffee options
- Additional context features
- Different reward functions
- Advanced visualization
- Integration with real coffee databases

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Professor's feedback and suggestions for Multi-Armed Bandits and context-aware recommendations
- Streamlit community for the excellent framework
- Coffee enthusiasts everywhere ☕

---

**Enjoy your personalized coffee recommendations!** ☕✨
