"""
BrewBuddy AI Agent - Q-Learning with Context-Aware Recommendations
Implements Sense-Think-Act-Learn architecture
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import os
from typing import Optional, Dict, Any, Union


class NoWork:
    """
    Explicit NoWork state indicating the agent is active but has no work to process.
    """
    def __init__(self, reason: str = "No work available"):
        self.reason = reason
    
    def __repr__(self):
        return f"NoWork(reason='{self.reason}')"
    
    def __eq__(self, other):
        return isinstance(other, NoWork)


class BrewBuddyAgent:
    """
    AI Agent that learns user preferences using Q-Learning
    Implements Sense-Think-Act-Learn architecture
    """
    
    def __init__(self, coffees, learning_rate=0.1, discount_factor=0.9, 
                 epsilon=0.3, use_context=True, strategy='qlearning'):
        """
        Initialize the BrewBuddy agent
        
        Args:
            coffees: List of available coffee options
            learning_rate: Q-learning learning rate (alpha)
            discount_factor: Q-learning discount factor (gamma)
            epsilon: Exploration rate for epsilon-greedy
            use_context: Whether to use context-aware states
            strategy: 'qlearning', 'thompson', or 'ucb'
        """
        self.coffees = coffees
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.use_context = use_context
        self.strategy = strategy
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # For Multi-Armed Bandits
        if strategy == 'thompson':
            # Thompson Sampling: Beta distribution parameters
            self.alpha = {coffee: 1.0 for coffee in coffees}  # Success count
            self.beta = {coffee: 1.0 for coffee in coffees}   # Failure count
        elif strategy == 'ucb':
            # UCB: Track counts and average rewards
            self.action_counts = {coffee: 0 for coffee in coffees}
            self.action_rewards = {coffee: [] for coffee in coffees}
            self.total_pulls = 0
        
        # Statistics
        self.interaction_history = []
        self.total_interactions = 0
        self.best_coffee = None
        self.best_rating = 0
        
        # Context state
        self.current_context = None
        
        # Background runner support
        self.context_queue = deque()  # Queue for pending context requests
        self.last_recommendation_time = None  # Track when last recommendation was made
        self.min_recommendation_interval = timedelta(seconds=5)  # Minimum time between recommendations
        self.pending_recommendation = None  # Current recommendation waiting for rating
        
    def sense(self, time_of_day=None, weather=None, temperature=None):
        """
        SENSE phase: Gather context information about the environment
        
        Args:
            time_of_day: 'morning', 'afternoon', 'evening', 'night'
            weather: 'sunny', 'rainy', 'cloudy', 'cold', 'hot'
            temperature: Numeric temperature value
        """
        # Auto-detect time of day if not provided
        if time_of_day is None:
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = 'morning'
            elif 12 <= hour < 17:
                time_of_day = 'afternoon'
            elif 17 <= hour < 22:
                time_of_day = 'evening'
            else:
                time_of_day = 'night'
        
        # Create context state
        if self.use_context:
            context_parts = [time_of_day]
            if weather:
                context_parts.append(weather)
            if temperature:
                temp_category = 'hot' if temperature > 25 else 'cold' if temperature < 15 else 'moderate'
                context_parts.append(temp_category)
            self.current_context = '_'.join(context_parts)
        else:
            self.current_context = 'default'
        
        return self.current_context
    
    def think(self):
        """
        THINK phase: Decide which coffee to recommend
        Uses epsilon-greedy for Q-learning, or MAB strategies
        """
        if self.strategy == 'qlearning':
            return self._think_qlearning()
        elif self.strategy == 'thompson':
            return self._think_thompson()
        elif self.strategy == 'ucb':
            return self._think_ucb()
        else:
            return np.random.choice(self.coffees)
    
    def _think_qlearning(self):
        """Q-learning decision making with epsilon-greedy"""
        # Exploration: random choice
        if np.random.random() < self.epsilon:
            return np.random.choice(self.coffees)
        
        # Exploitation: choose best action for current state
        state = self.current_context
        q_values = {coffee: self.q_table[state][coffee] for coffee in self.coffees}
        
        # If all Q-values are 0 (initialization), choose randomly
        max_q = max(q_values.values())
        if max_q == 0:
            return np.random.choice(self.coffees)
        
        # Choose action with highest Q-value
        best_actions = [coffee for coffee, q_val in q_values.items() if q_val == max_q]
        return np.random.choice(best_actions)
    
    def _think_thompson(self):
        """Thompson Sampling for Multi-Armed Bandit"""
        # Sample from Beta distribution for each coffee
        samples = {}
        for coffee in self.coffees:
            samples[coffee] = np.random.beta(self.alpha[coffee], self.beta[coffee])
        
        # Choose coffee with highest sample
        return max(samples, key=samples.get)
    
    def _think_ucb(self):
        """Upper Confidence Bound (UCB) for Multi-Armed Bandit"""
        if self.total_pulls == 0:
            return np.random.choice(self.coffees)
        
        ucb_values = {}
        for coffee in self.coffees:
            if self.action_counts[coffee] == 0:
                # If never tried, give it high priority
                ucb_values[coffee] = float('inf')
            else:
                # Average reward
                avg_reward = np.mean(self.action_rewards[coffee])
                # UCB formula: avg_reward + c * sqrt(ln(total_pulls) / count)
                c = 2.0  # Exploration constant
                confidence = c * np.sqrt(np.log(self.total_pulls) / self.action_counts[coffee])
                ucb_values[coffee] = avg_reward + confidence
        
        return max(ucb_values, key=ucb_values.get)
    
    def act(self):
        """
        ACT phase: Recommend a coffee
        """
        recommended_coffee = self.think()
        return recommended_coffee
    
    def add_context_request(self, time_of_day=None, weather=None, temperature=None):
        """
        Add a context request to the queue for background processing.
        
        Args:
            time_of_day: 'morning', 'afternoon', 'evening', 'night'
            weather: 'sunny', 'rainy', 'cloudy', 'cold', 'hot'
            temperature: Numeric temperature value
        """
        context_data = {
            'time_of_day': time_of_day,
            'weather': weather,
            'temperature': temperature,
            'timestamp': datetime.now()
        }
        self.context_queue.append(context_data)
    
    def tick(self) -> Union[str, NoWork]:
        """
        Process exactly one decision/recommendation.
        Called periodically by the background runner.
        
        Returns:
            str: Coffee recommendation if work is available
            NoWork: If no work is available (no context in queue or too soon since last recommendation)
        """
        # Check if there's a pending recommendation waiting for rating
        if self.pending_recommendation is not None:
            return NoWork("Waiting for user rating on pending recommendation")
        
        # Check if enough time has passed since last recommendation
        if self.last_recommendation_time is not None:
            time_since_last = datetime.now() - self.last_recommendation_time
            if time_since_last < self.min_recommendation_interval:
                remaining_seconds = (self.min_recommendation_interval - time_since_last).total_seconds()
                return NoWork(f"Not enough time passed since last recommendation ({remaining_seconds:.1f}s remaining)")
        
        # Check if there's context in the queue
        if len(self.context_queue) == 0:
            return NoWork("No new user context in queue")
        
        # Process one context from queue
        context_data = self.context_queue.popleft()
        
        # Sense phase: Gather context
        self.sense(
            time_of_day=context_data['time_of_day'],
            weather=context_data['weather'],
            temperature=context_data['temperature']
        )
        
        # Think and Act phase: Generate recommendation
        recommended_coffee = self.think()
        
        # Update tracking
        self.last_recommendation_time = datetime.now()
        self.pending_recommendation = recommended_coffee
        
        return recommended_coffee
    
    def learn(self, coffee, rating):
        """
        LEARN phase: Update agent's knowledge based on feedback
        
        Args:
            coffee: The coffee that was recommended
            rating: User rating (1-5)
        """
        reward = rating  # Map rating directly to reward (1-5 scale)
        state = self.current_context
        
        if self.strategy == 'qlearning':
            # Q-learning update
            current_q = self.q_table[state][coffee]
            
            # For next state, use the same context (simplified)
            # In a more complex scenario, next state could be different
            next_state = state
            max_next_q = max([self.q_table[next_state][c] for c in self.coffees], default=0)
            
            # Q-learning update rule
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
            self.q_table[state][coffee] = new_q
            
        elif self.strategy == 'thompson':
            # Update Beta distribution parameters
            # Normalize rating to [0, 1] for Beta distribution
            normalized_rating = (rating - 1) / 4.0  # Maps 1-5 to 0-1
            
            # Update alpha and beta based on success/failure
            # Higher rating = more success
            self.alpha[coffee] += normalized_rating
            self.beta[coffee] += (1 - normalized_rating)
            
        elif self.strategy == 'ucb':
            # Update UCB statistics
            self.action_counts[coffee] += 1
            self.action_rewards[coffee].append(rating)
            self.total_pulls += 1
        
        # Update statistics
        self.interaction_history.append({
            'coffee': coffee,
            'rating': rating,
            'reward': reward,
            'context': state,
            'timestamp': datetime.now().isoformat()
        })
        self.total_interactions += 1
        
        # Update best coffee
        if rating > self.best_rating:
            self.best_rating = rating
            self.best_coffee = coffee
        
        # Clear pending recommendation after learning
        if self.pending_recommendation == coffee:
            self.pending_recommendation = None
    
    def get_statistics(self):
        """Get agent statistics"""
        avg_rating = np.mean([h['rating'] for h in self.interaction_history]) if self.interaction_history else 0
        
        coffee_stats = {}
        for coffee in self.coffees:
            coffee_ratings = [h['rating'] for h in self.interaction_history if h['coffee'] == coffee]
            if coffee_ratings:
                coffee_stats[coffee] = {
                    'count': len(coffee_ratings),
                    'avg_rating': np.mean(coffee_ratings),
                    'total_rating': sum(coffee_ratings)
                }
        
        return {
            'total_interactions': self.total_interactions,
            'average_rating': avg_rating,
            'best_coffee': self.best_coffee,
            'best_rating': self.best_rating,
            'coffee_stats': coffee_stats
        }
    
    def get_q_table_df(self):
        """Convert Q-table to pandas DataFrame for visualization"""
        if not self.q_table:
            return pd.DataFrame()
        
        states = list(self.q_table.keys())
        data = {}
        for coffee in self.coffees:
            data[coffee] = [self.q_table[state][coffee] for state in states]
        
        df = pd.DataFrame(data, index=states)
        return df
    
    def save_state(self, filepath='agent_state.json'):
        """Save agent state to file"""
        state = {
            'q_table': {str(k): dict(v) for k, v in self.q_table.items()},
            'interaction_history': self.interaction_history,
            'total_interactions': self.total_interactions,
            'best_coffee': self.best_coffee,
            'best_rating': self.best_rating,
            'strategy': self.strategy,
            'use_context': self.use_context
        }
        
        if self.strategy == 'thompson':
            state['alpha'] = self.alpha
            state['beta'] = self.beta
        elif self.strategy == 'ucb':
            state['action_counts'] = self.action_counts
            state['action_rewards'] = self.action_rewards
            state['total_pulls'] = self.total_pulls
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath='agent_state.json'):
        """Load agent state from file"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in state.get('q_table', {}).items():
            self.q_table[k] = defaultdict(float, v)
        
        self.interaction_history = state.get('interaction_history', [])
        self.total_interactions = state.get('total_interactions', 0)
        self.best_coffee = state.get('best_coffee')
        self.best_rating = state.get('best_rating', 0)
        
        if self.strategy == 'thompson' and 'alpha' in state:
            self.alpha = state['alpha']
            self.beta = state['beta']
        elif self.strategy == 'ucb' and 'action_counts' in state:
            self.action_counts = state['action_counts']
            self.action_rewards = {k: v for k, v in state.get('action_rewards', {}).items()}
            self.total_pulls = state.get('total_pulls', 0)
        
        return True

