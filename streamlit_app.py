"""
BrewBuddy - AI Agent for Personalized Coffee Recommendations
Streamlit Application with Q-Learning and Context-Aware Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from brewbuddy_agent import BrewBuddyAgent, NoWork
from background_worker import BackgroundWorker
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="BrewBuddy AI ☕",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 7.5rem;
        font-weight: 900;
        color: #000000;
        margin-bottom: 0.5rem;
        margin-top: 0;
        line-height: 1.1;
        letter-spacing: -0.015em;
        font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
        margin-top: 0;
        text-align: center;
    }
    .coffee-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    # Define coffee options
    coffee_list = [
        "Espresso", "Cappuccino", "Latte", "Americano", 
        "Mocha", "Macchiato", "Flat White", "Cortado",
        "Cold Brew", "Iced Coffee", "Frappuccino", "Decaf"
    ]
    
    # Initialize agent
    st.session_state.agent = BrewBuddyAgent(
        coffees=coffee_list,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        use_context=True,
        strategy='qlearning'
    )
    
    # Load saved state if exists
    if os.path.exists('agent_state.json'):
        st.session_state.agent.load_state('agent_state.json')

# Initialize background worker
if 'background_worker' not in st.session_state:
    st.session_state.background_worker = BackgroundWorker(
        agent=st.session_state.agent,
        tick_interval=2.0  # Call tick() every 2 seconds
    )
    st.session_state.background_worker.start()

if 'current_recommendation' not in st.session_state:
    st.session_state.current_recommendation = None

if 'last_rating' not in st.session_state:
    st.session_state.last_rating = None

if 'worker_status' not in st.session_state:
    st.session_state.worker_status = None

if 'last_pending_recommendation' not in st.session_state:
    st.session_state.last_pending_recommendation = None

# Coffee descriptions for better UX
COFFEE_DESCRIPTIONS = {
    "Espresso": "Strong, concentrated coffee with rich flavor",
    "Cappuccino": "Espresso with steamed milk and foam",
    "Latte": "Smooth espresso with lots of steamed milk",
    "Americano": "Espresso diluted with hot water",
    "Mocha": "Chocolate-flavored espresso drink",
    "Macchiato": "Espresso with a dollop of foamed milk",
    "Flat White": "Espresso with microfoam, stronger than latte",
    "Cortado": "Equal parts espresso and warm milk",
    "Cold Brew": "Smooth, cold-steeped coffee",
    "Iced Coffee": "Chilled coffee served over ice",
    "Frappuccino": "Blended iced coffee drink",
    "Decaf": "Decaffeinated coffee option"
}

# Coffee name to image file mapping
def get_coffee_image_path(coffee_name):
    """Map coffee name to image file path"""
    # Normalize coffee name to match file naming (lowercase, handle spaces)
    normalized = coffee_name.lower()
    
    # Map coffee names to their image files
    image_mapping = {
        "espresso": "espresso.jpg",
        "cappuccino": "cappuccino.jpg",
        "latte": "latte.webp",
        "americano": "americano.jpg",
        "mocha": "mocha.png",
        "macchiato": "macchiato.jpg",
        "flat white": "flat white.jpg",
        "cortado": "cortado.webp",
        "cold brew": "cold brew.jpg",
        "iced coffee": "iced coffee.jpg",
        "frappuccino": "frappuccino.jpg",
        "decaf": "decaf.webp"
    }
    
    image_file = image_mapping.get(normalized)
    if image_file and os.path.exists(f"images/{image_file}"):
        return f"images/{image_file}"
    return None

# Main Header with Logo - Centered Layout
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])
with header_col1:
    # Logo on the left
    if os.path.exists("images/coffee1.png"):
        logo_image = Image.open("images/coffee1.png")
        st.image(logo_image, width=80)
    else:
        st.write("")  # Placeholder if image not found

with header_col2:
    # Centered title
    st.markdown('<h1 class="main-header">BrewBuddy AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Intelligent Barista - Learning Your Coffee Preferences</p>', unsafe_allow_html=True)

with header_col3:
    st.write("")  # Spacer for balance

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    # Strategy selection
    strategy = st.selectbox(
        "Learning Strategy",
        ["qlearning", "thompson", "ucb"],
        index=0,
        help="Q-Learning: Full RL with states. Thompson Sampling: Bayesian MAB. UCB: Upper Confidence Bound"
    )
    
    if st.session_state.agent.strategy != strategy:
        st.session_state.agent.strategy = strategy
        if strategy == 'thompson':
            st.session_state.agent.alpha = {coffee: 1.0 for coffee in st.session_state.agent.coffees}
            st.session_state.agent.beta = {coffee: 1.0 for coffee in st.session_state.agent.coffees}
        elif strategy == 'ucb':
            st.session_state.agent.action_counts = {coffee: 0 for coffee in st.session_state.agent.coffees}
            st.session_state.agent.action_rewards = {coffee: [] for coffee in st.session_state.agent.coffees}
            st.session_state.agent.total_pulls = 0
    
    # Hyperparameters
    st.subheader("Hyperparameters")
    learning_rate = st.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01)
    discount_factor = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9, 0.01)
    epsilon = st.slider("Exploration Rate (ε)", 0.0, 1.0, 0.3, 0.05)
    
    st.session_state.agent.learning_rate = learning_rate
    st.session_state.agent.discount_factor = discount_factor
    st.session_state.agent.epsilon = epsilon
    
    # Context settings
    st.subheader("Context Settings")
    use_context = st.checkbox("Use Context-Aware States", value=True)
    st.session_state.agent.use_context = use_context
    
    # Manual context input
    if use_context:
        time_of_day = st.selectbox(
            "Time of Day",
            ["morning", "afternoon", "evening", "night"],
            index=0
        )
        weather = st.selectbox(
            "Weather",
            [None, "sunny", "rainy", "cloudy", "cold", "hot"],
            index=0
        )
        temperature = st.slider("Temperature (°C)", 0, 40, 20, 1)
    else:
        time_of_day = None
        weather = None
        temperature = None
    
    st.divider()
    
    # Save/Load
    if st.button("💾 Save Agent State"):
        st.session_state.agent.save_state()
        st.success("Agent state saved!")
    
    if st.button("🔄 Reset Agent"):
        coffee_list = st.session_state.agent.coffees
        # Stop old worker
        if 'background_worker' in st.session_state:
            st.session_state.background_worker.stop()
        # Create new agent
        st.session_state.agent = BrewBuddyAgent(
            coffees=coffee_list,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            use_context=use_context,
            strategy=strategy
        )
        # Create new worker with new agent
        st.session_state.background_worker = BackgroundWorker(
            agent=st.session_state.agent,
            tick_interval=2.0
        )
        st.session_state.background_worker.start()
        st.session_state.current_recommendation = None
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Get Your Coffee Recommendation")
    
    # Display background worker status
    worker_status = st.session_state.background_worker.get_status()
    if worker_status['running']:
        st.success(f"🔄 Background worker running (ticks: {worker_status['tick_count']})")
    
    # Sense phase: Gather context (for display purposes)
    context = st.session_state.agent.sense(
        time_of_day=time_of_day if use_context else None,
        weather=weather if use_context else None,
        temperature=temperature if use_context else None
    )
    
    if use_context:
        st.info(f"📊 Current Context: **{context.replace('_', ' ').title()}**")
    
    # Request recommendation button - adds context to queue instead of directly calling act()
    if st.button("☕ Request Coffee Recommendation", type="primary", use_container_width=True):
        # Add context request to queue for background worker to process
        st.session_state.agent.add_context_request(
            time_of_day=time_of_day if use_context else None,
            weather=weather if use_context else None,
            temperature=temperature if use_context else None
        )
        st.info("✅ Request added to queue. Background worker will process it automatically.")
        st.rerun()
    
    # Check for pending recommendation in agent (source of truth)
    # This ensures we always display the recommendation waiting for rating
    current_pending = st.session_state.agent.pending_recommendation
    
    # Detect if pending_recommendation has changed
    if current_pending != st.session_state.last_pending_recommendation:
        # Pending recommendation changed - update UI state
        old_pending = st.session_state.last_pending_recommendation
        st.session_state.last_pending_recommendation = current_pending
        if current_pending is not None:
            # New recommendation available - update display
            st.session_state.current_recommendation = current_pending
            st.session_state.last_rating = None
            # Force UI update by clearing any cached state
            if old_pending is not None and old_pending != current_pending:
                # Recommendation changed - ensure UI refreshes
                pass
    
    if current_pending is not None:
        # Use pending recommendation as the source of truth
        display_coffee = current_pending
        # Ensure session state is in sync
        if st.session_state.current_recommendation != display_coffee:
            st.session_state.current_recommendation = display_coffee
            st.session_state.last_rating = None
    else:
        # No pending recommendation - check for new ones from worker
        latest_result = st.session_state.background_worker.get_latest_result()
        if latest_result is not None and not isinstance(latest_result, NoWork):
            # New recommendation available from background worker
            if st.session_state.current_recommendation != latest_result:
                st.session_state.current_recommendation = latest_result
                st.session_state.last_rating = None
                display_coffee = latest_result
            else:
                display_coffee = st.session_state.current_recommendation
        elif isinstance(latest_result, NoWork):
            # Display NoWork status only if no recommendation to show
            if st.session_state.current_recommendation is None:
                st.info(f"⏳ {latest_result.reason}")
            display_coffee = st.session_state.current_recommendation
        else:
            display_coffee = st.session_state.current_recommendation
    
    # Display recommendation
    if display_coffee:
        coffee_name = display_coffee
        coffee_image_path = get_coffee_image_path(coffee_name)
        
        # Display coffee image and information
        # The image will update automatically when coffee_name changes
        if coffee_image_path and os.path.exists(coffee_image_path):
            coffee_image = Image.open(coffee_image_path)
            st.image(coffee_image, width=400, use_container_width=True)
        
        st.markdown(f"""
        <div class="coffee-card" id="coffee-{coffee_name}">
            <h2 style="color: #8B4513; text-align: center;">{coffee_name}</h2>
            <p style="text-align: center; color: #666; font-size: 1.1rem;">
                {COFFEE_DESCRIPTIONS.get(coffee_name, "A delicious coffee option")}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Rating interface
        st.subheader("Rate Your Experience")
        rating = st.slider(
            "How much did you enjoy this coffee?",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key="rating_slider"
        )
        
        # Display rating stars
        stars = "⭐" * rating
        st.markdown(f"<h3 style='text-align: center;'>{stars}</h3>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ Submit Rating", type="primary", use_container_width=True):
                # Learn from feedback (this will clear pending_recommendation)
                st.session_state.agent.learn(coffee_name, rating)
                st.session_state.last_rating = rating
                st.session_state.agent.save_state()  # Auto-save
                st.success(f"Thank you! Your rating of {rating}/5 has been recorded.")
                # Clear current recommendation and tracking to allow new one
                # Note: agent.learn() already cleared pending_recommendation
                st.session_state.current_recommendation = None
                st.session_state.last_pending_recommendation = None
                st.rerun()
        
        with col_btn2:
            if st.button("🔄 Request New Recommendation", use_container_width=True):
                # Add new context request to queue
                st.session_state.agent.add_context_request(
                    time_of_day=time_of_day if use_context else None,
                    weather=weather if use_context else None,
                    temperature=temperature if use_context else None
                )
                st.session_state.current_recommendation = None
                st.info("✅ New request added to queue.")
                st.rerun()
    
    # Show last interaction feedback
    if st.session_state.last_rating and st.session_state.current_recommendation:
        st.info(f"📝 Last rating: {st.session_state.last_rating}/5 for {st.session_state.current_recommendation}")

with col2:
    st.header("📊 Statistics")
    stats = st.session_state.agent.get_statistics()
    
    # Key metrics
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Total Interactions", stats['total_interactions'])
    with metric_col2:
        st.metric("Avg Rating", f"{stats['average_rating']:.2f}")
    
    if stats['best_coffee']:
        st.success(f"🏆 Best Coffee: **{stats['best_coffee']}** ({stats['best_rating']}/5)")

# Detailed Statistics Section
st.divider()
st.header("📈 Learning Progress & Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Q-Table", "Coffee Performance", "Learning Curve", "Context Analysis"])

with tab1:
    st.subheader("Q-Table Visualization")
    if st.session_state.agent.strategy == 'qlearning':
        q_df = st.session_state.agent.get_q_table_df()
        if not q_df.empty:
            # Heatmap
            fig = px.imshow(
                q_df.T,
                labels=dict(x="State", y="Coffee", color="Q-Value"),
                aspect="auto",
                color_continuous_scale="YlOrBr",
                title="Q-Table Heatmap"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            st.dataframe(q_df.style.background_gradient(cmap='YlOrBr', axis=None), use_container_width=True)
        else:
            st.info("Q-table is empty. Start getting recommendations and rating them!")
    else:
        st.info(f"Q-table visualization is only available for Q-Learning strategy. Current strategy: {st.session_state.agent.strategy}")

with tab2:
    st.subheader("Coffee Performance Analysis")
    stats = st.session_state.agent.get_statistics()
    coffee_stats = stats['coffee_stats']
    
    if coffee_stats:
        # Prepare data for visualization
        coffee_names = []
        avg_ratings = []
        counts = []
        
        for coffee, data in coffee_stats.items():
            coffee_names.append(coffee)
            avg_ratings.append(data['avg_rating'])
            counts.append(data['count'])
        
        df_perf = pd.DataFrame({
            'Coffee': coffee_names,
            'Average Rating': avg_ratings,
            'Number of Tries': counts
        })
        
        # Bar chart
        fig = px.bar(
            df_perf,
            x='Coffee',
            y='Average Rating',
            color='Number of Tries',
            color_continuous_scale="YlOrBr",
            title="Average Rating by Coffee",
            text='Average Rating'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(df_perf.sort_values('Average Rating', ascending=False), use_container_width=True)
    else:
        st.info("No ratings yet. Start rating coffees to see performance analysis!")

with tab3:
    st.subheader("Learning Curve")
    history = st.session_state.agent.interaction_history
    
    if history:
        # Prepare data
        df_history = pd.DataFrame(history)
        df_history['cumulative_avg'] = df_history['rating'].expanding().mean()
        df_history['interaction'] = range(1, len(df_history) + 1)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history['interaction'],
            y=df_history['rating'],
            mode='markers+lines',
            name='Individual Rating',
            marker=dict(size=8, color='lightblue')
        ))
        fig.add_trace(go.Scatter(
            x=df_history['interaction'],
            y=df_history['cumulative_avg'],
            mode='lines',
            name='Cumulative Average',
            line=dict(width=3, color='orange')
        ))
        fig.update_layout(
            title="Learning Progress Over Time",
            xaxis_title="Interaction Number",
            yaxis_title="Rating",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent interactions
        st.subheader("Recent Interactions")
        recent_df = pd.DataFrame(history[-10:])[['coffee', 'rating', 'context', 'timestamp']]
        recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No interaction history yet. Start rating coffees to see the learning curve!")

with tab4:
    st.subheader("Context-Aware Analysis")
    if st.session_state.agent.use_context and st.session_state.agent.interaction_history:
        # Group by context
        df_context = pd.DataFrame(st.session_state.agent.interaction_history)
        context_stats = df_context.groupby('context').agg({
            'rating': ['mean', 'count'],
            'coffee': lambda x: x.mode()[0] if len(x) > 0 else None
        }).reset_index()
        context_stats.columns = ['Context', 'Avg Rating', 'Count', 'Most Popular']
        
        # Visualization
        fig = px.bar(
            context_stats,
            x='Context',
            y='Avg Rating',
            color='Count',
            color_continuous_scale="YlOrBr",
            title="Average Rating by Context",
            text='Avg Rating'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(context_stats, use_container_width=True)
    else:
        st.info("Context analysis requires context-aware mode and interaction history.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>BrewBuddy AI - Powered by Reinforcement Learning</p>
    <p>Sense → Think → Act → Learn</p>
</div>
""", unsafe_allow_html=True)

