"""
Standalone entry point for running the background worker independently of Streamlit UI.
This demonstrates that the agent can operate autonomously without any UI interaction.
"""

import time
from brewbuddy_agent import BrewBuddyAgent
from background_worker import BackgroundWorker


def main():
    """Main entry point for standalone background worker."""
    print("=" * 60)
    print("BrewBuddy Background Worker - Standalone Mode")
    print("=" * 60)
    
    # Define coffee options
    coffee_list = [
        "Espresso", "Cappuccino", "Latte", "Americano", 
        "Mocha", "Macchiato", "Flat White", "Cortado",
        "Cold Brew", "Iced Coffee", "Frappuccino", "Decaf"
    ]
    
    # Initialize agent
    agent = BrewBuddyAgent(
        coffees=coffee_list,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        use_context=True,
        strategy='qlearning'
    )
    
    # Load saved state if exists
    import os
    if os.path.exists('agent_state.json'):
        agent.load_state('agent_state.json')
        print("✓ Loaded agent state from agent_state.json")
    
    # Create and start background worker
    worker = BackgroundWorker(agent, tick_interval=2.0)
    worker.start()
    
    print("\nBackground worker is running autonomously...")
    print("The agent will process recommendations independently of any UI.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        # Simulate adding some context requests
        print("Adding sample context requests to queue...")
        agent.add_context_request(time_of_day='morning', weather='sunny', temperature=22)
        agent.add_context_request(time_of_day='afternoon', weather='cloudy', temperature=18)
        agent.add_context_request(time_of_day='evening', weather=None, temperature=20)
        
        # Run for a while, showing status
        while True:
            time.sleep(5)
            status = worker.get_status()
            latest = worker.get_latest_result()
            
            print(f"\n[Status] Ticks: {status['tick_count']}, "
                  f"Latest: {latest}")
            
            # Optionally add more context requests periodically
            # This simulates autonomous operation
            
    except KeyboardInterrupt:
        print("\n\nStopping background worker...")
        worker.stop()
        print("Background worker stopped.")
        print("\nAgent state saved.")


if __name__ == "__main__":
    main()

