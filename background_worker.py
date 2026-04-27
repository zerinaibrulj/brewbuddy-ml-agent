"""
Background Worker for BrewBuddy Agent
Runs autonomously and periodically calls agent.tick() to process recommendations
"""

import threading
import time
from datetime import datetime
from typing import Optional
from brewbuddy_agent import BrewBuddyAgent, NoWork


class BackgroundWorker:
    """
    Background worker that runs independently of the UI.
    Periodically calls agent.tick() to process recommendations.
    """
    
    def __init__(self, agent: BrewBuddyAgent, tick_interval: float = 2.0):
        """
        Initialize the background worker.
        
        Args:
            agent: The BrewBuddyAgent instance to run
            tick_interval: Time in seconds between tick() calls (default: 2.0)
        """
        self.agent = agent
        self.tick_interval = tick_interval
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.latest_result = None
        self.latest_result_time = None
        self.tick_count = 0
        
    def start(self):
        """Start the background worker thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
        print(f"Background worker started (tick interval: {self.tick_interval}s)")
    
    def stop(self):
        """Stop the background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        print("Background worker stopped")
    
    def _run_loop(self):
        """
        Main loop that runs in the background thread.
        Periodically calls agent.tick() and handles results.
        """
        while self.running:
            try:
                # Call tick() to process one decision/recommendation
                result = self.agent.tick()
                self.tick_count += 1
                
                # Store the latest result
                self.latest_result = result
                self.latest_result_time = datetime.now()
                
                # Handle result
                if isinstance(result, NoWork):
                    # Agent is active but has no work - this is expected behavior
                    pass
                else:
                    # Got a recommendation
                    print(f"[Worker] Generated recommendation: {result}")
                
            except Exception as e:
                print(f"[Worker] Error in tick(): {e}")
            
            # Sleep before next tick
            time.sleep(self.tick_interval)
    
    def get_latest_result(self):
        """
        Get the latest result from the worker.
        
        Returns:
            Latest result (recommendation string or NoWork instance)
        """
        return self.latest_result
    
    def get_status(self):
        """
        Get worker status information.
        
        Returns:
            dict with status information
        """
        return {
            'running': self.running,
            'tick_count': self.tick_count,
            'latest_result': str(self.latest_result),
            'latest_result_time': self.latest_result_time.isoformat() if self.latest_result_time else None,
            'tick_interval': self.tick_interval
        }

