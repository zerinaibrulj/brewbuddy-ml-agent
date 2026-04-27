"""
Standalone background worker: demonstrates queue + tick() with subjective context.
"""

import os
import time
from brewbuddy_data.database import get_user_profile, init_db
from brewbuddy_agent import BrewBuddyAgent
from background_worker import BackgroundWorker


def main() -> None:
    init_db()
    print("=" * 60)
    print("BrewBuddy background worker (standalone)")
    print("=" * 60)

    prof = get_user_profile()
    agent = BrewBuddyAgent(
        coffees=None,
        use_context=True,
        use_subjective=True,
        use_hybrid=True,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        strategy="qlearning",
    )
    if os.path.exists("agent_state.json"):
        agent.load_state("agent_state.json")
        print("Loaded agent state from agent_state.json")

    worker = BackgroundWorker(agent, tick_interval=2.0)
    worker.start()

    print("\nBackground worker is running. Press Ctrl+C to stop.\n")

    print("Enqueuing sample context requests (external + subjective)...")
    agent.add_context_request(
        time_of_day="morning",
        weather="sunny",
        temperature=22.0,
        subjective={"sleep_hours": 4.0, "fatigue": 9, "lactose_intolerance": True, "social_battery": "Empty"},
        user_profile=prof,
    )
    agent.add_context_request(
        time_of_day="afternoon",
        weather="cloudy",
        temperature=18.0,
        subjective={"sleep_hours": 7.0, "fatigue": 4, "lactose_intolerance": False, "social_battery": "Full"},
        user_profile=prof,
    )
    agent.add_context_request(
        time_of_day="evening",
        weather=None,
        temperature=20.0,
        subjective={"sleep_hours": 8.0, "fatigue": 3, "lactose_intolerance": False, "social_battery": "Empty"},
        user_profile=prof,
    )

    try:
        while True:
            time.sleep(5)
            st = worker.get_status()
            latest = worker.get_latest_result()
            print(
                f"[Status] Ticks: {st['tick_count']}, latest: {latest!r} | "
                f"ml_state: {getattr(agent, 'current_ml_state', '?')}"
            )
    except KeyboardInterrupt:
        print("\nStopping background worker...")
        worker.stop()
        print("Done.")


if __name__ == "__main__":
    main()
