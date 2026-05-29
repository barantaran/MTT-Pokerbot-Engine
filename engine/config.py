import json
import os


class Config:
    def __init__(self, config_file="config.json"):
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_file)
        self.load()

    def load(self):
        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
                
            self.simulation_count = data.get("simulation_count", 1)
            self.bot_decision_timeout_ms = data.get("bot_decision_timeout_ms", 500)
            self.starting_stack = data.get("starting_stack", 1500)
            self.max_players_per_table = data.get("max_players_per_table", 9)
            self.hands_per_level = data.get("hands_per_level", 20)
            self.blinds_schedule = data.get("blinds_schedule", [{"small": 10, "big": 20}])
            self.max_hands_per_tournament = data.get("max_hands_per_tournament", None)
            
            # Payouts is a dictionary mapping placement -> percentage
            # We convert the keys to integers and percentages to floats
            payouts_raw = data.get("payouts", {"1": 1.0})
            self.payouts = {int(k): float(v) for k, v in payouts_raw.items()}

        except FileNotFoundError:
            raise RuntimeError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Error decoding JSON configuration from {self.config_path}")

config = Config()
