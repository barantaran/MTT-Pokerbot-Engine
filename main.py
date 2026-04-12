import os
import sys
import importlib
import inspect
import csv
import json
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from engine.config import config
from engine.player_interface import Bot
from engine.tournament import Tournament

ALLOWED_MODULES = {
    'math',
    'random',
    'numpy',
    'pandas',
    'engine.player_interface',
    'engine.openai_bot_support',
}

def load_bots(players_dir="players"):
    """
    Dynamically loads all Bot subclasses from the players directory after checking for unallowed imports.
    """
    bots = []
    
    if not os.path.exists(players_dir):
        print(f"Warning: Players directory '{players_dir}' not found.")
        return bots

    for filename in os.listdir(players_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            filepath = os.path.join(players_dir, filename)
            
            # Check imports via AST
            try:
                with open(filepath, 'r') as f:
                    tree = ast.parse(f.read())
                
                safe = True
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            base_module = alias.name.split('.')[0]
                            if base_module not in ALLOWED_MODULES:
                                print(f"Security Error: Bot '{filename}' uses unallowed import '{alias.name}'. Allowed: {ALLOWED_MODULES}")
                                safe = False
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            base_module = node.module.split('.')[0]
                            full_module = node.module
                            if base_module not in ALLOWED_MODULES and full_module not in ALLOWED_MODULES:
                                print(f"Security Error: Bot '{filename}' uses unallowed import from '{node.module}'. Allowed: {ALLOWED_MODULES}")
                                safe = False
                                
                if not safe:
                    continue # Skip loading this bot

            except SyntaxError:
                print(f"Error parsing bot file {filename}")
                continue

            try:
                module = importlib.import_module(f"{players_dir}.{module_name}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Bot) and obj is not Bot:
                        bots.append(obj())
            except Exception as e:
                print(f"Failed to load bot from {filename}: {e}")
                
    return bots

def run_single_simulation(sim_id, bots):
    print(f"Starting Simulation #{sim_id}...")
    tournament = Tournament(bots)
    results, events = tournament.play()
    
    # Dump events to JSON log file
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    log_file = os.path.join(log_dir, f"sim_{sim_id}.json")
    with open(log_file, "w") as f:
         json.dump({
             "simulation_id": sim_id,
             "results": results,
             "events": events
         }, f)
         
    return sim_id, results

def main():
    print(f"--- Pokerbot MTT Engine ---")
    bots = load_bots()
    if not bots:
        print("No bots found. Exiting.")
        sys.exit(1)
        
    print(f"Loaded {len(bots)} player bots.")
    print(f"Running {config.simulation_count} simulations with a starting stack of {config.starting_stack}.")
    
    all_results = []
    
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    
    # Run multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_simulation, i+1, load_bots()): i+1 for i in range(config.simulation_count)}
        for future in as_completed(futures):
            sim_id = futures[future]
            try:
                _, results = future.result()
                print(f"Simulation #{sim_id} finished. Log written to logs/sim_{sim_id}.json.")
                for r in results:
                    r['simulation_id'] = sim_id
                all_results.extend(results)
            except Exception as e:
                 print(f"Simulation #{sim_id} generated an exception: {e}")

    # Export to CSV
    if all_results:
        csv_file = "simulation_results.csv"
        fieldnames = ["simulation_id", "position", "name", "bot_class", "payout_pct"]
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"Successfully exported {len(all_results)} results to {csv_file}")
    
if __name__ == "__main__":
    main()
