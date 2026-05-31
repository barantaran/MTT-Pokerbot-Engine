# Python Multi-Table Tournament (MTT) Pokerbot Engine

This program allows you to write your own Artificial Intelligence "Poker Bots" using Python, and pit them against each other in multi-table simulated poker tournaments. 

This engine creates an API for your player script to intereact with and make its decisions with.

This detailed guide assumes **no prior experience** with running servers, git, or advanced python. 

---

## Step 1: Installing the Prerequisites

To run this engine, you need **Python** to run the code. 

1. **Download Python:**
   - Download python from [python.org/downloads](https://www.python.org/downloads/) and make sure it's at least version 3.10.

2. **Open your Terminal (Command Prompt):**
   - Press the `Windows Key`, type `cmd`, and hit Enter. You should see a black text box pop up. (If on Mac, press Cmd+Space, type `Terminal`, and hit Enter).

---

## Step 2: Preparing the Engine

Now that Python is installed, we need to download the code and install its dependencies.

1. **Install Git:**
   * To download the code from GitHub, use Git. If you don't already have it installed, head over to [git-scm.com/downloads](https://git-scm.com/downloads) and download it.

2. **Get the Code:** * Open your Terminal (or Command Prompt) from Step 1.
   * First, tell the terminal where you want to save the project folder. For example, to navigate to your Desktop, run:
     `cd Desktop`
   * Next, download the code to your computer by running this command:
     `git clone https://github.com/Alex-Addison/MTT-Pokerbot-Engine.git`
   * Finally, move inside the newly downloaded folder:
     `cd MTT-Pokerbot-Engine`

3. **Install the Libraries:**
   * In your terminal window that is now pointing to the project folder, copy and paste this command and hit Enter:
     `python -m pip install -r requirements.txt`
   * Your computer will download the `treys`, `numpy`, and `pandas` libraries.
---

## Step 3: Getting to Know the Bots

Inside the folder, you will see a folder called `players`. This is where all bot scripts will be kept.

### Bot Rules
To ensure fairness, bots currently are strictly limited visually to exactly what you'd see sitting at a poker table. 

**Allowed Code Resources:** By default, bots are only allowed to use standard basic python and these specific libraries:
*   `math`
*   `random`
*   `numpy` 
*   `pandas`

If a bot tries to use something like `os` to cheat or steal files, the engine will automatically block it from playing.

### How to Create a Bot
Every bot is a simple python file. You can see examples in `players/random_bot.py`.
There is only one rule: Your bot must have a `get_action` function that takes in the `game_state` and returns an action.

**The Game State dictionary gives you:**
*   `hole_cards`: Your two hidden cards
*   `board_cards`: The cards on the table
*   `pot_size`: Total chips in the pot
*   `stack_size`: Your chips remaining
*   `call_amount`: Exactly how many chips you owe to stay in the hand
*   `min_raise`: The rules on what you must bet if you want to raise
*   `blinds`: The current blind level
*   `active_players`: How many people are left in the hand

**Your Bot can return one of three things:**
1.  `return ('fold', 0)` 
2.  `return ('call', 0)` (Match the current bet to see the next card. *If call_amount is 0, this acts as checking!*)
3.  `return ('raise', 200)` (Match the bet, then add 200 more chips on top of it)

**Note:** The Engine gives bots exactly **500 milliseconds (half a second) to decide**. If your bot takes too long doing complex math, the dealer will automatically fold your hand.

### Need Help Writing a Smart Bot?
Writing a great poker bot involves understanding game theory, probability, and hand strength! Here are great places to start learning how to evaluate poker mathematically:
*   [MIT's Pokerbots Course Lectures (Free)](https://pokerbots.org/)
*   [Introduction to Poker Strategy & Math](https://www.upswingpoker.com/poker-strategy/)
*   [Treys Hand Evaluator Docs](https://github.com/ihendley/treys) - Learn how your bots hole cards are formatted as numbers to do math on them.

---

## Step 4: Configuring the Tournament

There is a file called `config.json`. You can open this in Notepad or any text editor!

It controls everything about the simulation:
*   `simulation_count`: E.g., `100`. The engine will blast through 100 complete tournaments in parallel!
*   `starting_stack`: E.g., `10000`. Chips everybody starts with.
*   `hands_per_level`: Number of hands before the next level. This approximates timed live levels in deterministic simulations.
*   `max_hands_per_tournament`: Safety cap for bounded runs.
*   `blinds_schedule`: You can edit each level as `{"small": 100, "big": 200}`.
*   `bot_decision_timeout_ms`: The 500ms limit!
*   `payouts`: Set the percentages. `"1": 0.50` means 1st place gets 50% of the prize pool!

### Blind Level Speed

The engine advances blind levels by completed hand count, not by wall-clock minutes. This keeps simulations deterministic and fast, but it means `hands_per_level` is an approximation of real online MTT blind timers.

A reasonable 9-max online pace assumption is roughly 55-70 hands per hour while tables are full. With that assumption, these presets are useful starting points:

| Structure | Approx real level time | Suggested `hands_per_level` |
| --- | ---: | ---: |
| Hyper | 2-3 minutes | 3 |
| Turbo | 4-6 minutes | 6 |
| Regular | 8-10 minutes | 10 |
| Slow / deep | 12-15 minutes | 15 |

For example, `hands_per_level: 12` is closer to a regular-to-slow structure than a turbo structure. Use the same value across comparison runs when testing bot strength, and only change it intentionally when comparing performance across tournament speeds.

---

## Step 5: Run Simulation

If you're interested in running the engine using Docker, skip to Step 5 (Alternative).

In your terminal (the black box) pointing to the project folder:
1.  Type:
    `python main.py`
2.  Hit Enter!

The Engine will automatically load every single `.py` file inside the `players` folder and begin organizing them into 9-max tables. Because the engine is highly optimized utilizing "multiprocessing", it will run multiple entire tournaments simultaneously across your CPU cores.

**When it finishes:**
1.  It creates a cleanly organized `simulation_results.csv` file that you can open in Excel to analyze exactly which bot placed where!
2.  It creates incredibly detailed second-by-second logs inside the `/logs/` folder!

---

## Step 5 (Alternative): Run using Docker

If you don't want to install Python directly on your machine or want to guarantee the environment runs exactly the same everywhere, you can optionally run the engine using **Docker**.

1. **Install Docker Desktop:**
   - Download it from [docker.com](https://www.docker.com/products/docker-desktop/) and install it.
   - Keep Docker Desktop running in the background.

2. **Build and Run the Engine:**
   - Open your terminal pointing to the project folder (`MTT-Pokerbot-Engine`).
   - Run this command to build the standardized image (this packages all the dependencies automatically):
     `docker build -t mtt-pokerbot .`
   - Once it finishes building, run the simulation by typing:
     `docker run -v "${PWD}/logs:/app/logs" -v "${PWD}/simulation_results.csv:/app/simulation_results.csv" mtt-pokerbot`
     *(Note: If you are using standard Windows Command Prompt instead of PowerShell, replace `${PWD}` with `%cd%`. Mac/Linux users use `$(pwd)`)*

The engine will execute entirely inside the Docker container and automatically save the standard `simulation_results.csv` and `/logs/` back to your main folder for you to view on the visualizer.

---

## Poker AI Integration

Phase 21 wires a promoted `poker-ai-basemodel` checkpoint into the engine bot API before any larger engine simulation starts. It is a smoke test, not a tournament-quality evaluation.

From this directory:

```bash
../poker-ai-basemodel/.venv/bin/python -m engine.phase21_engine_bot_wiring --config configs/phase21_engine_bot_wiring_smoke_test.json
```

Or from the workspace root:

```bash
./scripts/phase21_engine_bot_wiring_smoke_test.sh
```

Expected result: the command reads the accepted Phase 20 promotion report, loads the promoted checkpoint through `engine/baseline_model_bot.py`, verifies visible-state privacy, 62-float observation encoding, legal engine action outputs, conservative fallback behavior, and timeout handling, then writes `runs/phase21_engine_bot_wiring_smoke_test/<run>/engine_wiring_report.json`. Phase 22 may start only when the report says `phase21_engine_wiring_status: accepted` and `small_mtt_engine_simulation_allowed: true`.

Phase 22 runs the accepted model bot through a bounded small MTT engine simulation with mixed opponents, including `RandomBot` and the equity-aware `AggressiveBot`. It is a diagnostic engine simulation, not a training phase or promotion proof.

Command from this directory:

```bash
../poker-ai-basemodel/.venv/bin/python -m engine.phase22_small_mtt_engine_simulation --config configs/phase22_small_mtt_engine_simulation.json
```

Command from the workspace root:

```bash
./scripts/phase22_small_mtt_engine_simulation.sh
```

Expected result: the command reads the accepted Phase 21 report, loads the promoted checkpoint, builds a fixed lineup with the model bot, random bots, the equity-aware aggressive bot, and fixed baseline bots, runs bounded engine tournaments, writes event logs and result artifacts under `runs/phase22_small_mtt_engine_simulation/<run>/`, and records whether larger MTT engine simulation may start.

Phase 23 expands this into a larger multi-table engine simulation campaign. It remains evaluation, not training, and decides whether Phase 24 should collect proper per-decision engine rollouts before any training update.

Command from this directory:

```bash
../poker-ai-basemodel/.venv/bin/python -m engine.phase23_larger_mtt_engine_simulation --config configs/phase23_larger_mtt_engine_simulation.json
```

Command from the workspace root:

```bash
./scripts/phase23_larger_mtt_engine_simulation.sh
```

Expected result: the command reads the accepted Phase 22 report, runs a larger mixed-opponent MTT engine campaign, writes aggregate tournament results and event summaries under `runs/phase23_larger_mtt_engine_simulation/<run>/`, and recommends whether to launch engine-backed rollout collection.

Phase 24 collects engine rollouts for retraining. It reads the accepted Phase 23 report, requires `engine_rollout_collection_allowed: true`, runs bounded mixed-opponent engine MTTs, and writes training-ready per-decision model records under `runs/phase24_engine_rollout_collection/<run>/`. The rollout artifact includes the 62-float observation, 9-way legal-action mask, selected model action id, mapped engine action, logprob/value fields, fallback diagnostics, reward fields, and terminal tournament outcomes. It is a data-collection phase, not a training phase.

Command from this directory:

```bash
../poker-ai-basemodel/.venv/bin/python -m engine.phase24_engine_rollout_collection --config configs/phase24_engine_rollout_collection.json
```

Command from the workspace root:

```bash
./scripts/phase24_engine_rollout_collection.sh
```

Expected result: the command writes `runs/phase24_engine_rollout_collection/<run>/engine_rollout_collection_report.json`, `rollouts/model_decisions.jsonl`, `rollouts/terminal_outcomes.jsonl`, and `rollouts/manifest.json`. Phase 25 training should start only when the report says `phase24_engine_rollout_collection_status: accepted` and `phase25_engine_training_allowed: true`.

Phase 26 evaluates the Phase 25 retrained candidate checkpoint against the Phase 24 source checkpoint in paired mixed-opponent engine MTT campaigns. It is an evaluation phase only; it does not promote the candidate.

Command from this directory:

```bash
../poker-ai-basemodel/.venv/bin/python -m engine.phase26_engine_retraining_evaluation --config configs/phase26_engine_retraining_evaluation.json
```

Command from the workspace root:

```bash
./scripts/phase26_engine_retraining_evaluation.sh
```

Expected result: the command reads the latest accepted Phase 25 report, runs source and candidate checkpoints with the same deterministic seeds and lineup, writes per-campaign results under `runs/phase26_engine_retraining_evaluation/<run>/campaigns/`, and writes `engine_retraining_evaluation_report.json`. Phase 27 promotion decision work should start only when the report says `phase26_engine_retraining_evaluation_status: accepted` and `phase27_promotion_decision_allowed: true`.

Phase 29 evaluates the Phase 28 reduced-observation supervised clone checkpoint through `engine/reduced_model_bot.py`. It is an evaluation phase only; it does not replace the 62-feature model bot or promote the reduced checkpoint. The reduced adapter now prefers PokerStove-backed equity and reports any fallback use.

Command from this directory:

```bash
../poker-ai-basemodel/.venv/bin/python -m engine.phase29_reduced_clone_engine_evaluation --config configs/phase29_reduced_clone_engine_evaluation.json
```

Command from the workspace root:

```bash
./scripts/phase29_reduced_clone_engine_evaluation.sh
```

Expected result: the command reads the latest accepted Phase 28 training report, loads the reduced checkpoint with `observation_size=8`, runs a PokerStove-backed mixed-opponent MTT evaluation, and writes `runs/phase29_reduced_clone_engine_evaluation/<run>/reduced_clone_engine_evaluation_report.json`. Phase 30 should start only when the report says `phase29_reduced_clone_engine_evaluation_status: accepted` and `phase30_reduced_clone_decision_allowed: true`.

---

## Step 6: Use the Visualizer

![Visualizer](visualizer.png)

1. In your terminal, type:
   `python server.py`
   and hit Enter.
2. The terminal will say `Serving at http://localhost:8000/visualizer`.
3. Open your web browser.
4. In the top URL bar, copy and paste this exact link:
   `http://localhost:8000/visualizer`
5. You are in! 
   - Click the blue "Load Simulation" button at the top.
   - Go to the `logs` folder, and select `sim_1.json`.
   - Hit the **Play ▶** button at the bottom!
   - You can increase the speed to 5x or 10x using the buttons in the corner.
   - You can see the tables coalescing dynamically. **Click on one of the round tables to zoom in** and see the actual cards being dealt, standard action bets being taken, and the side pots being grouped!
