import pathlib
import pyspiel
import numpy as np
import torch
import pandas as pd
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import random_agent

# Initialize environment
env = rl_environment.Environment("azul", players=2)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load DQN/DDDQN agents
def load_agent(path, player_id, use_dueling=False):
    agent = dqn.DQN(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=[256, 256],
        use_double_dqn=True,
        use_dueling=use_dueling,
    )
    agent.load(path)
    return agent

print("Loading agents...")
agents_p0 = {
    "DDDQN-WinFirst-900k": load_agent("./checkpoints_dddqn_900k", 0, use_dueling=True),
    "DDDQN-WinFirst-500k": load_agent("./checkpoints_dddqn_500k", 0, use_dueling=True),
    "DQN-WinFirst-400k": load_agent("./checkpoints_win_first", 0, use_dueling=False),
    "MCTS-50": pyspiel.MCTSBot(
        game=env.game,
        evaluator=pyspiel.RandomRolloutEvaluator(1, 42),
        uct_c=2.0,
        max_simulations=50,
        max_memory_mb=1000,
        solve=True,
        seed=42,
        verbose=False,
    ),
}

agents_p1 = {
    "DDDQN-WinFirst-900k": load_agent("./checkpoints_dddqn_900k", 1, use_dueling=True),
    "DDDQN-WinFirst-500k": load_agent("./checkpoints_dddqn_500k", 1, use_dueling=True),
    "DQN-WinFirst-400k": load_agent("./checkpoints_win_first", 1, use_dueling=False),
    "MCTS-50": pyspiel.MCTSBot(
        game=env.game,
        evaluator=pyspiel.RandomRolloutEvaluator(1, 42),
        uct_c=2.0,
        max_simulations=50,
        max_memory_mb=1000,
        solve=True,
        seed=42,
        verbose=False,
    ),
}

agent_names = ["DDDQN-WinFirst-900k", "DDDQN-WinFirst-500k", "DQN-WinFirst-400k", "MCTS-50"]
num_games_per_seat = 25  # 50 games total per pairing

summary_data = []

# Tournament Loop
for i, name_a in enumerate(agent_names):
    for j, name_b in enumerate(agent_names):
        if name_a == name_b:
            continue
            
        print(f"Running Match: {name_a} (P0) vs {name_b} (P1)...")
        a_scores = []
        b_scores = []
        a_wins = 0
        b_wins = 0
        draws = 0
        
        for _ in range(num_games_per_seat):
            # Seat 0: name_a is P0, name_b is P1
            agent_0 = agents_p0[name_a]
            agent_1 = agents_p1[name_b]
            
            time_step = env.reset()
            if hasattr(agent_0, "restart"): agent_0.restart()
            if hasattr(agent_1, "restart"): agent_1.restart()
            
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                state = env.get_state
                active_agent = agent_0 if player_id == 0 else agent_1
                if isinstance(active_agent, pyspiel.MCTSBot):
                    action = active_agent.step(state)
                else:
                    action = active_agent.step(time_step, is_evaluation=True).action
                time_step = env.step([action])
                
            returns = env.get_state.returns()
            score_a = returns[0]
            score_b = returns[1]
            a_scores.append(score_a)
            b_scores.append(score_b)
            if score_a > score_b: a_wins += 1
            elif score_b > score_a: b_wins += 1
            else: draws += 1

            # Seat 1: name_b is P0, name_a is P1
            agent_0 = agents_p0[name_b]
            agent_1 = agents_p1[name_a]
            
            time_step = env.reset()
            if hasattr(agent_0, "restart"): agent_0.restart()
            if hasattr(agent_1, "restart"): agent_1.restart()
            
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                state = env.get_state
                active_agent = agent_0 if player_id == 0 else agent_1
                if isinstance(active_agent, pyspiel.MCTSBot):
                    action = active_agent.step(state)
                else:
                    action = active_agent.step(time_step, is_evaluation=True).action
                time_step = env.step([action])
                
            returns = env.get_state.returns()
            score_b = returns[0]
            score_a = returns[1]
            a_scores.append(score_a)
            b_scores.append(score_b)
            if score_a > score_b: a_wins += 1
            elif score_b > score_a: b_wins += 1
            else: draws += 1
            
        summary_data.append({
            "Agent A": name_a,
            "Agent B": name_b,
            "Wins A": a_wins,
            "Wins B": b_wins,
            "Draws": draws,
            "Avg Score A": np.mean(a_scores),
            "Avg Score B": np.mean(b_scores)
        })

# Compute Leaderboard
leaderboard = {}
for name in agent_names:
    leaderboard[name] = {"games": 0, "wins": 0, "losses": 0, "draws": 0, "score_sum": 0.0}

for match in summary_data:
    a, b = match["Agent A"], match["Agent B"]
    leaderboard[a]["games"] += (match["Wins A"] + match["Wins B"] + match["Draws"])
    leaderboard[a]["wins"] += match["Wins A"]
    leaderboard[a]["losses"] += match["Wins B"]
    leaderboard[a]["draws"] += match["Draws"]
    leaderboard[a]["score_sum"] += match["Avg Score A"] * (match["Wins A"] + match["Wins B"] + match["Draws"])
    
    leaderboard[b]["games"] += (match["Wins A"] + match["Wins B"] + match["Draws"])
    leaderboard[b]["wins"] += match["Wins B"]
    leaderboard[b]["losses"] += match["Wins A"]
    leaderboard[b]["draws"] += match["Draws"]
    leaderboard[b]["score_sum"] += match["Avg Score B"] * (match["Wins A"] + match["Wins B"] + match["Draws"])

lb_list = []
for name, stats in leaderboard.items():
    wr = (stats["wins"] + 0.5 * stats["draws"]) / stats["games"] * 100
    avg_score = stats["score_sum"] / stats["games"]
    lb_list.append({
        "Agent": name,
        "Wins": stats["wins"],
        "Losses": stats["losses"],
        "Draws": stats["draws"],
        "Win Rate %": f"{wr:.1f}%",
        "Avg Score": f"{avg_score:.1f}"
    })

df_lb = pd.DataFrame(lb_list).sort_values(by="Wins", ascending=False)
df_summary = pd.DataFrame(summary_data)

print("\n=== LEADERBOARD ===")
print(df_lb.to_string(index=False))

print("\n=== HEAD-TO-HEAD MATCHES ===")
print(df_summary.to_string(index=False))
