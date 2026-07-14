import pathlib
import pyspiel
import numpy as np
import torch
import pandas as pd
from contextlib import ExitStack
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.pytorch import nfsp
from open_spiel.python.algorithms import random_agent

# Initialize environment
env = rl_environment.Environment("azul", players=2)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dqn(player_id):
    agent = dqn.DQN(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=[256, 256],
        use_double_dqn=True,
    )
    agent.load("./checkpoints_double_256")
    return agent

def load_nfsp(player_id):
    agent = nfsp.NFSP(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=[256, 256],
        reservoir_buffer_capacity=2000000,
        anticipatory_param=0.1,
        optimizer_str="adam",
        device=device,
    )
    agent.restore(pathlib.Path("./checkpoints_nfsp_256") / f"agent_{player_id}")
    return agent

print("Loading agents...")
agents_p0 = {
    "Random": random_agent.RandomAgent(player_id=0, num_actions=num_actions),
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
    "DDQN": load_dqn(0),
    "NFSP-Avg": load_nfsp(0),
    "NFSP-BR": load_nfsp(0)  # Same agent, mode switched during execution
}

agents_p1 = {
    "Random": random_agent.RandomAgent(player_id=1, num_actions=num_actions),
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
    "DDQN": load_dqn(1),
    "NFSP-Avg": load_nfsp(1),
    "NFSP-BR": load_nfsp(1)
}
print("Agents loaded successfully.")

agent_names = ["Random", "MCTS-50", "DDQN", "NFSP-Avg", "NFSP-BR"]
num_games_per_seat = 25

# Stats tracking: keys are (Agent A, Agent B) -> list of results for A vs B
match_results = {}

for i, name_a in enumerate(agent_names):
    for j, name_b in enumerate(agent_names):
        if name_a == name_b:
            continue
        
        print(f"Running match: {name_a} (P0) vs {name_b} (P1)...")
        p0_agent = agents_p0[name_a]
        p1_agent = agents_p1[name_b]
        
        a_scores = []
        b_scores = []
        a_wins = 0
        b_wins = 0
        draws = 0
        
        for _ in range(num_games_per_seat):
            time_step = env.reset()
            if hasattr(p0_agent, "restart"):
                p0_agent.restart()
            if hasattr(p1_agent, "restart"):
                p1_agent.restart()
                
            with ExitStack() as stack:
                if name_a == "NFSP-Avg":
                    stack.enter_context(p0_agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY))
                elif name_a == "NFSP-BR":
                    stack.enter_context(p0_agent.temp_mode_as(nfsp.MODE.BEST_RESPONSE))
                    
                if name_b == "NFSP-Avg":
                    stack.enter_context(p1_agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY))
                elif name_b == "NFSP-BR":
                    stack.enter_context(p1_agent.temp_mode_as(nfsp.MODE.BEST_RESPONSE))
                
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    state = env.get_state
                    if player_id == 0:
                        if name_a == "MCTS-50":
                            action = p0_agent.step(state)
                        else:
                            agent_output = p0_agent.step(time_step, is_evaluation=True)
                            action = agent_output.action
                    else:
                        if name_b == "MCTS-50":
                            action = p1_agent.step(state)
                        else:
                            agent_output = p1_agent.step(time_step, is_evaluation=True)
                            action = agent_output.action
                    time_step = env.step([action])
                
            returns = env.get_state.returns()
            score_a = returns[0]
            score_b = returns[1]
            a_scores.append(score_a)
            b_scores.append(score_b)
            
            if score_a > score_b:
                a_wins += 1
            elif score_b > score_a:
                b_wins += 1
            else:
                draws += 1
                
        # Record results
        match_results[(name_a, name_b)] = {
            "wins": a_wins,
            "losses": b_wins,
            "draws": draws,
            "scores_a": a_scores,
            "scores_b": b_scores
        }

# Compile tournament summary
print("\n--- Compile Summary ---")
summary_data = []
leaderboard = {name: {"wins": 0, "losses": 0, "draws": 0, "score_sum": 0.0, "games": 0} for name in agent_names}

for name_a in agent_names:
    for name_b in agent_names:
        if name_a == name_b:
            continue
        
        # Seat 1: A as P0, B as P1
        res1 = match_results[(name_a, name_b)]
        # Seat 2: B as P0, A as P1
        res2 = match_results[(name_b, name_a)]
        
        total_games = 2 * num_games_per_seat
        a_win_sum = res1["wins"] + res2["losses"]
        b_win_sum = res1["losses"] + res2["wins"]
        draw_sum = res1["draws"] + res2["draws"]
        
        a_score_avg = np.mean(res1["scores_a"] + res2["scores_b"])
        b_score_avg = np.mean(res1["scores_b"] + res2["scores_a"])
        
        summary_data.append({
            "Agent A": name_a,
            "Agent B": name_b,
            "Wins A": a_win_sum,
            "Wins B": b_win_sum,
            "Draws": draw_sum,
            "Avg Score A": f"{a_score_avg:.1f}",
            "Avg Score B": f"{b_score_avg:.1f}",
        })
        
        # Add to leaderboard
        leaderboard[name_a]["wins"] += a_win_sum
        leaderboard[name_a]["losses"] += b_win_sum
        leaderboard[name_a]["draws"] += draw_sum
        leaderboard[name_a]["score_sum"] += np.sum(res1["scores_a"] + res2["scores_b"])
        leaderboard[name_a]["games"] += total_games

# Create leaderboard dataframe
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

def df_to_markdown_custom(df):
    headers = list(df.columns)
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(val) for val in row) + " |\n"
    return markdown

df_lb = pd.DataFrame(lb_list).sort_values(by="Wins", ascending=False)
df_summary = pd.DataFrame(summary_data)

print("\n=== LEADERBOARD ===")
print(df_lb.to_string(index=False))

print("\n=== HEAD-TO-HEAD MATCHES ===")
print(df_summary.to_string(index=False))

# Save tournament results to artifacts
artifact_path = pathlib.Path("C:/Users/hesha/.gemini/antigravity-cli/brain/8f4d3287-6e04-4156-bd13-c4807a3aa233/tournament_results.md")
with open(artifact_path, "w") as f:
    f.write("# Azul Agent Tournament Results\n\n")
    f.write("A round-robin tournament of 50 head-to-head games per pair (25 games in each seat).\n\n")
    f.write("## Leaderboard\n\n")
    f.write(df_to_markdown_custom(df_lb) + "\n\n")
    f.write("## Head-to-Head Matches\n\n")
    f.write(df_to_markdown_custom(df_summary) + "\n")

print(f"\nTournament results saved to artifact: {artifact_path}")
