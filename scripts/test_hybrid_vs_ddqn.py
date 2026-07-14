import pathlib
import pyspiel
import numpy as np
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import mcts

# Initialize environment
env = rl_environment.Environment("azul", players=2)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# Load the trained Double DQN agents (for Player 0 and Player 1)
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

ddqn_p0 = load_dqn(0)
ddqn_p1 = load_dqn(1)
print("Double DQN agents loaded successfully.")

# Custom MCTS Evaluator powered by the DDQN Value Network
class DQNEvaluator(mcts.Evaluator):
    def __init__(self, dqn_0, dqn_1, device):
        self._dqn_0 = dqn_0
        self._dqn_1 = dqn_1
        self._device = device

    def evaluate(self, state: pyspiel.State) -> np.ndarray:
        # Convert state observation to PyTorch tensors
        obs_0 = np.asarray(state.observation_tensor(0))
        obs_1 = np.asarray(state.observation_tensor(1))
        
        t_0 = torch.tensor(obs_0, dtype=torch.float32, device=self._device).unsqueeze(0)
        t_1 = torch.tensor(obs_1, dtype=torch.float32, device=self._device).unsqueeze(0)
        
        with torch.no_grad():
            q_0 = self._dqn_0._q_network(t_0).squeeze(0).cpu().numpy()
            q_1 = self._dqn_1._q_network(t_1).squeeze(0).cpu().numpy()
            
        curr_player = state.current_player()
        legal = state.legal_actions()
        
        # Estimate state values
        if curr_player == 0:
            val_0 = max(q_0[a] for a in legal) if len(legal) > 0 else 0.0
            val_1 = max(q_1)  # Proxy for opponent
        elif curr_player == 1:
            val_0 = max(q_0)  # Proxy for opponent
            val_1 = max(q_1[a] for a in legal) if len(legal) > 0 else 0.0
        else:
            # Chance node
            val_0 = max(q_0)
            val_1 = max(q_1)
            
        return np.array([val_0, val_1])

    def prior(self, state: pyspiel.State) -> list:
        # Standard UCT does not require priors, return uniform distribution over legal actions
        legal = state.legal_actions()
        return [(action, 1.0 / len(legal)) for action in legal]

# Instantiate Hybrid Agents using Python MCTS and our DQNEvaluator
evaluator = DQNEvaluator(ddqn_p0, ddqn_p1, device)

# Use 20 simulations per search step for a good speed/tactical balance in Python
hybrid_bot = mcts.MCTSBot(
    game=env.game,
    uct_c=2.0,
    max_simulations=20,
    evaluator=evaluator,
    solve=True,
)

print("MCTS-DDQN Hybrid agent initialized.")

# Tournament settings
num_games_per_seat = 10
hybrid_wins = 0
ddqn_wins = 0
draws = 0
hybrid_scores = []
ddqn_scores = []

# Match 1: Hybrid as P0 (P1 is DDQN)
print("\n--- Match 1: Hybrid (Player 0) vs DDQN (Player 1) ---")
for g in range(num_games_per_seat):
    time_step = env.reset()
    hybrid_bot.restart()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == 0:
            # Hybrid Turn
            action = hybrid_bot.step(state)
        else:
            # DDQN Turn
            agent_output = ddqn_p1.step(time_step, is_evaluation=True)
            action = agent_output.action
        time_step = env.step([action])
        
    returns = env.get_state.returns()
    score_hybrid = returns[0]
    score_ddqn = returns[1]
    hybrid_scores.append(score_hybrid)
    ddqn_scores.append(score_ddqn)
    
    if score_hybrid > score_ddqn:
        hybrid_wins += 1
        result = "Hybrid Wins"
    elif score_ddqn > score_hybrid:
        ddqn_wins += 1
        result = "DDQN Wins"
    else:
        draws += 1
        result = "Draw"
    print(f"Game {g+1}: Hybrid={score_hybrid:.1f}, DDQN={score_ddqn:.1f} -> {result}")

# Match 2: DDQN as P0 (P1 is Hybrid)
print("\n--- Match 2: DDQN (Player 0) vs Hybrid (Player 1) ---")
for g in range(num_games_per_seat):
    time_step = env.reset()
    hybrid_bot.restart()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == 0:
            # DDQN Turn
            agent_output = ddqn_p0.step(time_step, is_evaluation=True)
            action = agent_output.action
        else:
            # Hybrid Turn
            action = hybrid_bot.step(state)
        time_step = env.step([action])
        
    returns = env.get_state.returns()
    score_ddqn = returns[0]
    score_hybrid = returns[1]
    hybrid_scores.append(score_hybrid)
    ddqn_scores.append(score_ddqn)
    
    if score_hybrid > score_ddqn:
        hybrid_wins += 1
        result = "Hybrid Wins"
    elif score_ddqn > score_hybrid:
        ddqn_wins += 1
        result = "DDQN Wins"
    else:
        draws += 1
        result = "Draw"
    print(f"Game {g+1}: DDQN={score_ddqn:.1f}, Hybrid={score_hybrid:.1f} -> {result}")

# Print final results
total_games = 2 * num_games_per_seat
print("\n=== TOURNAMENT RESULTS ===")
print(f"Total Games Played: {total_games}")
print(f"Hybrid MCTS-DDQN Wins: {hybrid_wins} ({(hybrid_wins / total_games)*100:.1f}%)")
print(f"Raw DDQN Wins: {ddqn_wins} ({(ddqn_wins / total_games)*100:.1f}%)")
print(f"Draws: {draws}")
print(f"Hybrid Average Score: {np.mean(hybrid_scores):.2f}")
print(f"Raw DDQN Average Score: {np.mean(ddqn_scores):.2f}")
