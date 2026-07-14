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

# Load agent weights
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

# UCT Evaluator (uniform prior)
class UCTEvaluator(mcts.Evaluator):
    def __init__(self, dqn_0, dqn_1, device):
        self._dqn_0 = dqn_0
        self._dqn_1 = dqn_1
        self._device = device

    def evaluate(self, state: pyspiel.State) -> np.ndarray:
        obs_0 = np.asarray(state.observation_tensor(0))
        obs_1 = np.asarray(state.observation_tensor(1))
        t_0 = torch.tensor(obs_0, dtype=torch.float32, device=self._device).unsqueeze(0)
        t_1 = torch.tensor(obs_1, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_0 = self._dqn_0._q_network(t_0).squeeze(0).cpu().numpy()
            q_1 = self._dqn_1._q_network(t_1).squeeze(0).cpu().numpy()
            
        curr_player = state.current_player()
        legal = state.legal_actions()
        
        if curr_player == 0:
            val_0 = max(q_0[a] for a in legal) if len(legal) > 0 else 0.0
            val_1 = max(q_1)
        elif curr_player == 1:
            val_0 = max(q_0)
            val_1 = max(q_1[a] for a in legal) if len(legal) > 0 else 0.0
        else:
            val_0 = max(q_0)
            val_1 = max(q_1)
        return np.array([val_0, val_1])

    def prior(self, state: pyspiel.State) -> list:
        legal = state.legal_actions()
        return [(action, 1.0 / len(legal)) for action in legal]

# PUCT Evaluator (softmax Q-value prior)
class PUCTEvaluator(mcts.Evaluator):
    def __init__(self, dqn_0, dqn_1, device):
        self._dqn_0 = dqn_0
        self._dqn_1 = dqn_1
        self._device = device

    def evaluate(self, state: pyspiel.State) -> np.ndarray:
        # Same evaluation logic as UCT
        obs_0 = np.asarray(state.observation_tensor(0))
        obs_1 = np.asarray(state.observation_tensor(1))
        t_0 = torch.tensor(obs_0, dtype=torch.float32, device=self._device).unsqueeze(0)
        t_1 = torch.tensor(obs_1, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            q_0 = self._dqn_0._q_network(t_0).squeeze(0).cpu().numpy()
            q_1 = self._dqn_1._q_network(t_1).squeeze(0).cpu().numpy()
            
        curr_player = state.current_player()
        legal = state.legal_actions()
        
        if curr_player == 0:
            val_0 = max(q_0[a] for a in legal) if len(legal) > 0 else 0.0
            val_1 = max(q_1)
        elif curr_player == 1:
            val_0 = max(q_0)
            val_1 = max(q_1[a] for a in legal) if len(legal) > 0 else 0.0
        else:
            val_0 = max(q_0)
            val_1 = max(q_1)
        return np.array([val_0, val_1])

    def prior(self, state: pyspiel.State) -> list:
        curr_player = state.current_player()
        if curr_player not in [0, 1]:
            return state.chance_outcomes()
            
        obs = np.asarray(state.observation_tensor(curr_player))
        t_obs = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            agent = self._dqn_0 if curr_player == 0 else self._dqn_1
            q = agent._q_network(t_obs).squeeze(0).cpu().numpy()
            
        legal_actions = state.legal_actions()
        legal_q = np.array([q[a] for a in legal_actions])
        
        T = 5.0
        exp_q = np.exp((legal_q - np.max(legal_q)) / T)
        probs = exp_q / np.sum(exp_q)
        return list(zip(legal_actions, probs))

uct_eval = UCTEvaluator(ddqn_p0, ddqn_p1, device)
puct_eval = PUCTEvaluator(ddqn_p0, ddqn_p1, device)

# Bots running 200 simulations
bot_uct = mcts.MCTSBot(game=env.game, uct_c=50.0, max_simulations=200, evaluator=uct_eval, solve=True)
bot_puct = mcts.MCTSBot(game=env.game, uct_c=50.0, max_simulations=200, evaluator=puct_eval, solve=True)

num_games = 10
puct_wins = 0
uct_wins = 0
draws = 0
puct_scores = []
uct_scores = []

# Match 1: PUCT (P0) vs UCT (P1)
print("Running Match 1: PUCT (P0) vs UCT (P1)...")
for g in range(num_games):
    time_step = env.reset()
    bot_puct.restart()
    bot_uct.restart()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == 0:
            action = bot_puct.step(state)
        else:
            action = bot_uct.step(state)
        time_step = env.step([action])
    s_puct = env.get_state.returns()[0]
    s_uct = env.get_state.returns()[1]
    puct_scores.append(s_puct)
    uct_scores.append(s_uct)
    if s_puct > s_uct: puct_wins += 1
    elif s_uct > s_puct: uct_wins += 1
    else: draws += 1

# Match 2: UCT (P0) vs PUCT (P1)
print("Running Match 2: UCT (P0) vs PUCT (P1)...")
for g in range(num_games):
    time_step = env.reset()
    bot_puct.restart()
    bot_uct.restart()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == 0:
            action = bot_uct.step(state)
        else:
            action = bot_puct.step(state)
        time_step = env.step([action])
    s_uct = env.get_state.returns()[0]
    s_puct = env.get_state.returns()[1]
    puct_scores.append(s_puct)
    uct_scores.append(s_uct)
    if s_puct > s_uct: puct_wins += 1
    elif s_uct > s_puct: uct_wins += 1
    else: draws += 1

print("\n=== PUCT vs UCT TOURNAMENT RESULTS ===")
print(f"Total Games Played: {2 * num_games}")
print(f"PUCT (Softmax Prior) Wins: {puct_wins} ({(puct_wins / (2 * num_games)) * 100:.1f}%)")
print(f"UCT (Uniform Prior) Wins: {uct_wins} ({(uct_wins / (2 * num_games)) * 100:.1f}%)")
print(f"Draws: {draws}")
print(f"PUCT Average Score: {np.mean(puct_scores):.2f}")
print(f"UCT Average Score: {np.mean(uct_scores):.2f}")
