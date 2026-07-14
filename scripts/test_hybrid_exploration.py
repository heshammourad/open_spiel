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

# Load agents
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

class DQNEvaluator(mcts.Evaluator):
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

evaluator = DQNEvaluator(ddqn_p0, ddqn_p1, device)

# Run evaluation for uct_c = 50.0
hybrid_bot_50 = mcts.MCTSBot(
    game=env.game,
    uct_c=50.0,
    max_simulations=20,
    evaluator=evaluator,
    solve=True,
)

print("Starting evaluation for uct_c = 50.0...")
wins_50 = 0
scores_50 = []
for g in range(10):
    # Seat 0
    time_step = env.reset()
    hybrid_bot_50.restart()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == 0:
            action = hybrid_bot_50.step(state)
        else:
            action = ddqn_p1.step(time_step, is_evaluation=True).action
        time_step = env.step([action])
    s_h = env.get_state.returns()[0]
    s_d = env.get_state.returns()[1]
    scores_50.append(s_h)
    if s_h > s_d: wins_50 += 1
    
    # Seat 1
    time_step = env.reset()
    hybrid_bot_50.restart()
    while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == 0:
            action = ddqn_p0.step(time_step, is_evaluation=True).action
        else:
            action = hybrid_bot_50.step(state)
        time_step = env.step([action])
    s_d = env.get_state.returns()[0]
    s_h = env.get_state.returns()[1]
    scores_50.append(s_h)
    if s_h > s_d: wins_50 += 1

print(f"Results for uct_c = 50.0:")
print(f"Wins: {wins_50} / 20 ({wins_50/20*100:.1f}%)")
print(f"Average score: {np.mean(scores_50):.2f}")
