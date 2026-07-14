import pathlib
import pyspiel
import numpy as np
import torch
import time
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn
from open_spiel.python.algorithms import mcts

# 1. Initialize environment
env = rl_environment.Environment("azul", players=2)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# 2. Load DQN agents (Win-First checkpoint)
def load_dqn(path, player_id):
    agent = dqn.DQN(
        player_id=player_id,
        state_representation_size=info_state_size,
        num_actions=num_actions,
        hidden_layers_sizes=[256, 256],
        use_double_dqn=True,
    )
    agent.load(path)
    return agent

try:
    dqn_winfirst_p0 = load_dqn("./checkpoints_win_first", 0)
    dqn_winfirst_p1 = load_dqn("./checkpoints_win_first", 1)
    print("DQN WinFirst agents loaded successfully.")
except Exception as e:
    print(f"Error loading agents: {e}")
    # Fallback to random agents or standard init
    dqn_winfirst_p0 = dqn.DQN(0, info_state_size, num_actions, [256, 256])
    dqn_winfirst_p1 = dqn.DQN(1, info_state_size, num_actions, [256, 256])

# 3. Helper to parse observation tensor
def parse_board(obs, player_id):
    # Player 0 board starts at 31, Player 1 board starts at 93 in the 170-element tensor
    offset = 31 if player_id == 0 else 93
    score = obs[offset]
    offset += 1
    
    pattern_lines = []
    for r in range(5):
        color_onehot = obs[offset : offset + 5]
        color = -1
        for c in range(5):
            if color_onehot[c] > 0.5:
                color = c
                break
        count = obs[offset + 5]
        pattern_lines.append({"color": color, "count": int(count)})
        offset += 6
        
    wall = []
    for r in range(5):
        row = []
        for c in range(5):
            row.append(obs[offset] > 0.5)
            offset += 1
        wall.append(row)
        
    floor_line_colors = obs[offset : offset + 5]
    offset += 5
    floor_line_has_starting_player_token = obs[offset] > 0.5
    offset += 1
    
    floor_line_count = int(sum(floor_line_colors) + (1.0 if floor_line_has_starting_player_token else 0.0))
    
    return {
        "score": score,
        "pattern_lines": pattern_lines,
        "wall": wall,
        "floor_line_count": floor_line_count
    }

# 4. Strategic Heuristic Value Function
def heuristic_value(board):
    h = 0.0
    # A. Base Score (weighted 10x)
    h += board["score"] * 10.0
    
    # B. Wall placement potential (predicting score from completed/partially completed pattern lines)
    wall = board["wall"]
    for r, line in enumerate(board["pattern_lines"]):
        if line["color"] != -1:
            color = line["color"]
            col = (color + r) % 5
            
            # Check if this placement will be done at the end of the round (line is full)
            if not wall[r][col] and line["count"] == r + 1:
                # Calculate exact placement score
                left = 0
                for c in range(col - 1, -1, -1):
                    if wall[r][c]: left += 1
                    else: break
                right = 0
                for c in range(col + 1, 5):
                    if wall[r][c]: right += 1
                    else: break
                h_count = 1 + left + right
                
                up = 0
                for row_idx in range(r - 1, -1, -1):
                    if wall[row_idx][col]: up += 1
                    else: break
                down = 0
                for row_idx in range(r + 1, 5):
                    if wall[row_idx][col]: down += 1
                    else: break
                v_count = 1 + up + down
                
                points = 0
                if h_count > 1 and v_count > 1:
                    points = h_count + v_count
                elif h_count > 1:
                    points = h_count
                elif v_count > 1:
                    points = v_count
                else:
                    points = 1
                h += points * 10.0  # Equal weight to actual score
                
            elif not wall[r][col] and line["count"] > 0:
                # Fractional bonus for partially filled lines
                progress = line["count"] / (r + 1)
                h += progress * 4.0
                
    # C. Floor line penalties
    floor_count = board["floor_line_count"]
    if floor_count > 0:
        penalty = 0
        for i in range(min(floor_count, 7)):
            if i == 0 or i == 1:
                penalty += 1
            elif 2 <= i <= 4:
                penalty += 2
            else:
                penalty += 3
        h -= penalty * 10.0  # Weight penalty equally to score loss
        
    # D. End-game bonus potential
    # Row completions (+2 points)
    for r in range(5):
        if all(wall[r]):
            h += 2.0 * 10.0
    # Column completions (+7 points)
    for c in range(5):
        if all(wall[r][c] for r in range(5)):
            h += 7.0 * 10.0
    # Color completions (+10 points)
    for color in range(5):
        if all(wall[r][(color + r) % 5] for r in range(5)):
            h += 10.0 * 10.0
            
    # E. Adjacency layout density bonus
    adj_count = 0
    for r in range(5):
        for c in range(5):
            if wall[r][c]:
                if c < 4 and wall[r][c+1]: adj_count += 1
                if r < 4 and wall[r+1][c]: adj_count += 1
    h += adj_count * 2.0
    
    return h

# 5. Heuristic Evaluator
class AzulHeuristicEvaluator(mcts.Evaluator):
    def evaluate(self, state: pyspiel.State) -> np.ndarray:
        if state.is_terminal():
            return np.array(state.returns())
        obs = np.asarray(state.observation_tensor(0))
        b0 = parse_board(obs, 0)
        b1 = parse_board(obs, 1)
        
        # Returns [val_0, val_1] scaled to typical score range
        return np.array([heuristic_value(b0) / 10.0, heuristic_value(b1) / 10.0])

    def prior(self, state: pyspiel.State) -> list:
        legal = state.legal_actions()
        return [(action, 1.0 / len(legal)) for action in legal]

# 6. Cached PUCT Neural Evaluator
class CachedPUCTEvaluator(mcts.Evaluator):
    def __init__(self, dqn_0, dqn_1, device, use_cache=True):
        self._dqn_0 = dqn_0
        self._dqn_1 = dqn_1
        self._device = device
        self._use_cache = use_cache
        self._cache = {}  # key (bytes) -> (q_0, q_1)
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_q_values(self, state: pyspiel.State):
        obs_0 = np.asarray(state.observation_tensor(0))
        
        if self._use_cache:
            cache_key = obs_0.tobytes()
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
            self.cache_misses += 1

        obs_1 = np.asarray(state.observation_tensor(1))
        t_0 = torch.tensor(obs_0, dtype=torch.float32, device=self._device).unsqueeze(0)
        t_1 = torch.tensor(obs_1, dtype=torch.float32, device=self._device).unsqueeze(0)
        
        with torch.no_grad():
            q_0 = self._dqn_0._q_network(t_0).squeeze(0).cpu().numpy()
            q_1 = self._dqn_1._q_network(t_1).squeeze(0).cpu().numpy()
            
        res = (q_0, q_1)
        if self._use_cache:
            if len(self._cache) > 10000:
                self._cache.clear()
            self._cache[cache_key] = res
            
        return res

    def evaluate(self, state: pyspiel.State) -> np.ndarray:
        if state.is_terminal():
            return np.array(state.returns())
            
        q_0, q_1 = self._get_q_values(state)
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
        if state.is_chance_node():
            return state.chance_outcomes()
            
        curr_player = state.current_player()
        if curr_player not in [0, 1]:
            legal = state.legal_actions()
            return [(action, 1.0 / len(legal)) for action in legal]
            
        q_0, q_1 = self._get_q_values(state)
        q = q_0 if curr_player == 0 else q_1
        
        legal_actions = state.legal_actions()
        legal_q = np.array([q[a] for a in legal_actions])
        
        T = 5.0
        exp_q = np.exp((legal_q - np.max(legal_q)) / T)
        probs = exp_q / np.sum(exp_q)
        return list(zip(legal_actions, probs))

# 7. Wrapper class to track move times
class TimedAgent:
    def __init__(self, agent, name):
        self.agent = agent
        self.name = name
        self.total_time = 0.0
        self.move_count = 0
        
    def step(self, state, time_step=None):
        start_time = time.perf_counter()
        if isinstance(self.agent, mcts.MCTSBot):
            action = self.agent.step(state)
        elif hasattr(self.agent, "step"):
            action = self.agent.step(time_step, is_evaluation=True).action
        else:
            action = self.agent(state)
        
        elapsed = time.perf_counter() - start_time
        self.total_time += elapsed
        self.move_count += 1
        return action
        
    def restart(self):
        if hasattr(self.agent, "restart"):
            self.agent.restart()
            
    def get_avg_time_ms(self):
        if self.move_count == 0:
            return 0.0
        return (self.total_time / self.move_count) * 1000.0

# 8. Setup Tournament Agents
eval_heuristic = AzulHeuristicEvaluator()
eval_cached = CachedPUCTEvaluator(dqn_winfirst_p0, dqn_winfirst_p1, device, use_cache=True)
eval_uncached = CachedPUCTEvaluator(dqn_winfirst_p0, dqn_winfirst_p1, device, use_cache=False)

bots = {
    "DQN-Raw": dqn_winfirst_p0, # Will be mapped dynamically to p0/p1 in the loop
    "Heuristic-MCTS-100": mcts.MCTSBot(game=env.game, uct_c=50.0, max_simulations=100, evaluator=eval_heuristic, solve=True),
    "Heuristic-MCTS-300": mcts.MCTSBot(game=env.game, uct_c=50.0, max_simulations=300, evaluator=eval_heuristic, solve=True),
    "Cached-PUCT-MCTS-100": mcts.MCTSBot(game=env.game, uct_c=50.0, max_simulations=100, evaluator=eval_cached, solve=True),
    "Uncached-PUCT-MCTS-100": mcts.MCTSBot(game=env.game, uct_c=50.0, max_simulations=100, evaluator=eval_uncached, solve=True),
}

# Run a small Round Robin tournament
pairings = [
    ("Heuristic-MCTS-300", "DQN-Raw"),
    ("Cached-PUCT-MCTS-100", "DQN-Raw"),
    ("Cached-PUCT-MCTS-100", "Uncached-PUCT-MCTS-100"),
    ("Heuristic-MCTS-300", "Cached-PUCT-MCTS-100"),
]

num_games_per_pairing = 50
results = []

print("\n--- Starting Strategy Validation Tournament ---")
for p1_name, p2_name in pairings:
    print(f"\nMatchup: {p1_name} vs {p2_name}")
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    # Track times specifically for this matchup
    agent1_wrapper = TimedAgent(bots[p1_name] if p1_name != "DQN-Raw" else dqn_winfirst_p0, p1_name)
    agent2_wrapper = TimedAgent(bots[p2_name] if p2_name != "DQN-Raw" else dqn_winfirst_p1, p2_name)
    
    for g in range(num_games_per_pairing):
        # Swap seats alternate games
        if g % 2 == 0:
            a0, a1 = agent1_wrapper, agent2_wrapper
            name_0, name_1 = p1_name, p2_name
        else:
            a0, a1 = agent2_wrapper, agent1_wrapper
            name_0, name_1 = p2_name, p1_name
            
        # Re-map DQN player IDs depending on seat assignment
        if name_0 == "DQN-Raw":
            a0.agent = dqn_winfirst_p0
        if name_1 == "DQN-Raw":
            a1.agent = dqn_winfirst_p1
            
        time_step = env.reset()
        a0.restart()
        a1.restart()
        
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            state = env.get_state
            
            if player_id == 0:
                action = a0.step(state, time_step)
            else:
                action = a1.step(state, time_step)
                
            time_step = env.step([action])
            
        returns = env.get_state.returns()
        score_0 = returns[0]
        score_1 = returns[1]
        
        # Attribute wins correctly to original p1 / p2 names
        if g % 2 == 0:
            s_p1, s_p2 = score_0, score_1
        else:
            s_p1, s_p2 = score_1, score_0
            
        if s_p1 > s_p2:
            p1_wins += 1
            res_str = f"{p1_name} Wins"
        elif s_p2 > s_p1:
            p2_wins += 1
            res_str = f"{p2_name} Wins"
        else:
            draws += 1
            res_str = "Draw"
            
        if (g + 1) % 10 == 0 or g == 0:
            print(f"  Progress: {g+1}/{num_games_per_pairing} games completed (Last game: {p1_name}={s_p1:.1f}, {p2_name}={s_p2:.1f} -> {res_str})")
        
    print(f"Matchup Results:")
    print(f"  {p1_name} Wins: {p1_wins} ({(p1_wins/num_games_per_pairing)*100:.1f}%)")
    print(f"  {p2_name} Wins: {p2_wins} ({(p2_wins/num_games_per_pairing)*100:.1f}%)")
    print(f"  Draws: {draws}")
    print(f"  {p1_name} Avg Move Time: {agent1_wrapper.get_avg_time_ms():.2f} ms")
    print(f"  {p2_name} Avg Move Time: {agent2_wrapper.get_avg_time_ms():.2f} ms")

print("\n--- Cache Performance Statistics ---")
print(f"Cache Hits: {eval_cached.cache_hits}")
print(f"Cache Misses: {eval_cached.cache_misses}")
hit_ratio = (eval_cached.cache_hits / (eval_cached.cache_hits + eval_cached.cache_misses)) * 100 if (eval_cached.cache_hits + eval_cached.cache_misses) > 0 else 0
print(f"Cache Hit Rate: {hit_ratio:.2f}%")
