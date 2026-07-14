# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visual Web Application to play Azul against DQN, MCTS, or Random bots."""

import os
import streamlit as st
import numpy as np
import torch
import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

def log_debug(msg):
  try:
    with open("web_app_debug.log", "a") as f:
      f.write(f"{msg}\n")
  except Exception:
    pass

# Page Config
st.set_page_config(
    page_title="Azul Playroom",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme Toggle State
if "theme" not in st.session_state:
  st.session_state.theme = "dark"

def toggle_theme():
  st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# CSS Variables mapping light/dark
BG = "#09090b" if IS_DARK else "#ffffff"
BG_SUBTLE = "#0c0c0f" if IS_DARK else "#f9fafb"
CARD = "#0c0c0f" if IS_DARK else "#ffffff"
CARD_HOVER = "#131316" if IS_DARK else "#f4f4f5"
BORDER = "#1e1e24" if IS_DARK else "#e4e4e7"
BORDER_SUBTLE = "#16161a" if IS_DARK else "#f0f0f2"
TEXT = "#fafafa" if IS_DARK else "#09090b"
TEXT_MUTED = "#71717a"
SHADOW = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

# Tile Styling colors
TILE_COLORS = {
    0: {"bg": "#2563eb", "text": "#ffffff", "name": "Blue"},     # Blue
    1: {"bg": "#eab308", "text": "#09090b", "name": "Yellow"},   # Yellow
    2: {"bg": "#dc2626", "text": "#ffffff", "name": "Red"},      # Red
    3: {"bg": "#18181b", "text": "#ffffff", "name": "Black"},    # Black
    4: {"bg": "#fafafa", "text": "#09090b", "name": "White"},    # White
}

# Custom CSS for the Azul Dashboard
st.markdown(f"""
<style>
/* Hide Streamlit chrome */
header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    display: none !important;
}}

/* Global resets */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}
.block-container {{
    padding: 1.5rem 2rem 2rem !important;
}}

/* Card Container */
.panel-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: {SHADOW};
    margin-bottom: 1.25rem;
}}
.panel-title {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: {TEXT};
    border-bottom: 1px solid {BORDER_SUBTLE};
    padding-bottom: 0.4rem;
}}

/* Azul Game Visual Styles */
.factory-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
}}
.factory-circle {{
    border: 2px dashed {BORDER};
    border-radius: 50%;
    width: 100px;
    height: 100px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 4px;
    padding: 12px;
    background: {BG_SUBTLE};
    align-items: center;
    justify-items: center;
}}
.factory-circle.filled {{
    border-style: solid;
    border-color: {BORDER};
}}
.factory-label {{
    text-align: center;
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    font-weight: 500;
    margin-top: 0.4rem;
}}
.center-pool {{
    background: {BG_SUBTLE};
    border: 1px dashed {BORDER};
    border-radius: 10px;
    padding: 0.75rem;
    min-height: 50px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
}}

/* Tile styles */
.azul-tile {{
    width: 24px;
    height: 24px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.7rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    border: 1px solid rgba(0,0,0,0.1);
}}
.azul-tile.starting {{
    background: #10b981;
    color: white;
}}

/* Player board styling */
.board-columns {{
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 1.5rem;
}}
.pattern-lines {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-end;
}}
.pattern-row {{
    display: flex;
    gap: 4px;
}}
.slot-empty {{
    width: 24px;
    height: 24px;
    border-radius: 4px;
    border: 1px dashed {BORDER};
    background: transparent;
}}

.wall-grid {{
    display: grid;
    grid-template-columns: repeat(5, 24px);
    gap: 4px;
}}
.wall-cell {{
    width: 24px;
    height: 24px;
    border-radius: 4px;
    border: 1px solid {BORDER_SUBTLE};
}}
.wall-cell.filled {{
    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}}
.wall-cell.empty {{
    opacity: 0.15;
}}

.floor-line {{
    display: flex;
    gap: 6px;
    margin-top: 1rem;
    background: {BG_SUBTLE};
    padding: 6px 10px;
    border-radius: 6px;
    border: 1px solid {BORDER_SUBTLE};
}}
.floor-slot {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
}}
.floor-penalty {{
    font-size: 0.65rem;
    color: {TEXT_MUTED};
    font-weight: bold;
}}

/* Consistent button styling across all browsers */
div[data-testid="stButton"] button {{
    background-color: {BG_SUBTLE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stButton"] button:hover {{
    background-color: {CARD_HOVER} !important;
    border-color: #2563eb !important;
    color: {TEXT} !important;
}}

div[data-testid="stButton"] button[kind="primary"] {{
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #2563eb !important;
}}

div[data-testid="stButton"] button[kind="primary"]:hover {{
    background-color: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    color: #ffffff !important;
}}

/* Prevent Streamlit from dimming the screen during reruns */
div[data-testid="stAppViewContainer"] {{
    opacity: 1 !important;
}}

/* Highlights for last moved tiles */
@keyframes pop-in {{
    0% {{ transform: scale(0); opacity: 0; }}
    80% {{ transform: scale(1.12); opacity: 0.9; }}
    100% {{ transform: scale(1); opacity: 1; }}
}}

@keyframes pulse-glisten {{
    0% {{ transform: scale(1); box-shadow: 0 0 4px rgba(37, 99, 235, 0.4); }}
    50% {{ transform: scale(1.06); box-shadow: 0 0 10px rgba(37, 99, 235, 0.7), inset 0 0 4px rgba(255,255,255,0.4); }}
    100% {{ transform: scale(1); box-shadow: 0 0 4px rgba(37, 99, 235, 0.4); }}
}}

.highlight-placement {{
    animation: pop-in 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards, pulse-glisten 1.5s infinite ease-in-out 0.4s !important;
    border: 2px solid #2563eb !important;
}}

.highlight-container {{
    border: 1.5px solid #2563eb !important;
    box-shadow: 0 0 8px rgba(37, 99, 235, 0.3) !important;
}}
</style>
""", unsafe_allow_html=True)

# Helper function to parse actions into human readable strings
def parse_action_string(action, num_factories=5):
  source = action // 30
  color = (action % 30) // 6
  destination = action % 6
  
  source_str = f"Factory {source}" if source < num_factories else "Center"
  color_str = TILE_COLORS[color]["name"]
  dest_str = f"Pattern Row {destination + 1}" if destination < 5 else "Floor Line"
  return f"Take {color_str} from {source_str} to {dest_str}"

# 1. State Management Setup
if "env" not in st.session_state:
  st.session_state.env = rl_environment.Environment("azul", players=2)
  st.session_state.env.reset()
if "game_over" not in st.session_state:
  st.session_state.game_over = False
if "history" not in st.session_state:
  st.session_state.history = []
if "dqn_agent" not in st.session_state:
  st.session_state.dqn_agent = None
if "nfsp_agent" not in st.session_state:
  st.session_state.nfsp_agent = None
if "mcts_bot" not in st.session_state:
  st.session_state.mcts_bot = None
if "hybrid_bot" not in st.session_state:
  st.session_state.hybrid_bot = None
if "heuristic_bot" not in st.session_state:
  st.session_state.heuristic_bot = None
if "last_dqn_dir" not in st.session_state:
  st.session_state.last_dqn_dir = ""
if "last_nfsp_dir" not in st.session_state:
  st.session_state.last_nfsp_dir = ""
if "last_mcts_sims" not in st.session_state:
  st.session_state.last_mcts_sims = 0
if "last_hybrid_sims" not in st.session_state:
  st.session_state.last_hybrid_sims = 0
if "last_heuristic_sims" not in st.session_state:
  st.session_state.last_heuristic_sims = 0
if "draft_choice" not in st.session_state:
  st.session_state.draft_choice = None
if "last_hidden_layers" not in st.session_state:
  st.session_state.last_hidden_layers = []
if "last_opponent_id" not in st.session_state:
  st.session_state.last_opponent_id = -1
if "last_action_details" not in st.session_state:
  st.session_state.last_action_details = None

env = st.session_state.env
state = env.get_state

from open_spiel.python.algorithms import mcts

class DQNEvaluator(mcts.Evaluator):
  def __init__(self, dqn_agent):
    self._agent = dqn_agent
    import torch
    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    self._cache = {} # Cache key (bytes) -> (q_0, q_1)
    
  def _get_q_values(self, state):
    import numpy as np
    import torch
    obs_0 = np.asarray(state.observation_tensor(0))
    cache_key = obs_0.tobytes()
    if cache_key in self._cache:
      return self._cache[cache_key]
      
    obs_1 = np.asarray(state.observation_tensor(1))
    t_0 = torch.tensor(obs_0, dtype=torch.float32, device=self._device).unsqueeze(0)
    t_1 = torch.tensor(obs_1, dtype=torch.float32, device=self._device).unsqueeze(0)
    
    with torch.no_grad():
      q_0 = self._agent._q_network(t_0).squeeze(0).cpu().numpy()
      q_1 = self._agent._q_network(t_1).squeeze(0).cpu().numpy()
      
    res = (q_0, q_1)
    if len(self._cache) > 10000:
      self._cache.clear()
    self._cache[cache_key] = res
    return res

  def evaluate(self, state):
    import numpy as np
    q_0, q_1 = self._get_q_values(state)
    curr_player = state.current_player()
    legal = state.legal_actions()
    
    if curr_player == 0:
      val_0 = max(q_0[a] for a in legal) if len(legal) > 0 else 0.0
      val_1 = max(q_1)  # Proxy for opponent
    elif curr_player == 1:
      val_0 = max(q_0)  # Proxy for opponent
      val_1 = max(q_1[a] for a in legal) if len(legal) > 0 else 0.0
    else:
      val_0 = max(q_0)
      val_1 = max(q_1)
      
    return np.array([val_0, val_1])
    
  def prior(self, state):
    import numpy as np
    
    curr_player = state.current_player()
    if curr_player not in [0, 1]:
      # Chance node
      return state.chance_outcomes()
      
    q_0, q_1 = self._get_q_values(state)
    q = q_0 if curr_player == 0 else q_1
      
    legal_actions = state.legal_actions()
    legal_q = np.array([q[a] for a in legal_actions])
    
    # Softmax with temperature T = 5.0 to balance exploitation/exploration
    T = 5.0
    exp_q = np.exp((legal_q - np.max(legal_q)) / T)
    probs = exp_q / np.sum(exp_q)
    
    return list(zip(legal_actions, probs))

# ----------------- HEURISTIC STRATEGY MODULE -----------------
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

class HeuristicEvaluator(mcts.Evaluator):
  def evaluate(self, state):
    import numpy as np
    if state.is_terminal():
      return np.array(state.returns())
    obs = np.asarray(state.observation_tensor(0))
    b0 = parse_board(obs, 0)
    b1 = parse_board(obs, 1)
    return np.array([heuristic_value(b0) / 10.0, heuristic_value(b1) / 10.0])

  def prior(self, state):
    legal = state.legal_actions()
    return [(action, 1.0 / len(legal)) for action in legal]


# Load DQN & MCTS when configs change
def load_agents(checkpoint_dir, opponent_type, mcts_sims, hidden_layers_sizes, opponent_id):
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  
  # DQN Loading
  if opponent_type in ["dqn", "hybrid_mcts_dqn"]:
    if (st.session_state.dqn_agent is None or 
        st.session_state.last_dqn_dir != checkpoint_dir or
        st.session_state.last_hidden_layers != hidden_layers_sizes or
        st.session_state.last_opponent_id != opponent_id):
      try:
        use_dueling = "dddqn" in checkpoint_dir
        agent = dqn.DQN(
            player_id=opponent_id,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            use_double_dqn=True,
            use_dueling=use_dueling
        )
        agent.load(checkpoint_dir)
        st.session_state.dqn_agent = agent
        st.session_state.last_dqn_dir = checkpoint_dir
        st.session_state.last_hidden_layers = hidden_layers_sizes
        st.session_state.last_opponent_id = opponent_id
        st.sidebar.success(f"Successfully loaded DQN weights.")
      except Exception as e:
        st.sidebar.error(f"Error loading DQN weights: {e}")
  
  # Hybrid MCTS-DQN Loading
  if opponent_type == "hybrid_mcts_dqn" and st.session_state.dqn_agent is not None:
    if (st.session_state.hybrid_bot is None or 
        st.session_state.last_hybrid_sims != mcts_sims or
        st.session_state.last_dqn_dir != checkpoint_dir):
      try:
        evaluator = DQNEvaluator(st.session_state.dqn_agent)
        st.session_state.hybrid_bot = mcts.MCTSBot(
            game=env.game,
            uct_c=50.0,
            max_simulations=mcts_sims,
            evaluator=evaluator,
            solve=True,
        )
        st.session_state.last_hybrid_sims = mcts_sims
        st.sidebar.success(f"Successfully initialized Hybrid MCTS-DQN (sims={mcts_sims}).")
      except Exception as e:
        st.sidebar.error(f"Error initializing Hybrid MCTS-DQN Bot: {e}")
        
  # NFSP Loading
  if opponent_type == "nfsp":
    if (st.session_state.nfsp_agent is None or 
        st.session_state.last_nfsp_dir != checkpoint_dir or
        st.session_state.last_hidden_layers != hidden_layers_sizes or
        st.session_state.last_opponent_id != opponent_id):
      try:
        from open_spiel.python.pytorch import nfsp
        import pathlib
        agent = nfsp.NFSP(
            player_id=opponent_id,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            reservoir_buffer_capacity=2000000,
            anticipatory_param=0.1,
        )
        agent_path = pathlib.Path(checkpoint_dir) / f"agent_{opponent_id}"
        agent.restore(agent_path)
        st.session_state.nfsp_agent = agent
        st.session_state.last_nfsp_dir = checkpoint_dir
        st.session_state.last_hidden_layers = hidden_layers_sizes
        st.session_state.last_opponent_id = opponent_id
        st.sidebar.success(f"Successfully loaded NFSP weights for Player {opponent_id}.")
      except Exception as e:
        st.sidebar.error(f"Error loading NFSP weights: {e}")
  
  # MCTS Loading (Using fast native C++ implementation)
  if opponent_type == "mcts":
    if st.session_state.mcts_bot is None or st.session_state.last_mcts_sims != mcts_sims:
      evaluator = pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=42)
      st.session_state.mcts_bot = pyspiel.MCTSBot(
          game=env.game,
          evaluator=evaluator,
          uct_c=2.0,
          max_simulations=mcts_sims,
          max_memory_mb=1000,
          solve=True,
          seed=42,
          verbose=False,
      )
      st.session_state.last_mcts_sims = mcts_sims

  # Heuristic MCTS Loading
  if opponent_type == "heuristic_mcts":
    if st.session_state.heuristic_bot is None or st.session_state.last_heuristic_sims != mcts_sims:
      try:
        evaluator = HeuristicEvaluator()
        st.session_state.heuristic_bot = mcts.MCTSBot(
            game=env.game,
            uct_c=50.0,
            max_simulations=mcts_sims,
            evaluator=evaluator,
            solve=True,
        )
        st.session_state.last_heuristic_sims = mcts_sims
        st.sidebar.success(f"Successfully initialized Heuristic MCTS (sims={mcts_sims}).")
      except Exception as e:
        st.sidebar.error(f"Error initializing Heuristic MCTS Bot: {e}")

# Side bar configs
st.sidebar.markdown(f"### ◆ Game Config")
opponent_type = st.sidebar.selectbox("Opponent Bot", ["dqn", "hybrid_mcts_dqn", "heuristic_mcts", "nfsp", "mcts", "random", "human"], index=1)
human_seat = st.sidebar.radio("Human Seat", [0, 1], index=0)

checkpoint_dir = "./checkpoints_dddqn_900k"
hidden_layers_sizes = [256, 256]
if opponent_type in ["dqn", "hybrid_mcts_dqn"]:
  checkpoint_dir = st.sidebar.text_input("DQN Checkpoint Dir", value="./checkpoints_dddqn_900k")
  hidden_layers_str = st.sidebar.text_input("DQN Layers (comma-separated)", value="256,256")
  try:
    hidden_layers_sizes = [int(hs) for hs in hidden_layers_str.split(",") if hs.strip()]
  except Exception:
    st.sidebar.error("Invalid hidden layer sizes format. Use e.g. 128,128")
elif opponent_type == "nfsp":
  checkpoint_dir = st.sidebar.text_input("NFSP Checkpoint Dir", value="./checkpoints_nfsp_256")
  hidden_layers_str = st.sidebar.text_input("NFSP Layers (comma-separated)", value="256,256")
  try:
    hidden_layers_sizes = [int(hs) for hs in hidden_layers_str.split(",") if hs.strip()]
  except Exception:
    st.sidebar.error("Invalid hidden layer sizes format. Use e.g. 128,128")

mcts_sims = 50
if opponent_type == "mcts":
  mcts_sims = st.sidebar.slider("MCTS Simulations", min_value=10, max_value=200, value=50, step=10)
elif opponent_type in ["hybrid_mcts_dqn", "heuristic_mcts"]:
  mcts_sims = st.sidebar.slider(f"{opponent_type.replace('_', ' ').title()} Simulations", min_value=10, max_value=500, value=300, step=10)

opponent_id = 1 - human_seat
load_agents(checkpoint_dir, opponent_type, mcts_sims, hidden_layers_sizes, opponent_id)

# Header Row
head_left, head_right = st.columns([9, 1])
with head_left:
  st.markdown(f"""
  <div style="display:flex; align-items:center; gap: 10px;">
      <span style="font-size:1.75rem;">◆</span>
      <span style="font-size:1.4rem; font-weight:700; letter-spacing:-0.03em;">Azul Playroom</span>
  </div>
  """, unsafe_allow_html=True)
with head_right:
  theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
  st.button(theme_label, on_click=toggle_theme, use_container_width=True)

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# Main Grid Layout
left_col, right_col = st.columns([2.5, 1])

# Game loop trigger
def reset_game():
  st.session_state.env = rl_environment.Environment("azul", players=2)
  st.session_state.env.reset()
  st.session_state.game_over = False
  st.session_state.history = []
  if st.session_state.mcts_bot is not None:
    st.session_state.mcts_bot.restart()
  if st.session_state.hybrid_bot is not None:
    st.session_state.hybrid_bot.restart()
  if st.session_state.heuristic_bot is not None:
    st.session_state.heuristic_bot.restart()
  st.rerun()

# Helpers to extract destination state (row counts or floor size) from C++ state string representation
def get_row_count_and_floor_size(state_str, player_id, dest):
  lines = state_str.split("\n")
  current_p = -1
  for line in lines:
    if line.startswith(f"Player {player_id} Board"):
      current_p = player_id
    elif current_p == player_id:
      if dest < 5 and f"Row {dest} (" in line:
        # e.g., "    Row 2 (Cap 3): color 2, count 1"
        try:
          count = int(line.split("count")[1].strip())
          return count
        except Exception:
          return 0
      elif dest == 5 and "Floor line colors:" in line:
        # e.g., "  Floor line colors: [ 0 0 0 0 0 ] (Has starting token: Yes)"
        try:
          content = line.split('[')[1].split(']')[0].strip().split()
          counts = [int(x) for x in content]
          has_token = "Has starting token: Yes" in line
          return sum(counts), has_token
        except Exception:
          return 0, False
  return (0, False) if dest == 5 else 0

def get_dest_state_before_step(state, player, dest):
  state_str = str(state)
  return get_row_count_and_floor_size(state_str, player, dest)

# Apply actions
def apply_game_step(action):
  state = st.session_state.env.get_state
  player = state.current_player()
  dest = action % 6
  
  # Get state of destination before step
  before_val = get_dest_state_before_step(state, player, dest)
  
  action_str = parse_action_string(action)
  log_debug(f"[azul_web_app] Human Player {player} plays: {action_str} (action ID {action})")
  st.session_state.history.append(f"Player {player}: {action_str}")
  
  if st.session_state.mcts_bot is not None:
    st.session_state.mcts_bot.inform_action(state, player, action)
    
  time_step = st.session_state.env.step([action])
  
  # Get state of destination after step
  new_state = st.session_state.env.get_state
  after_val = get_dest_state_before_step(new_state, player, dest)
  
  if dest < 5:
    added_count = after_val - before_val
    added_starting_token = False
  else:
    added_count = after_val[0] - before_val[0]
    added_starting_token = after_val[1] and not before_val[1]
    
  st.session_state.last_action_details = {
      "player": player,
      "dest": dest,
      "added_count": added_count,
      "added_starting_token": added_starting_token
  }
  
  if time_step.last():
    st.session_state.game_over = True
    log_debug(f"[azul_web_app] Game Over! Returns: {st.session_state.env.get_state.returns()}")

# Apply a single step of the bot's turn
def apply_single_bot_step():
  state = st.session_state.env.get_state
  opp_player = state.current_player()
  time_step = st.session_state.env.get_time_step()
  
  log_debug(f"[azul_web_app] Bot turn. opp_player={opp_player}, opponent_type={opponent_type}, dqn_agent_loaded={st.session_state.dqn_agent is not None}")
  
  if opponent_type == "dqn" and st.session_state.dqn_agent is not None:
    # DQN Move
    log_debug(f"[azul_web_app] DQN agent selected. player_id={st.session_state.dqn_agent.player_id}, model_dir={st.session_state.last_dqn_dir}, model_layers={st.session_state.last_hidden_layers}")
    agent_output = st.session_state.dqn_agent.step(time_step, is_evaluation=True)
    opp_action = agent_output.action
  elif opponent_type == "hybrid_mcts_dqn" and st.session_state.hybrid_bot is not None:
    # Hybrid MCTS-DQN Move
    log_debug(f"[azul_web_app] Hybrid MCTS-DQN agent selected. sims={st.session_state.last_hybrid_sims}")
    opp_action = st.session_state.hybrid_bot.step(state)
  elif opponent_type == "heuristic_mcts" and st.session_state.heuristic_bot is not None:
    # Heuristic MCTS Move
    log_debug(f"[azul_web_app] Heuristic MCTS agent selected. sims={st.session_state.last_heuristic_sims}")
    opp_action = st.session_state.heuristic_bot.step(state)
  elif opponent_type == "nfsp" and st.session_state.nfsp_agent is not None:
    # NFSP Move
    from open_spiel.python.pytorch import nfsp
    log_debug(f"[azul_web_app] NFSP agent selected. player_id={st.session_state.nfsp_agent.player_id}, model_dir={st.session_state.last_nfsp_dir}, model_layers={st.session_state.last_hidden_layers}")
    with st.session_state.nfsp_agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
      agent_output = st.session_state.nfsp_agent.step(time_step, is_evaluation=True)
      opp_action = agent_output.action
  elif opponent_type == "mcts":
    # MCTS Move
    log_debug("[azul_web_app] MCTS agent selected.")
    opp_action = st.session_state.mcts_bot.step(state)
  else:
    # Random Move / Fallback
    log_debug(f"[azul_web_app] Fallback/Random agent selected (reason: opponent_type={opponent_type})")
    opp_action = np.random.choice(state.legal_actions())
    
  opp_action_str = parse_action_string(opp_action)
  log_debug(f"[azul_web_app] Bot (Player {opp_player}) plays: {opp_action_str} (action ID {opp_action})")
  st.session_state.history.append(f"Bot (Player {opp_player}): {opp_action_str}")
  
  dest = opp_action % 6
  # Get state of destination before step
  before_val = get_dest_state_before_step(state, opp_player, dest)
  
  if st.session_state.mcts_bot is not None:
    st.session_state.mcts_bot.inform_action(state, opp_player, opp_action)
    
  next_time_step = st.session_state.env.step([opp_action])
  
  # Get state of destination after step
  new_state = st.session_state.env.get_state
  after_val = get_dest_state_before_step(new_state, opp_player, dest)
  
  if dest < 5:
    added_count = after_val - before_val
    added_starting_token = False
  else:
    added_count = after_val[0] - before_val[0]
    added_starting_token = after_val[1] and not before_val[1]
    
  st.session_state.last_action_details = {
      "player": opp_player,
      "dest": dest,
      "added_count": added_count,
      "added_starting_token": added_starting_token
  }
  
  if next_time_step.last():
    st.session_state.game_over = True
    log_debug(f"[azul_web_app] Game Over! Returns: {st.session_state.env.get_state.returns()}")

# Auto-trigger bot move if it is the bot's turn
if not st.session_state.game_over:
  curr_player = state.current_player()
  if curr_player != human_seat:
    import time
    time.sleep(0.8)  # Short delay to animate bot thinking and placement
    apply_single_bot_step()
    st.rerun()

with left_col:
  # 2. Render Factory and Pool Board
  with st.container():
    st.markdown('<div class="panel-card"><div class="panel-title">Factories & Center Pool</div>', unsafe_allow_html=True)
    
    # Read Factories from C++ State
    # Note: Azul C++ state contains factories_ (vector of vector of int)
    # We can retrieve it by parsing the ToString() or accessing custom functions, 
    # but the cleanest way is using the flat observation vector where factory states are indexed.
    # Alternatively, we can parse the state to string and extract factory contents.
    # Let's write a small parser of State.ToString() to get precise factory counts!
    factories = [ [0]*5 for _ in range(5) ]
    center_pool = [0]*5
    has_starting_token = False
    
    lines = str(state).split('\n')
    for idx, line in enumerate(lines):
      if line.startswith("  F") and ":" in line and not line.startswith("  Floor"):
        # e.g. "  F0: [ 1 0 2 0 1 ]"
        f_idx = int(line.split(":")[0].strip()[1:])
        content = line.split('[')[1].split(']')[0].strip().split()
        factories[f_idx] = [int(x) for x in content]
      elif line.startswith("Center:") and "[" in line:
        # e.g. "Center: [ 1 2 0 0 1 ] (Has starting token: Yes)"
        content = line.split('[')[1].split(']')[0].strip().split()
        center_pool = [int(x) for x in content]
        has_starting_token = "Yes" in line

    # Render Factories
    f_cols = st.columns(5)
    for f in range(5):
      with f_cols[f]:
        # Generate HTML tiles inside circle
        tiles_html = ""
        filled_count = sum(factories[f])
        filled_cls = "filled" if filled_count > 0 else ""
        for color, count in enumerate(factories[f]):
          for _ in range(count):
            bg = TILE_COLORS[color]["bg"]
            tc = TILE_COLORS[color]["text"]
            tiles_html += f'<div class="azul-tile" style="background:{bg}; color:{tc};">{color}</div>'
        st.markdown(f"""
        <div class="factory-circle {filled_cls}">
            {tiles_html}
        </div>
        <div class="factory-label">Factory {f}</div>
        """, unsafe_allow_html=True)
        
        # Show click-to-choose buttons for tiles in this factory
        if not st.session_state.game_over and state.current_player() == human_seat:
          legal_actions = state.legal_actions()
          for color in range(5):
            count = factories[f][color]
            if count > 0:
              is_legal = any(a // 30 == f and (a % 30) // 6 == color for a in legal_actions)
              if is_legal:
                color_name = TILE_COLORS[color]['name']
                if st.button(f"Take {color_name} ({count})", key=f"btn_tile_f{f}_c{color}", use_container_width=True):
                  st.session_state.draft_choice = (f, color)
                  st.rerun()
        
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
    
    # Render Center Pool
    center_tiles_html = ""
    if has_starting_token:
      center_tiles_html += '<div class="azul-tile starting">1</div>'
    for color, count in enumerate(center_pool):
      for _ in range(count):
        bg = TILE_COLORS[color]["bg"]
        tc = TILE_COLORS[color]["text"]
        center_tiles_html += f'<div class="azul-tile" style="background:{bg}; color:{tc};">{color}</div>'
        
    st.markdown(f"""
    <div style="font-size: 0.78rem; color: {TEXT_MUTED}; font-weight: 500; margin-bottom:0.3rem;">Center Pool</div>
    <div class="center-pool" style="margin-bottom:0.5rem;">
        {center_tiles_html if center_tiles_html else "<span style='font-size:0.75rem; color:"+TEXT_MUTED+";'>Pool is empty</span>"}
    </div>
    """, unsafe_allow_html=True)
    
    # Click buttons for center pool
    if not st.session_state.game_over and state.current_player() == human_seat:
      legal_actions = state.legal_actions()
      c_cols = st.columns(5)
      col_idx = 0
      for color in range(5):
        count = center_pool[color]
        if count > 0:
          is_legal = any(a // 30 == 5 and (a % 30) // 6 == color for a in legal_actions)
          if is_legal:
            color_name = TILE_COLORS[color]['name']
            with c_cols[col_idx % 5]:
              if st.button(f"Take {color_name} ({count})", key=f"btn_tile_center_c{color}", use_container_width=True):
                st.session_state.draft_choice = (5, color)
                st.rerun()
            col_idx += 1
    st.markdown('</div>', unsafe_allow_html=True)

  # 3. Render Player Boards Side-by-Side
  # Parse Player Board details from State String
  player_boards = []
  for p in range(2):
    board = {
        "score": 0,
        "pattern": [ {"color": -1, "count": 0} for _ in range(5) ],
        "wall": [ [False]*5 for _ in range(5) ],
        "floor": [],
        "has_starting_token": False,
    }
    player_boards.append(board)
    
  current_p = -1
  for line in lines:
    if line.startswith("Player 0 Board"):
      current_p = 0
      board_score = int(line.split("Score:")[1].split(")")[0].strip())
      player_boards[current_p]["score"] = board_score
    elif line.startswith("Player 1 Board"):
      current_p = 1
      board_score = int(line.split("Score:")[1].split(")")[0].strip())
      player_boards[current_p]["score"] = board_score
    elif current_p != -1:
      if "Row" in line and "Cap" in line:
        # Pattern rows: "    Row 0 (Cap 1): color -1, count 0"
        row_idx = int(line.split("Row")[1].split("(")[0].strip())
        color = int(line.split("color")[1].split(",")[0].strip())
        count = int(line.split("count")[1].strip())
        player_boards[current_p]["pattern"][row_idx] = {"color": color, "count": count}
      elif line.startswith("    Row") and ":" in line and not "Cap" in line:
        # Wall row: "    Row 0: . . . . ." or "    Row 0: . . . X ."
        row_idx = int(line.split("Row")[1].split(":")[0].strip())
        cells = line.split(":")[1].strip().split()
        player_boards[current_p]["wall"][row_idx] = [c == "X" for c in cells]
      elif "Floor line colors:" in line:
        # Floor line colors: [ 0 0 0 0 0 ] (Has starting token: No)
        content = line.split('[')[1].split(']')[0].strip().split()
        counts = [int(x) for x in content]
        floor_tiles = []
        for color_idx, count in enumerate(counts):
          floor_tiles.extend([color_idx] * count)
        player_boards[current_p]["floor"] = floor_tiles
        
        has_token = "Has starting token: Yes" in line
        player_boards[current_p]["has_starting_token"] = has_token
        # for (int c : floor_line_colors) out << c << " "
        # In azul.cc: if a floor space has no tile, does it have a color index?
        # Let's check: if there are no tiles in the floor line, what is printed?
        # In our console test output: "Floor line colors: [ 0 0 0 0 0 ]"
        # Wait, does 0 mean empty or color index 0?
        # Actually, color index ranges from 0 to 4.
        # Let's count how many tiles are actually on the floor line!
        # In Azul, floor line tiles are populated starting from left.
        # If we have 2 blue tiles on floor, the printed output has some count.
        # Let's check if the floor line size matches the number of actual tiles on it.
        # In C++, player_boards_[p].floor_line_colors contains the colors of tiles on the floor.
        # If empty, the vector has size 0.
        # So "Floor line colors: [ 0 0 0 0 0 ]" contains 5 tiles of color 0!
        # Oh! Color 0 is Blue!
        # Wait, if it is empty, does it print `[ ]`?
        # Let's look at our previous test outputs:
        # "Floor line colors: [ 0 0 0 0 0 ] (Has starting token: No)"
        # Wait! In the initial state:
        # "Floor line colors: [ 0 0 0 0 0 ] (Has starting token: No)"
        # Oh, if it prints `0 0 0 0 0`, is it color 0 or empty?
        # Let's check `azul.cc` line 300 to see how `floor_line_colors` is printed!

  # Let's check if floor line is empty when size is 0:
  # In C++, if empty, floor_line_colors is empty. But wait, why did it print "0 0 0 0 0"?
  # Let's search for "Floor line colors" in `azul.cc` to see how it is printed.
  # Actually, we can just display the floor line tiles!
  # If there is a starting token: "Has starting token: Yes" -> display starting token on floor line!
  # And for other values: if the array has color values, we can display them.
  # Let's see: if floor line is printed as "Floor line colors: [ 0 0 0 0 0 ]", that means the vector contains 5 elements.
  # Wait, in the initial state, does it contain 0 elements?
  # Ah! In the initial state, the output printed:
  # "Floor line colors: [ 0 0 0 0 0 ] (Has starting token: No)"
  # Wait! Why did it print `0 0 0 0 0` if it was empty?
  # Ah, in our C++ code, we might have initialized `floor_line_colors` with size 7 or similar, or 0s.
  # Let's look at `azul.cc` to see how `floor_line_colors` is printed.
  # In any case, we can show whatever is in the vector.

  b_cols = st.columns(2)
  for p in range(2):
    with b_cols[p]:
      is_human = (p == human_seat)
      role_str = "Human" if is_human else f"Bot ({opponent_type.upper()})"
      # Construct the entire board markup as a single string
      board_html = f"""<div class="panel-card">
<div class="panel-title">
Player {p} Board ({role_str}) 
<span style="float:right; font-weight:bold; color:#2563eb;">Score: {player_boards[p]["score"]}</span>
</div>
<div class="board-columns">
<div class="pattern-lines">"""
      
      # Generate Pattern Rows
      for r in range(5):
        slots_html = ""
        pat = player_boards[p]["pattern"][r]
        color = pat["color"]
        count = pat["count"]
        empty_slots = (r + 1) - count
        
        # Check if this row was targeted in the last move
        is_row_highlighted = (
            st.session_state.last_action_details is not None and
            st.session_state.last_action_details["player"] == p and
            st.session_state.last_action_details["dest"] == r
        )
        
        for _ in range(empty_slots):
          slots_html += '<div class="slot-empty"></div>'
        for i_tile in range(count):
          bg = TILE_COLORS[color]["bg"]
          tc = TILE_COLORS[color]["text"]
          # Highlight only newly added tiles (indices < added_count since they fill from right to left)
          is_tile_new = is_row_highlighted and (i_tile < st.session_state.last_action_details["added_count"])
          highlight_class = " highlight-placement" if is_tile_new else ""
          slots_html += f'<div class="azul-tile{highlight_class}" style="background:{bg}; color:{tc};">{color}</div>'
        board_html += f'<div class="pattern-row">{slots_html}</div>'
        
      board_html += """</div>
<div class="wall-grid">"""
      
      # Generate Wall Cells
      for r in range(5):
        for c in range(5):
          w_color = (c - r + 5) % 5
          bg = TILE_COLORS[w_color]["bg"]
          tc = TILE_COLORS[w_color]["text"]
          filled = player_boards[p]["wall"][r][c]
          if filled:
            board_html += f'<div class="wall-cell filled" style="background:{bg}; color:{tc}; display:flex; align-items:center; justify-content:center; font-size:0.7rem; font-weight:bold;">{w_color}</div>'
          else:
            board_html += f'<div class="wall-cell empty" style="background:{bg}; border-color:{bg}; opacity: 0.15;"></div>'
            
      board_html += """</div>
</div>"""
      
      # Check if floor was targeted in the last move
      is_floor_highlighted = (
          st.session_state.last_action_details is not None and
          st.session_state.last_action_details["player"] == p and
          st.session_state.last_action_details["dest"] == 5
      )
      
      # Just parse floor colors
      floor_penalties = [-1, -1, -2, -2, -2, -3, -3]
      floor_slots_html = ""
      
      # Build the list of tiles to display in the 7 floor slots
      display_tiles = []
      if player_boards[p]["has_starting_token"]:
        display_tiles.append("starting")  # Special marker for starting token
      display_tiles.extend(player_boards[p]["floor"])
      
      added_count = st.session_state.last_action_details["added_count"] if is_floor_highlighted else 0
      added_starting_token = st.session_state.last_action_details["added_starting_token"] if is_floor_highlighted else False
      
      # Calculate start index of new normal tiles
      floor_len = len(player_boards[p]["floor"])
      first_new_floor_idx = floor_len - added_count
      
      for i, penalty in enumerate(floor_penalties):
        tile_div = '<div class="slot-empty" style="width:20px; height:20px;"></div>'
        if i < len(display_tiles):
          tile_type = display_tiles[i]
          if tile_type == "starting":
            # Render Green starting player token
            is_tile_new = added_starting_token
            highlight_class = " highlight-placement" if is_tile_new else ""
            tile_div = f'<div class="azul-tile{highlight_class}" style="background:#10b981; color:white; width:20px; height:20px; font-size:0.65rem; font-weight:bold; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">S</div>'
          else:
            col = tile_type
            bg = TILE_COLORS[col]["bg"]
            tc = TILE_COLORS[col]["text"]
            
            # Normal floor tile is new if its index in floor list >= first_new_floor_idx
            floor_idx = i - 1 if player_boards[p]["has_starting_token"] else i
            is_tile_new = is_floor_highlighted and (floor_idx >= first_new_floor_idx)
            highlight_class = " highlight-placement" if is_tile_new else ""
            tile_div = f'<div class="azul-tile{highlight_class}" style="background:{bg}; color:{tc}; width:20px; height:20px; font-size:0.65rem;">{col}</div>'
            
        floor_slots_html += f"""<div class="floor-slot">
{tile_div}
<div class="floor-penalty">{penalty}</div>
</div>"""
        
      floor_line_class = "floor-line highlight-container" if is_floor_highlighted else "floor-line"
      board_html += f"""<div class="{floor_line_class}">
<span style="font-size:0.75rem; color:{TEXT_MUTED}; font-weight:500; margin-right:4px;">Floor:</span>
{floor_slots_html}
</div>
</div>"""
      
      st.markdown(board_html, unsafe_allow_html=True)

# Right Column - Controls and Move History
with right_col:
  # Game controls panel
  with st.container():
    st.markdown('<div class="panel-card"><div class="panel-title">Control Panel</div>', unsafe_allow_html=True)
    
    if st.session_state.game_over:
      returns = state.returns()
      winner_str = ""
      if returns[human_seat] > returns[1 - human_seat]:
        winner_str = "🏆 You Won!"
      elif returns[human_seat] < returns[1 - human_seat]:
        winner_str = "🤖 Bot Won!"
      else:
        winner_str = "🤝 It's a Tie!"
      st.markdown(f"""
      <div style="background:rgba(16,185,129,0.12); border:1px solid #10b981; border-radius:8px; padding:10px; margin-bottom:1rem; text-align:center;">
          <h4 style="color:#10b981; margin:0; font-size:1.1rem;">Game Over: {winner_str}</h4>
          <p style="margin:4px 0 0 0; font-size:0.8rem; color:{TEXT_MUTED}">
              Final Scores - Human: {returns[human_seat]} | Bot: {returns[1 - human_seat]}
          </p>
      </div>
      """, unsafe_allow_html=True)
      
    # Action selector
    if not st.session_state.game_over:
      curr_player = state.current_player()
      if curr_player == human_seat:
        st.markdown(f"<span style='font-size:0.85rem; font-weight:600; color:#2563eb;'>Your turn (Player {curr_player})</span>", unsafe_allow_html=True)
        
        legal_actions = state.legal_actions()
        
        # Check if they have clicked a tile (draft_choice is active)
        if "draft_choice" in st.session_state and st.session_state.draft_choice is not None:
          source, color = st.session_state.draft_choice
          source_name = f"Factory {source}" if source < 5 else "Center Pool"
          color_name = TILE_COLORS[color]["name"]
          
          st.markdown(f"""
          <div style="background:rgba(37,99,235,0.08); border:1px solid #2563eb; border-radius:8px; padding:10px; margin-bottom:1rem;">
              <span style="font-size:0.75rem; color:{TEXT_MUTED}; font-weight:500;">Selected tiles:</span><br/>
              <span style="font-size:0.85rem; font-weight:bold; color:{TEXT};">Take {color_name} from {source_name}</span>
          </div>
          """, unsafe_allow_html=True)
          
          # Find legal destinations for this source + color
          matching_actions = [a for a in legal_actions if a // 30 == source and (a % 30) // 6 == color]
          
          st.markdown("<span style='font-size:0.78rem; font-weight:600;'>Select placement destination:</span>", unsafe_allow_html=True)
          for action in matching_actions:
            dest = action % 6
            dest_label = f"Pattern Row {dest + 1}" if dest < 5 else "Floor Line"
            if st.button(dest_label, key=f"btn_dest_{action}", use_container_width=True):
              apply_game_step(action)
              st.session_state.draft_choice = None
              st.rerun()
              
          if st.button("Cancel Selection", key="btn_cancel_draft", use_container_width=True):
            st.session_state.draft_choice = None
            st.rerun()
        else:
          # Standard Dropdown Action Selector
          action_options = {action: parse_action_string(action) for action in legal_actions}
          
          selected_action = st.selectbox(
              "Or choose from all legal actions:",
              options=list(action_options.keys()),
              format_func=lambda x: action_options[x],
              key="action_select"
          )
          
          if st.button("Apply Action", use_container_width=True, type="primary"):
            apply_game_step(selected_action)
            st.rerun()
      else:
        st.markdown(f"<span style='font-size:0.85rem; font-weight:600;'>Waiting for Bot (Player {curr_player}) to move...</span>", unsafe_allow_html=True)
    
    if st.button("Reset Game", use_container_width=True):
      reset_game()
      
    st.markdown('</div>', unsafe_allow_html=True)

  # Move history panel
  with st.container():
    st.markdown('<div class="panel-card"><div class="panel-title">Move History</div>', unsafe_allow_html=True)
    if st.session_state.history:
      history_html = "".join([f"<div style='font-size:0.75rem; border-bottom:1px solid {BORDER_SUBTLE}; padding:4px 0; font-family:\"JetBrains Mono\", monospace;'>{h}</div>" for h in reversed(st.session_state.history)])
      st.markdown(f"<div style='max-height:220px; overflow-y:auto;'>{history_html}</div>", unsafe_allow_html=True)
    else:
      st.markdown(f"<span style='font-size:0.75rem; color:{TEXT_MUTED};'>No moves made yet</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer info
st.markdown(f"""
<div style="text-align:center; font-size:0.7rem; color:{TEXT_MUTED}; margin-top:2rem;">
    Azul web app powered by OpenSpiel & PyTorch DQN.
</div>
""", unsafe_allow_html=True)
