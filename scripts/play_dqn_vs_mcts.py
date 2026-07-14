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

"""Plays a game of Azul between a trained DQN agent and an MCTS bot."""

import os
import tempfile
from absl import app
from absl import flags
from absl import logging
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.pytorch import dqn

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir",
    os.path.join(tempfile.gettempdir(), "azul_dqn"),
    "Directory to load the DQN agent model from.",
)
flags.DEFINE_integer("dqn_player", 0, "Which player (0 or 1) is the DQN agent.")
flags.DEFINE_integer("max_simulations", 100, "Number of MCTS simulations per turn.")
flags.DEFINE_integer("num_games", 1, "Number of games to play.")
flags.DEFINE_bool("verbose", True, "Show detailed board state after each move.")
flags.DEFINE_enum("opponent_type", "mcts", ["mcts", "random"], "Type of opponent to play against.")
flags.DEFINE_list("hidden_layers_sizes", [128, 128],
                  "Number of hidden units in the Q-Network MLP.")


def play_one_game(env, dqn_agent, mcts_bot, dqn_player_id):
  """Plays a single game of Azul."""
  time_step = env.reset()
  
  # Initialize the MCTS bot tree
  if mcts_bot is not None:
    mcts_bot.restart()
  
  if FLAGS.verbose:
    print("Initial State:\n", env.get_state)

  while not time_step.last():
    player_id = time_step.observations["current_player"]
    state = env.get_state
    
    if player_id == dqn_player_id:
      # DQN Agent turn
      agent_output = dqn_agent.step(time_step, is_evaluation=True)
      action = agent_output.action
      action_str = state.action_to_string(player_id, action)
      if FLAGS.verbose:
        print(f"Player {player_id} (DQN) chooses: {action_str}")
    else:
      # Opponent turn
      if mcts_bot is not None:
        action = mcts_bot.step(state)
      else:
        action = np.random.choice(state.legal_actions())
      action_str = state.action_to_string(player_id, action)
      if FLAGS.verbose:
        print(f"Player {player_id} ({FLAGS.opponent_type.upper()}) chooses: {action_str}")
        
    # Inform MCTS bot of the chosen action
    if mcts_bot is not None:
      mcts_bot.inform_action(state, player_id, action)
    
    # Step the environment
    time_step = env.step([action])
    
    if FLAGS.verbose:
      print("Next State:\n", env.get_state)
      
  # End of game
  final_returns = env.get_state.returns()
  return final_returns


def main(_):
  game = "azul"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # 1. Instantiate the DQN Agent
  dqn_agent = dqn.DQN(
      player_id=FLAGS.dqn_player,
      state_representation_size=info_state_size,
      num_actions=num_actions,
      hidden_layers_sizes=[int(hs) for hs in FLAGS.hidden_layers_sizes],
  )
  
  # Try to load DQN checkpoints if they exist
  try:
    dqn_agent.load(FLAGS.checkpoint_dir)
    logging.info(f"Successfully loaded DQN checkpoint from {FLAGS.checkpoint_dir}")
  except Exception as e:
    logging.warning(f"Could not load checkpoint: {e}. DQN agent will play with random weights.")

  # 2. Instantiate the MCTS Bot if needed (using fast native C++ implementation)
  mcts_bot = None
  if FLAGS.opponent_type == "mcts":
    import pyspiel
    evaluator = pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=42)
    mcts_bot = pyspiel.MCTSBot(
        game=env.game,
        evaluator=evaluator,
        uct_c=2.0,
        max_simulations=FLAGS.max_simulations,
        max_memory_mb=1000,
        solve=True,
        seed=42,
        verbose=False,
    )

  # 3. Play Games
  logging.info(f"Starting matches: DQN (Player {FLAGS.dqn_player}) vs {FLAGS.opponent_type.upper()} (Player {1 - FLAGS.dqn_player})")
  
  all_returns = []
  for g in range(FLAGS.num_games):
    print(f"\n--- Game {g+1} ---")
    returns = play_one_game(env, dqn_agent, mcts_bot, FLAGS.dqn_player)
    all_returns.append(returns)
    print(f"Game {g+1} Returns: DQN (Player {FLAGS.dqn_player}): {returns[FLAGS.dqn_player]}, {FLAGS.opponent_type.upper()}: {returns[1 - FLAGS.dqn_player]}")
    
  avg_returns = np.mean(all_returns, axis=0)
  print(f"\nAverage returns over {FLAGS.num_games} games:")
  print(f"  DQN (Player {FLAGS.dqn_player}): {avg_returns[FLAGS.dqn_player]}")
  print(f"  {FLAGS.opponent_type.upper()} (Player {1 - FLAGS.dqn_player}): {avg_returns[1 - FLAGS.dqn_player]}")


if __name__ == "__main__":
  app.run(main)
