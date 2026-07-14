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

"""Play interactively against a trained DQN agent on Azul."""

import os
import tempfile
from absl import app
from absl import flags
from absl import logging
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_dir",
    "./checkpoints",
    "Directory to load the DQN agent model from.",
)
flags.DEFINE_integer("human_player", 0, "Which player (0 or 1) is the human.")
flags.DEFINE_list("hidden_layers_sizes", [128, 128],
                  "Number of hidden units in the Q-Network MLP.")


def play_interactive_game(env, dqn_agent, human_player_id):
  """Runs an interactive game loop between a human and the DQN agent."""
  time_step = env.reset()
  dqn_player_id = 1 - human_player_id

  print("\n=======================================================")
  print("             Welcome to Azul: Human vs DQN             ")
  print(f"             Human is Player {human_player_id}, DQN is Player {dqn_player_id}")
  print("=======================================================\n")

  while not time_step.last():
    player_id = time_step.observations["current_player"]
    state = env.get_state

    print("\n-------------------------------------------------------")
    print(state)
    print("-------------------------------------------------------")

    if player_id == human_player_id:
      # Human Turn
      legal_actions = state.legal_actions()
      print(f"\nPlayer {player_id} (Human) - It's your turn!")
      print("Legal Moves:")
      
      # Print legal actions grouped for readability
      for action in legal_actions:
        action_str = state.action_to_string(player_id, action)
        print(f"  [{action}] {action_str}")
        
      # Prompt user
      while True:
        try:
          user_input = input("\nEnter the action ID in brackets (e.g., 30): ").strip()
          action = int(user_input)
          if action in legal_actions:
            break
          else:
            print("Action is not legal! Please choose from the listed IDs.")
        except ValueError:
          print("Invalid input! Please enter a valid integer ID.")
      
      action_str = state.action_to_string(player_id, action)
      print(f"\nYou chose: {action_str}")
    else:
      # DQN Agent Turn
      print(f"\nPlayer {player_id} (DQN) is thinking...")
      agent_output = dqn_agent.step(time_step, is_evaluation=True)
      action = agent_output.action
      action_str = state.action_to_string(player_id, action)
      print(f"DQN chose: {action_str}")

    # Step the environment
    time_step = env.step([action])

  # Game Over
  print("\n=======================================================")
  print("                      GAME OVER!                       ")
  print("=======================================================\n")
  print(env.get_state)
  
  returns = env.get_state.returns()
  print(f"\nFinal Scores:")
  print(f"  Human (Player {human_player_id}): {returns[human_player_id]}")
  print(f"  DQN (Player {dqn_player_id}): {returns[dqn_player_id]}")
  
  if returns[human_player_id] > returns[dqn_player_id]:
    print("\nCongratulations! You won!")
  elif returns[human_player_id] < returns[dqn_player_id]:
    print("\nDQN won! Better luck next time!")
  else:
    print("\nIt's a tie!")


def main(_):
  game = "azul"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  dqn_player_id = 1 - FLAGS.human_player

  # 1. Instantiate the DQN Agent
  dqn_agent = dqn.DQN(
      player_id=dqn_player_id,
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

  # 2. Run the game loop
  play_interactive_game(env, dqn_agent, FLAGS.human_player)


if __name__ == "__main__":
  app.run(main)
