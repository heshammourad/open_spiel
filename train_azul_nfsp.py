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

"""Trains NFSP agents on Azul by self-play."""

import os
import tempfile
import pathlib

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_enum("opponent_type", "self", ["self", "mcts", "random"], "Opponent type to train against.")
flags.DEFINE_integer("mcts_simulations", 10, "Number of MCTS simulations per turn during training.")

# Training parameters
flags.DEFINE_string(
    "checkpoint_dir",
    os.path.join(tempfile.gettempdir(), "azul_nfsp"),
    "Directory to save/load the agent models.",
)
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(5e4),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the agents are evaluated.")
flags.DEFINE_integer(
    "num_eval_episodes", 100,
    "Number of episodes for evaluation against random bots.")

# NFSP model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [128, 128],
                  "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the Q-learning replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_float("sl_learning_rate", 0.001,
                   "Supervised learning rate.")
flags.DEFINE_float("rl_learning_rate", 0.001,
                   "RL (Q-learning) learning rate.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("epsilon_decay_duration", int(1e5),
                     "Number of steps over which Q-learning epsilon decays.")
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          # Use AVERAGE_POLICY for evaluation if supported
          agent = cur_agents[player_id]
          if hasattr(agent, "temp_mode_as"):
            with agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
              agent_output = agent.step(time_step, is_evaluation=True)
          else:
            agent_output = agent.step(time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = []
          for agent in cur_agents:
            if hasattr(agent, "temp_mode_as"):
              with agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
                agents_output.append(agent.step(time_step, is_evaluation=True))
            else:
              agents_output.append(agent.step(time_step, is_evaluation=True))
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def main(_):
  game = "azul"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  # random agents for evaluation
  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  hidden_layers_sizes = [int(hs) for hs in FLAGS.hidden_layers_sizes]
  
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "device": "cuda" if torch.cuda.is_available() else "cpu",
      "optimizer_str": "adam",
  }

  agents = [
      nfsp.NFSP(
          player_id=idx,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          hidden_layers_sizes=hidden_layers_sizes,
          reservoir_buffer_capacity=FLAGS.reservoir_buffer_capacity,
          anticipatory_param=FLAGS.anticipatory_param,
          batch_size=FLAGS.batch_size,
          **kwargs
      )
      for idx in range(num_players)
  ]

  # Load existing checkpoints if available to resume training
  if FLAGS.use_checkpoints:
    for agent in agents:
      try:
        agent_path = pathlib.Path(FLAGS.checkpoint_dir) / f"agent_{agent.player_id}"
        agent.restore(agent_path)
        logging.info(f"Successfully loaded NFSP agent {agent.player_id} checkpoint from {agent_path}")
      except Exception as e:
        logging.info(f"No checkpoint found for agent {agent.player_id} in {FLAGS.checkpoint_dir}. Starting from scratch. ({e})")

  # Initialize MCTS Bot if needed (using fast native C++ implementation)
  mcts_bot = None
  if FLAGS.opponent_type == "mcts":
    import pyspiel
    evaluator = pyspiel.RandomRolloutEvaluator(n_rollouts=1, seed=42)
    mcts_bot = pyspiel.MCTSBot(
        game=env.game,
        evaluator=evaluator,
        uct_c=2.0,
        max_simulations=FLAGS.mcts_simulations,
        max_memory_mb=1000,
        solve=True,
        seed=42,
        verbose=False,
    )

  logging.info(f"Starting training of NFSP agents on Azul (opponent: {FLAGS.opponent_type})...")

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      r_mean = eval_against_random_bots(env, agents, random_agents, FLAGS.num_eval_episodes)
      logging.info("[%s] Mean episode rewards vs Random: %s", ep + 1, r_mean)

    if FLAGS.use_checkpoints and (ep + 1) % FLAGS.save_every == 0:
      for agent in agents:
        agent_path = pathlib.Path(FLAGS.checkpoint_dir) / f"agent_{agent.player_id}"
        agent.save(agent_path)

    if FLAGS.opponent_type == "self":
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
        else:
          agents_output = [agent.step(time_step) for agent in agents]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)
    else:
      # Opponent is MCTS or Random
      nfsp_player_id = ep % 2
      nfsp_agent = agents[nfsp_player_id]
      nfsp_agent.player_id = nfsp_player_id

      time_step = env.reset()
      if FLAGS.opponent_type == "mcts":
        mcts_bot.restart()

      while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if player_id == nfsp_player_id:
          agent_output = nfsp_agent.step(time_step)
          action = agent_output.action
        else:
          if FLAGS.opponent_type == "mcts":
            action = mcts_bot.step(state)
          else:
            action = np.random.choice(state.legal_actions())

        if FLAGS.opponent_type == "mcts":
          mcts_bot.inform_action(state, player_id, action)
        time_step = env.step([action])

      # Episode is over, step NFSP agent with final info state.
      nfsp_agent.step(time_step)

  if FLAGS.use_checkpoints:
    for agent in agents:
      agent_path = pathlib.Path(FLAGS.checkpoint_dir) / f"agent_{agent.player_id}"
      agent.save(agent_path)
    logging.info(f"Saved final checkpoints to {FLAGS.checkpoint_dir}")

  logging.info("Training complete.")


if __name__ == "__main__":
  app.run(main)
