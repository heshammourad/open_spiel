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

"""Trains DQN agents on Azul by independent Q-learning."""

import os
import tempfile

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import random_agent
from open_spiel.python.pytorch import dqn

FLAGS = flags.FLAGS

flags.DEFINE_enum("opponent_type", "self", ["self", "mcts", "random"], "Opponent type to train against.")
flags.DEFINE_integer("mcts_simulations", 10, "Number of MCTS simulations per turn during training.")

# Training parameters
flags.DEFINE_string(
    "checkpoint_dir",
    os.path.join(tempfile.gettempdir(), "azul_dqn"),
    "Directory to save/load the agent models.",
)
flags.DEFINE_integer(
    "save_every", int(1e4),
    "Episode frequency at which the DQN agent models are saved.")
flags.DEFINE_integer("num_train_episodes", int(5e4),
                     "Number of training episodes.")
flags.DEFINE_integer(
    "eval_every", 1000,
    "Episode frequency at which the DQN agents are evaluated.")
flags.DEFINE_integer(
    "num_eval_episodes", 100,
    "Number of episodes for evaluation against random bots.")

# DQN model hyper-parameters
flags.DEFINE_list("hidden_layers_sizes", [128, 128],
                  "Number of hidden units in the Q-Network MLP.")
flags.DEFINE_integer("replay_buffer_capacity", int(1e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("batch_size", 32,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("epsilon_decay_duration", int(1e5),
                     "Number of steps over which epsilon decays.")
flags.DEFINE_bool("use_checkpoints", False, "Save/load neural network weights.")
flags.DEFINE_bool("use_double_dqn", False, "Whether to use Double DQN.")
flags.DEFINE_bool("use_dueling", False, "Whether to use Dueling DQN.")
flags.DEFINE_bool("win_first_reward", False, "Whether to use win-first terminal rewards + scaled intermediate rewards.")
flags.DEFINE_bool("relative_rewards", False, "Whether to use relative intermediate rewards (r_agent - r_opponent).")
flags.DEFINE_bool("reset_exploration", False, "Whether to reset the iteration step count to 0 and restart exploration on checkpoint resume.")


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
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
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
  agents = [
      dqn.DQN(
          player_id=idx,
          state_representation_size=info_state_size,
          num_actions=num_actions,
          hidden_layers_sizes=hidden_layers_sizes,
          replay_buffer_capacity=FLAGS.replay_buffer_capacity,
          batch_size=FLAGS.batch_size,
          learning_rate=1e-3,
          optimizer_str="adam",
          epsilon_decay_duration=FLAGS.epsilon_decay_duration,
          epsilon_start=0.3 if FLAGS.reset_exploration else 1.0,
          epsilon_end=0.01 if FLAGS.reset_exploration else 0.1,
          discount_factor=0.99,
          use_double_dqn=FLAGS.use_double_dqn,
          use_dueling=FLAGS.use_dueling,
      )
      for idx in range(num_players)
  ]

  # Load existing checkpoints if available to resume training
  if FLAGS.use_checkpoints:
    for agent in agents:
      try:
        agent.load(FLAGS.checkpoint_dir)
        if FLAGS.reset_exploration:
          agent._iteration = 0
          agent.epsilon_schedule = dqn.linear_schedule(0.3, 0.01, FLAGS.epsilon_decay_duration)
          logging.info(f"Reset agent {agent.player_id} iteration to 0 and set linear exploration schedule [0.3 -> 0.01] over {FLAGS.epsilon_decay_duration} steps.")
        logging.info(f"Successfully loaded DQN agent {agent.player_id} checkpoint from {FLAGS.checkpoint_dir}")
      except Exception as e:
        logging.info(f"No checkpoint found for agent {agent.player_id} in {FLAGS.checkpoint_dir}. Starting from scratch.")

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

  logging.info(f"Starting training of DQN agents on Azul (opponent: {FLAGS.opponent_type})...")

  for ep in range(FLAGS.num_train_episodes):
    if (ep + 1) % FLAGS.eval_every == 0:
      r_mean = eval_against_random_bots(env, agents, random_agents, FLAGS.num_eval_episodes)
      logging.info("[%s] Mean episode rewards vs Random: %s", ep + 1, r_mean)

    if FLAGS.use_checkpoints and (ep + 1) % FLAGS.save_every == 0:
      for agent in agents:
        agent.save(FLAGS.checkpoint_dir)

    if FLAGS.opponent_type == "self":
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        if time_step.rewards:
          r0, r1 = time_step.rewards
          if FLAGS.relative_rewards:
            shaped_rewards = [r0 - r1, r1 - r0]
          else:
            shaped_rewards = [r0, r1]
          if FLAGS.win_first_reward:
            shaped_rewards = [0.1 * r for r in shaped_rewards]
          time_step = time_step._replace(rewards=shaped_rewards)
          
        if env.is_turn_based:
          agent_output = agents[player_id].step(time_step)
          action_list = [agent_output.action]
        else:
          agents_output = [agent.step(time_step) for agent in agents]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      if time_step.rewards:
        r0, r1 = time_step.rewards
        if FLAGS.relative_rewards:
          shaped_rewards = [r0 - r1, r1 - r0]
        else:
          shaped_rewards = [r0, r1]
        if FLAGS.win_first_reward:
          shaped_rewards = [0.1 * r for r in shaped_rewards]
        time_step = time_step._replace(rewards=shaped_rewards)

      if FLAGS.win_first_reward:
        returns = env.get_state.returns()
        if returns[0] > returns[1]:
          win_bonus = [10.0, -10.0]
        elif returns[1] > returns[0]:
          win_bonus = [-10.0, 10.0]
        else:
          win_bonus = [0.0, 0.0]
        
        if FLAGS.relative_rewards:
          final_rewards = [0.1 * (returns[0] - returns[1]) + win_bonus[0], 0.1 * (returns[1] - returns[0]) + win_bonus[1]]
        else:
          final_rewards = [0.1 * returns[0] + win_bonus[0], 0.1 * returns[1] + win_bonus[1]]
        time_step = time_step._replace(rewards=final_rewards)
      elif FLAGS.relative_rewards:
        returns = env.get_state.returns()
        final_rewards = [returns[0] - returns[1], returns[1] - returns[0]]
        time_step = time_step._replace(rewards=final_rewards)

      for agent in agents:
        agent.step(time_step)
    else:
      # Opponent is MCTS or Random
      dqn_player_id = ep % 2
      dqn_agent = agents[dqn_player_id]
      dqn_agent.player_id = dqn_player_id

      time_step = env.reset()
      if FLAGS.opponent_type == "mcts":
        mcts_bot.restart()

      while not time_step.last():
        player_id = time_step.observations["current_player"]
        state = env.get_state
        if time_step.rewards:
          r0, r1 = time_step.rewards
          if FLAGS.relative_rewards:
            shaped_rewards = [r0 - r1, r1 - r0]
          else:
            shaped_rewards = [r0, r1]
          if FLAGS.win_first_reward:
            shaped_rewards = [0.1 * r for r in shaped_rewards]
          time_step = time_step._replace(rewards=shaped_rewards)
          
        if player_id == dqn_player_id:
          agent_output = dqn_agent.step(time_step)
          action = agent_output.action
        else:
          if FLAGS.opponent_type == "mcts":
            action = mcts_bot.step(state)
          else:
            action = np.random.choice(state.legal_actions())

        if FLAGS.opponent_type == "mcts":
          mcts_bot.inform_action(state, player_id, action)
        time_step = env.step([action])

      # Episode is over, step DQN agent with final info state.
      if time_step.rewards:
        r0, r1 = time_step.rewards
        if FLAGS.relative_rewards:
          shaped_rewards = [r0 - r1, r1 - r0]
        else:
          shaped_rewards = [r0, r1]
        if FLAGS.win_first_reward:
          shaped_rewards = [0.1 * r for r in shaped_rewards]
        time_step = time_step._replace(rewards=shaped_rewards)

      if FLAGS.win_first_reward:
        returns = env.get_state.returns()
        if returns[dqn_player_id] > returns[1 - dqn_player_id]:
          win_bonus = 10.0
        elif returns[1 - dqn_player_id] > returns[dqn_player_id]:
          win_bonus = -10.0
        else:
          win_bonus = 0.0
          
        if FLAGS.relative_rewards:
          dqn_reward = 0.1 * (returns[dqn_player_id] - returns[1 - dqn_player_id]) + win_bonus
          opp_reward = 0.1 * (returns[1 - dqn_player_id] - returns[dqn_player_id]) - win_bonus
        else:
          dqn_reward = 0.1 * returns[dqn_player_id] + win_bonus
          opp_reward = 0.1 * returns[1 - dqn_player_id] - win_bonus
          
        final_rewards = [0.0, 0.0]
        final_rewards[dqn_player_id] = dqn_reward
        final_rewards[1 - dqn_player_id] = opp_reward
        time_step = time_step._replace(rewards=final_rewards)
      elif FLAGS.relative_rewards:
        returns = env.get_state.returns()
        final_rewards = [0.0, 0.0]
        final_rewards[dqn_player_id] = returns[dqn_player_id] - returns[1 - dqn_player_id]
        final_rewards[1 - dqn_player_id] = returns[1 - dqn_player_id] - returns[dqn_player_id]
        time_step = time_step._replace(rewards=final_rewards)

      dqn_agent.step(time_step)

  if FLAGS.use_checkpoints:
    for agent in agents:
      agent.save(FLAGS.checkpoint_dir)
    logging.info(f"Saved final checkpoints to {FLAGS.checkpoint_dir}")

  logging.info("Training complete.")


if __name__ == "__main__":
  app.run(main)
