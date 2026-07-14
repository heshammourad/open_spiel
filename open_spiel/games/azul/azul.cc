// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/azul/azul.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace azul {

namespace {

// Game parameters
constexpr int kDefaultPlayers = 2;

const GameType kGameType{
    /*short_name=*/"azul",
    /*long_name=*/"Azul",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kMaxPlayers,
    /*min_num_players=*/kMinPlayers,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"players", GameParameter(kDefaultPlayers)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new AzulGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

}  // namespace

AzulAction DecodeAction(Action action_id) {
  int temp = action_id;
  int destination = temp % 6;
  temp /= 6;
  int color = temp % 5;
  int source = temp / 5;
  return {source, color, destination};
}

Action EncodeAction(int source, int color, int destination) {
  return source * 30 + color * 6 + destination;
}

AzulState::AzulState(std::shared_ptr<const Game> game, int num_players)
    : State(game), num_players_(num_players) {
  phase_ = GamePhase::kChance;
  current_player_ = kChancePlayerId;
  turn_player_ = 0;
  next_round_first_player_ = 0;

  // Initialize bag with 20 of each color
  std::fill(bag_.begin(), bag_.end(), kTilesPerColor);
  std::fill(box_lid_.begin(), box_lid_.end(), 0);

  int num_factories = 2 * num_players_ + 1;
  factories_.resize(num_factories);
  for (int f = 0; f < num_factories; ++f) {
    std::fill(factories_[f].begin(), factories_[f].end(), 0);
  }

  std::fill(center_.begin(), center_.end(), 0);
  center_has_starting_player_token_ = true;

  player_boards_.resize(num_players_);
  for (int p = 0; p < num_players_; ++p) {
    player_boards_[p].score = 0;
    for (int r = 0; r < kWallSize; ++r) {
      player_boards_[p].pattern_lines[r].color = -1;
      player_boards_[p].pattern_lines[r].count = 0;
      std::fill(player_boards_[p].wall[r].begin(),
                player_boards_[p].wall[r].end(), false);
    }
    std::fill(player_boards_[p].floor_line_colors.begin(),
              player_boards_[p].floor_line_colors.end(), 0);
    player_boards_[p].floor_line_has_starting_player_token = false;
  }

  factory_fill_idx_ = 0;
  tile_fill_idx_ = 0;
}

Player AzulState::CurrentPlayer() const {
  if (phase_ == GamePhase::kTerminal) {
    return kTerminalPlayerId;
  }
  if (phase_ == GamePhase::kChance) {
    return kChancePlayerId;
  }
  return current_player_;
}

std::string AzulState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Draw tile color: ", move_id);
  }
  AzulAction action = DecodeAction(move_id);
  std::string src_str = (action.source == factories_.size())
                            ? "Center"
                            : absl::StrCat("Factory ", action.source);
  std::string dest_str = (action.destination == 5)
                             ? "Floor Line"
                             : absl::StrCat("Pattern Line ", action.destination);
  return absl::StrCat("Player ", player, " drafts color ", action.color,
                      " from ", src_str, " to ", dest_str);
}

void AzulState::RefillBagFromBoxLid() {
  int bag_sum = std::accumulate(bag_.begin(), bag_.end(), 0);
  if (bag_sum == 0) {
    for (int c = 0; c < kNumColors; ++c) {
      bag_[c] = box_lid_[c];
      box_lid_[c] = 0;
    }
  }
}

std::vector<std::pair<Action, double>> AzulState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  // Use const_cast since ChanceOutcomes is logically const, but replenishment is a lazy cache update.
  const_cast<AzulState*>(this)->RefillBagFromBoxLid();

  int bag_sum = std::accumulate(bag_.begin(), bag_.end(), 0);
  if (bag_sum == 0) {
    return outcomes;
  }

  for (int c = 0; c < kNumColors; ++c) {
    if (bag_[c] > 0) {
      outcomes.push_back({c, static_cast<double>(bag_[c]) / bag_sum});
    }
  }
  return outcomes;
}

std::string AzulState::ToString() const {
  std::string s = "";
  if (phase_ == GamePhase::kChance) {
    absl::StrAppend(&s, "Phase: Chance refilling. Currently filling factory ",
                    factory_fill_idx_, ", tile ", tile_fill_idx_, "\n");
  } else if (phase_ == GamePhase::kDraft) {
    absl::StrAppend(&s, "Phase: Drafting. Current player: ", current_player_, "\n");
  } else {
    absl::StrAppend(&s, "Phase: Terminal\n");
  }

  absl::StrAppend(&s, "Factories:\n");
  for (int f = 0; f < factories_.size(); ++f) {
    absl::StrAppend(&s, "  F", f, ": [");
    for (int c = 0; c < kNumColors; ++c) {
      absl::StrAppend(&s, " ", factories_[f][c]);
    }
    absl::StrAppend(&s, " ]\n");
  }

  absl::StrAppend(&s, "Center: [");
  for (int c = 0; c < kNumColors; ++c) {
    absl::StrAppend(&s, " ", center_[c]);
  }
  absl::StrAppend(&s, " ] (Has starting token: ",
                  center_has_starting_player_token_ ? "Yes" : "No", ")\n");

  for (int p = 0; p < num_players_; ++p) {
    absl::StrAppend(&s, "Player ", p, " Board (Score: ", player_boards_[p].score, "):\n");
    absl::StrAppend(&s, "  Pattern lines:\n");
    for (int r = 0; r < kWallSize; ++r) {
      absl::StrAppend(&s, "    Row ", r, " (Cap ", r + 1, "): color ",
                      player_boards_[p].pattern_lines[r].color, ", count ",
                      player_boards_[p].pattern_lines[r].count, "\n");
    }
    absl::StrAppend(&s, "  Wall:\n");
    for (int r = 0; r < kWallSize; ++r) {
      absl::StrAppend(&s, "    Row ", r, ":");
      for (int c = 0; c < kWallSize; ++c) {
        absl::StrAppend(&s, player_boards_[p].wall[r][c] ? " X" : " .");
      }
      absl::StrAppend(&s, "\n");
    }
    absl::StrAppend(&s, "  Floor line colors: [");
    for (int c = 0; c < kNumColors; ++c) {
      absl::StrAppend(&s, " ", player_boards_[p].floor_line_colors[c]);
    }
    absl::StrAppend(&s, " ] (Has starting token: ",
                    player_boards_[p].floor_line_has_starting_player_token ? "Yes" : "No", ")\n");
  }

  absl::StrAppend(&s, "Bag: [");
  for (int c = 0; c < kNumColors; ++c) {
    absl::StrAppend(&s, " ", bag_[c]);
  }
  absl::StrAppend(&s, " ]\nBox Lid: [");
  for (int c = 0; c < kNumColors; ++c) {
    absl::StrAppend(&s, " ", box_lid_[c]);
  }
  absl::StrAppend(&s, " ]\n");

  return s;
}

bool AzulState::IsTerminal() const {
  return phase_ == GamePhase::kTerminal;
}

std::vector<double> AzulState::Returns() const {
  std::vector<double> returns(num_players_, 0.0);
  if (!IsTerminal()) {
    return returns;
  }
  for (int p = 0; p < num_players_; ++p) {
    returns[p] = player_boards_[p].score;
  }
  return returns;
}

std::string AzulState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void AzulState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0f);
  int offset = 0;

  // 1. Factories
  for (const auto& factory : factories_) {
    for (int c = 0; c < kNumColors; ++c) {
      values[offset++] = factory[c];
    }
  }

  // 2. Center
  for (int c = 0; c < kNumColors; ++c) {
    values[offset++] = center_[c];
  }
  values[offset++] = center_has_starting_player_token_ ? 1.0f : 0.0f;

  // 3. Player boards
  for (int p = 0; p < num_players_; ++p) {
    int p_idx = (player + p) % num_players_;
    const auto& board = player_boards_[p_idx];

    values[offset++] = board.score;

    for (int r = 0; r < kWallSize; ++r) {
      int color = board.pattern_lines[r].color;
      if (color >= 0) {
        values[offset + color] = 1.0f;
      }
      offset += kNumColors;
      values[offset++] = board.pattern_lines[r].count;
    }

    for (int r = 0; r < kWallSize; ++r) {
      for (int c = 0; c < kWallSize; ++c) {
        values[offset++] = board.wall[r][c] ? 1.0f : 0.0f;
      }
    }

    for (int c = 0; c < kNumColors; ++c) {
      values[offset++] = board.floor_line_colors[c];
    }
    values[offset++] = board.floor_line_has_starting_player_token ? 1.0f : 0.0f;
  }

  // 4. Bag & Box Lid
  for (int c = 0; c < kNumColors; ++c) {
    values[offset++] = bag_[c];
  }
  for (int c = 0; c < kNumColors; ++c) {
    values[offset++] = box_lid_[c];
  }

  // 5. Active Player
  if (phase_ == GamePhase::kDraft) {
    int relative_active = (current_player_ - player + num_players_) % num_players_;
    values[offset + relative_active] = 1.0f;
  }
  offset += num_players_;

  // 6. Phase
  if (phase_ == GamePhase::kChance) {
    values[offset + 0] = 1.0f;
  } else if (phase_ == GamePhase::kDraft) {
    values[offset + 1] = 1.0f;
  } else if (phase_ == GamePhase::kTerminal) {
    values[offset + 2] = 1.0f;
  }
  offset += 3;

  SPIEL_CHECK_EQ(offset, values.size());
}

std::unique_ptr<State> AzulState::Clone() const {
  return std::unique_ptr<State>(new AzulState(*this));
}

std::vector<Action> AzulState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  }
  std::vector<Action> actions;
  if (phase_ != GamePhase::kDraft) {
    return actions;
  }

  int num_factories = factories_.size();
  const auto& board = player_boards_[current_player_];

  for (int source = 0; source <= num_factories; ++source) {
    for (int color = 0; color < kNumColors; ++color) {
      bool has_tiles = false;
      if (source < num_factories) {
        has_tiles = (factories_[source][color] > 0);
      } else {
        has_tiles = (center_[color] > 0);
      }

      if (!has_tiles) {
        continue;
      }

      // Check destinations
      for (int dest = 0; dest <= kWallSize; ++dest) {
        if (dest < kWallSize) { // Pattern Line Row dest
          // Wall restriction: the color cannot be in this row on the wall
          int col_on_wall = (color + dest) % 5;
          if (board.wall[dest][col_on_wall]) {
            continue;
          }

          // Line empty or has the same color and is not full
          const auto& line = board.pattern_lines[dest];
          if (line.color == -1 || (line.color == color && line.count < dest + 1)) {
            actions.push_back(EncodeAction(source, color, dest));
          }
        } else { // Floor Line
          actions.push_back(EncodeAction(source, color, dest));
        }
      }
    }
  }

  std::sort(actions.begin(), actions.end());
  return actions;
}

int AzulState::CalculatePlacementScore(int player, int row, int col) const {
  const auto& wall = player_boards_[player].wall;

  // Check horizontal connection
  int left = 0;
  for (int c = col - 1; c >= 0; --c) {
    if (wall[row][c]) {
      left++;
    } else {
      break;
    }
  }

  int right = 0;
  for (int c = col + 1; c < kWallSize; ++c) {
    if (wall[row][c]) {
      right++;
    } else {
      break;
    }
  }

  int h_count = 1 + left + right;

  // Check vertical connection
  int up = 0;
  for (int r = row - 1; r >= 0; --r) {
    if (wall[r][col]) {
      up++;
    } else {
      break;
    }
  }

  int down = 0;
  for (int r = row + 1; r < kWallSize; ++r) {
    if (wall[r][col]) {
      down++;
    } else {
      break;
    }
  }

  int v_count = 1 + up + down;

  int points = 0;
  if (h_count > 1 && v_count > 1) {
    points = h_count + v_count;
  } else if (h_count > 1) {
    points = h_count;
  } else if (v_count > 1) {
    points = v_count;
  } else {
    points = 1;
  }

  return points;
}

bool AzulState::CheckGameEnd() const {
  for (int p = 0; p < num_players_; ++p) {
    const auto& wall = player_boards_[p].wall;
    for (int r = 0; r < kWallSize; ++r) {
      bool row_complete = true;
      for (int c = 0; c < kWallSize; ++c) {
        if (!wall[r][c]) {
          row_complete = false;
          break;
        }
      }
      if (row_complete) {
        return true;
      }
    }
  }
  return false;
}

void AzulState::ApplyFinalBonuses() {
  for (int p = 0; p < num_players_; ++p) {
    auto& board = player_boards_[p];

    // 1. Horizontal completed rows (+2 points)
    for (int r = 0; r < kWallSize; ++r) {
      bool row_complete = true;
      for (int c = 0; c < kWallSize; ++c) {
        if (!board.wall[r][c]) {
          row_complete = false;
          break;
        }
      }
      if (row_complete) {
        board.score += 2;
      }
    }

    // 2. Vertical completed columns (+7 points)
    for (int c = 0; c < kWallSize; ++c) {
      bool col_complete = true;
      for (int r = 0; r < kWallSize; ++r) {
        if (!board.wall[r][c]) {
          col_complete = false;
          break;
        }
      }
      if (col_complete) {
        board.score += 7;
      }
    }

    // 3. Completed colors (+10 points)
    for (int color = 0; color < kNumColors; ++color) {
      bool color_complete = true;
      for (int r = 0; r < kWallSize; ++r) {
        int col = (color + r) % 5;
        if (!board.wall[r][col]) {
          color_complete = false;
          break;
        }
      }
      if (color_complete) {
        board.score += 10;
      }
    }
  }
}

void AzulState::TileAndScoreWall() {
  for (int p = 0; p < num_players_; ++p) {
    auto& board = player_boards_[p];

    // 1. Wall Tiling
    for (int r = 0; r < kWallSize; ++r) {
      if (board.pattern_lines[r].count == r + 1) {
        int color = board.pattern_lines[r].color;
        int col = (color + r) % 5;
        board.wall[r][col] = true;

        int points = CalculatePlacementScore(p, r, col);
        board.score += points;

        // Discard remaining tiles to box lid
        box_lid_[color] += r;

        // Clear pattern line
        board.pattern_lines[r].color = -1;
        board.pattern_lines[r].count = 0;
      }
    }

    // 2. Floor Line penalties
    int floor_count = board.FloorLineCount();
    if (floor_count > 0) {
      int penalty = 0;
      for (int i = 0; i < std::min(floor_count, kFloorLineCapacity); ++i) {
        if (i == 0 || i == 1) {
          penalty += 1;
        } else if (i >= 2 && i <= 4) {
          penalty += 2;
        } else {
          penalty += 3;
        }
      }

      board.score -= penalty;
      if (board.score < 0) {
        board.score = 0;
      }

      // Move floor tiles to box lid
      for (int c = 0; c < kNumColors; ++c) {
        box_lid_[c] += board.floor_line_colors[c];
        board.floor_line_colors[c] = 0;
      }
      board.floor_line_has_starting_player_token = false;
    }
  }
}

void AzulState::DoApplyAction(Action move_id) {
  if (phase_ == GamePhase::kChance) {
    int color = move_id;
    bag_[color]--;
    factories_[factory_fill_idx_][color]++;
    tile_fill_idx_++;

    if (tile_fill_idx_ == kFactoryCapacity) {
      tile_fill_idx_ = 0;
      factory_fill_idx_++;
    }

    // Check if factories are completely filled
    int num_factories = factories_.size();
    RefillBagFromBoxLid();
    int bag_sum = std::accumulate(bag_.begin(), bag_.end(), 0);

    if (factory_fill_idx_ == num_factories || bag_sum == 0) {
      // Done refilling factories, transition to drafting
      phase_ = GamePhase::kDraft;
      current_player_ = turn_player_;
    }
    return;
  }

  // Phase: Draft
  AzulAction action = DecodeAction(move_id);
  int p = current_player_;
  auto& board = player_boards_[p];

  int count = 0;
  int num_factories = factories_.size();

  if (action.source < num_factories) {
    // Draft from a factory
    count = factories_[action.source][action.color];
    factories_[action.source][action.color] = 0;

    // Remaining tiles in the factory go to center
    for (int c = 0; c < kNumColors; ++c) {
      if (factories_[action.source][c] > 0) {
        center_[c] += factories_[action.source][c];
        factories_[action.source][c] = 0;
      }
    }
  } else {
    // Draft from the center
    count = center_[action.color];
    center_[action.color] = 0;

    // Check if this player gets the starting player token
    if (center_has_starting_player_token_) {
      center_has_starting_player_token_ = false;
      board.floor_line_has_starting_player_token = true;
      next_round_first_player_ = p;
    }
  }

  // Place tiles onto destination
  if (action.destination < kWallSize) {
    // Place on pattern line Row
    auto& line = board.pattern_lines[action.destination];
    if (line.color == -1) {
      line.color = action.color;
    }
    int capacity = action.destination + 1;
    int space = capacity - line.count;

    if (count <= space) {
      line.count += count;
    } else {
      line.count = capacity;
      int excess = count - space;
      board.floor_line_colors[action.color] += excess;
    }
  } else {
    // Place directly on the floor line
    board.floor_line_colors[action.color] += count;
  }

  // Check if round ends (no tiles left in any factories or center)
  bool round_over = true;
  for (int f = 0; f < num_factories; ++f) {
    for (int c = 0; c < kNumColors; ++c) {
      if (factories_[f][c] > 0) {
        round_over = false;
        break;
      }
    }
    if (!round_over) break;
  }
  if (round_over) {
    for (int c = 0; c < kNumColors; ++c) {
      if (center_[c] > 0) {
        round_over = false;
        break;
      }
    }
  }

  if (round_over) {
    // Wall-Tiling and Scoring
    TileAndScoreWall();

    // Check game end condition
    if (CheckGameEnd()) {
      ApplyFinalBonuses();
      phase_ = GamePhase::kTerminal;
      current_player_ = kTerminalPlayerId;
    } else {
      // Refill for next round
      turn_player_ = next_round_first_player_;
      phase_ = GamePhase::kChance;
      current_player_ = kChancePlayerId;
      factory_fill_idx_ = 0;
      tile_fill_idx_ = 0;
      center_has_starting_player_token_ = true;

      // Handle edge case where the bag and discard are empty immediately at the start of a round
      RefillBagFromBoxLid();
      int bag_sum = std::accumulate(bag_.begin(), bag_.end(), 0);
      if (bag_sum == 0) {
        phase_ = GamePhase::kDraft;
        current_player_ = turn_player_;
      }
    }
  } else {
    // Shift to next player
    current_player_ = (current_player_ + 1) % num_players_;
  }
}

AzulGame::AzulGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")) {}

int AzulGame::NumDistinctActions() const {
  return 300;
}

std::unique_ptr<State> AzulGame::NewInitialState() const {
  return std::unique_ptr<State>(new AzulState(shared_from_this(), num_players_));
}

int AzulGame::MaxChanceOutcomes() const {
  return kNumColors;
}

int AzulGame::MaxGameLength() const {
  return 2000;
}

int AzulGame::MaxChanceNodesInHistory() const {
  return 2000;
}

int AzulGame::NumPlayers() const {
  return num_players_;
}

double AzulGame::MinUtility() const {
  return 0.0;
}

absl::optional<double> AzulGame::UtilitySum() const {
  return absl::nullopt;
}

double AzulGame::MaxUtility() const {
  // Absolute upper bound of scoring is roughly 300 points
  return 300.0;
}

std::vector<int> AzulGame::ObservationTensorShape() const {
  int size = (2 * num_players_ + 1) * 5 + 6 + num_players_ * 62 + 10 + num_players_ + 3;
  return {size};
}

}  // namespace azul
}  // namespace open_spiel
