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

#ifndef OPEN_SPIEL_GAMES_AZUL_H_
#define OPEN_SPIEL_GAMES_AZUL_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace azul {

// Constants
inline constexpr int kNumColors = 5;
inline constexpr int kMaxPlayers = 4;
inline constexpr int kMinPlayers = 2;
inline constexpr int kTilesPerColor = 20;
inline constexpr int kFactoryCapacity = 4;
inline constexpr int kFloorLineCapacity = 7;
inline constexpr int kWallSize = 5;

enum class GamePhase {
  kChance,      // Refilling factories
  kDraft,       // Players taking tiles
  kTerminal,    // Game is over
};

struct PatternLine {
  int color = -1;  // -1 if empty, otherwise 0..4
  int count = 0;   // 0 to row_index + 1
};

struct PlayerBoard {
  int score = 0;
  std::array<PatternLine, kWallSize> pattern_lines;
  std::array<std::array<bool, kWallSize>, kWallSize> wall;
  std::array<int, kNumColors> floor_line_colors;
  bool floor_line_has_starting_player_token = false;

  int FloorLineCount() const {
    int count = floor_line_has_starting_player_token ? 1 : 0;
    for (int c = 0; c < kNumColors; ++c) {
      count += floor_line_colors[c];
    }
    return count;
  }
};

struct AzulAction {
  int source;
  int color;
  int destination;
};

AzulAction DecodeAction(Action action_id);
Action EncodeAction(int source, int color, int destination);

class AzulGame;

class AzulState : public State {
 public:
  AzulState(std::shared_ptr<const Game> game, int num_players);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void InitializeRound();
  void RefillBagFromBoxLid();
  void TileAndScoreWall();
  int CalculatePlacementScore(int player, int row, int col) const;
  bool CheckGameEnd() const;
  void ApplyFinalBonuses();

  int num_players_;
  GamePhase phase_;
  Player current_player_;
  Player turn_player_;  // Tracks active player during chance events

  std::array<int, kNumColors> bag_;
  std::array<int, kNumColors> box_lid_;
  std::vector<std::array<int, kNumColors>> factories_;
  std::array<int, kNumColors> center_;
  bool center_has_starting_player_token_;
  int next_round_first_player_;

  // Chance refill indexing
  int factory_fill_idx_ = 0;
  int tile_fill_idx_ = 0;

  std::vector<PlayerBoard> player_boards_;
};

class AzulGame : public Game {
 public:
  explicit AzulGame(const GameParameters& params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  absl::optional<double> UtilitySum() const override;
  double MaxUtility() const override;
  std::vector<int> ObservationTensorShape() const override;

 private:
  int num_players_;
};

}  // namespace azul
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_AZUL_H_
