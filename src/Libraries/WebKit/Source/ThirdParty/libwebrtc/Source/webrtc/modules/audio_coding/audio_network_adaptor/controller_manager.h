/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CONTROLLER_MANAGER_H_
#define MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CONTROLLER_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/audio_network_adaptor/controller.h"

namespace webrtc {

class DebugDumpWriter;

class ControllerManager {
 public:
  virtual ~ControllerManager() = default;

  // Sort controllers based on their significance.
  virtual std::vector<Controller*> GetSortedControllers(
      const Controller::NetworkMetrics& metrics) = 0;

  virtual std::vector<Controller*> GetControllers() const = 0;
};

class ControllerManagerImpl final : public ControllerManager {
 public:
  struct Config {
    Config(int min_reordering_time_ms, float min_reordering_squared_distance);
    ~Config();
    // Least time since last reordering for a new reordering to be made.
    int min_reordering_time_ms;
    // Least squared distance from last scoring point for a new reordering to be
    // made.
    float min_reordering_squared_distance;
  };

  static std::unique_ptr<ControllerManager> Create(
      absl::string_view config_string,
      size_t num_encoder_channels,
      rtc::ArrayView<const int> encoder_frame_lengths_ms,
      int min_encoder_bitrate_bps,
      size_t intial_channels_to_encode,
      int initial_frame_length_ms,
      int initial_bitrate_bps,
      bool initial_fec_enabled,
      bool initial_dtx_enabled);

  static std::unique_ptr<ControllerManager> Create(
      absl::string_view config_string,
      size_t num_encoder_channels,
      rtc::ArrayView<const int> encoder_frame_lengths_ms,
      int min_encoder_bitrate_bps,
      size_t intial_channels_to_encode,
      int initial_frame_length_ms,
      int initial_bitrate_bps,
      bool initial_fec_enabled,
      bool initial_dtx_enabled,
      DebugDumpWriter* debug_dump_writer);

  explicit ControllerManagerImpl(const Config& config);

  // Dependency injection for testing.
  ControllerManagerImpl(
      const Config& config,
      std::vector<std::unique_ptr<Controller>> controllers,
      const std::map<const Controller*, std::pair<int, float>>&
          chracteristic_points);

  ~ControllerManagerImpl() override;

  ControllerManagerImpl(const ControllerManagerImpl&) = delete;
  ControllerManagerImpl& operator=(const ControllerManagerImpl&) = delete;

  // Sort controllers based on their significance.
  std::vector<Controller*> GetSortedControllers(
      const Controller::NetworkMetrics& metrics) override;

  std::vector<Controller*> GetControllers() const override;

 private:
  // Scoring point is a subset of NetworkMetrics that is used for comparing the
  // significance of controllers.
  struct ScoringPoint {
    // TODO(eladalon): Do we want to experiment with RPLR-based scoring?
    ScoringPoint(int uplink_bandwidth_bps, float uplink_packet_loss_fraction);

    // Calculate the normalized [0,1] distance between two scoring points.
    float SquaredDistanceTo(const ScoringPoint& scoring_point) const;

    int uplink_bandwidth_bps;
    float uplink_packet_loss_fraction;
  };

  const Config config_;

  std::vector<std::unique_ptr<Controller>> controllers_;

  std::optional<int64_t> last_reordering_time_ms_;
  ScoringPoint last_scoring_point_;

  std::vector<Controller*> default_sorted_controllers_;

  std::vector<Controller*> sorted_controllers_;

  // `scoring_points_` saves the scoring points of various
  // controllers.
  std::map<const Controller*, ScoringPoint> controller_scoring_points_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_AUDIO_NETWORK_ADAPTOR_CONTROLLER_MANAGER_H_
