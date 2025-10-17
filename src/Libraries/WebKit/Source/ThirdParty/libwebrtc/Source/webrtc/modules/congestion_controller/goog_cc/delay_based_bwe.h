/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#ifndef MODULES_CONGESTION_CONTROLLER_GOOG_CC_DELAY_BASED_BWE_H_
#define MODULES_CONGESTION_CONTROLLER_GOOG_CC_DELAY_BASED_BWE_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <vector>

#include "api/field_trials_view.h"
#include "api/network_state_predictor.h"
#include "api/transport/bandwidth_usage.h"
#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "modules/congestion_controller/goog_cc/delay_increase_detector_interface.h"
#include "modules/congestion_controller/goog_cc/inter_arrival_delta.h"
#include "modules/congestion_controller/goog_cc/link_capacity_estimator.h"
#include "modules/congestion_controller/goog_cc/probe_bitrate_estimator.h"
#include "modules/remote_bitrate_estimator/aimd_rate_control.h"
#include "modules/remote_bitrate_estimator/inter_arrival.h"
#include "rtc_base/experiments/struct_parameters_parser.h"
#include "rtc_base/race_checker.h"

namespace webrtc {
class RtcEventLog;

struct BweSeparateAudioPacketsSettings {
  static constexpr char kKey[] = "WebRTC-Bwe-SeparateAudioPackets";

  BweSeparateAudioPacketsSettings() = default;
  explicit BweSeparateAudioPacketsSettings(
      const FieldTrialsView* key_value_config);

  bool enabled = false;
  int packet_threshold = 10;
  TimeDelta time_threshold = TimeDelta::Seconds(1);

  std::unique_ptr<StructParametersParser> Parser();
};

class DelayBasedBwe {
 public:
  struct Result {
    Result();
    ~Result() = default;
    bool updated;
    bool probe;
    DataRate target_bitrate = DataRate::Zero();
    bool recovered_from_overuse;
    BandwidthUsage delay_detector_state;
  };

  explicit DelayBasedBwe(const FieldTrialsView* key_value_config,
                         RtcEventLog* event_log,
                         NetworkStatePredictor* network_state_predictor);

  DelayBasedBwe() = delete;
  DelayBasedBwe(const DelayBasedBwe&) = delete;
  DelayBasedBwe& operator=(const DelayBasedBwe&) = delete;

  virtual ~DelayBasedBwe();

  Result IncomingPacketFeedbackVector(
      const TransportPacketsFeedback& msg,
      std::optional<DataRate> acked_bitrate,
      std::optional<DataRate> probe_bitrate,
      std::optional<NetworkStateEstimate> network_estimate,
      bool in_alr);
  void OnRttUpdate(TimeDelta avg_rtt);
  bool LatestEstimate(std::vector<uint32_t>* ssrcs, DataRate* bitrate) const;
  void SetStartBitrate(DataRate start_bitrate);
  void SetMinBitrate(DataRate min_bitrate);
  TimeDelta GetExpectedBwePeriod() const;
  DataRate TriggerOveruse(Timestamp at_time,
                          std::optional<DataRate> link_capacity);
  DataRate last_estimate() const { return prev_bitrate_; }
  BandwidthUsage last_state() const { return prev_state_; }

 private:
  friend class GoogCcStatePrinter;
  void IncomingPacketFeedback(const PacketResult& packet_feedback,
                              Timestamp at_time);
  Result MaybeUpdateEstimate(std::optional<DataRate> acked_bitrate,
                             std::optional<DataRate> probe_bitrate,
                             std::optional<NetworkStateEstimate> state_estimate,
                             bool recovered_from_overuse,
                             bool in_alr,
                             Timestamp at_time);
  // Updates the current remote rate estimate and returns true if a valid
  // estimate exists.
  bool UpdateEstimate(Timestamp at_time,
                      std::optional<DataRate> acked_bitrate,
                      DataRate* target_rate);

  rtc::RaceChecker network_race_;
  RtcEventLog* const event_log_;
  const FieldTrialsView* const key_value_config_;

  // Alternatively, run two separate overuse detectors for audio and video,
  // and fall back to the audio one if we haven't seen a video packet in a
  // while.
  BweSeparateAudioPacketsSettings separate_audio_;
  int64_t audio_packets_since_last_video_;
  Timestamp last_video_packet_recv_time_;

  NetworkStatePredictor* network_state_predictor_;
  std::unique_ptr<InterArrival> video_inter_arrival_;
  std::unique_ptr<InterArrivalDelta> video_inter_arrival_delta_;
  std::unique_ptr<DelayIncreaseDetectorInterface> video_delay_detector_;
  std::unique_ptr<InterArrival> audio_inter_arrival_;
  std::unique_ptr<InterArrivalDelta> audio_inter_arrival_delta_;
  std::unique_ptr<DelayIncreaseDetectorInterface> audio_delay_detector_;
  DelayIncreaseDetectorInterface* active_delay_detector_;

  Timestamp last_seen_packet_;
  bool uma_recorded_;
  AimdRateControl rate_control_;
  DataRate prev_bitrate_;
  BandwidthUsage prev_state_;
};

}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_GOOG_CC_DELAY_BASED_BWE_H_
