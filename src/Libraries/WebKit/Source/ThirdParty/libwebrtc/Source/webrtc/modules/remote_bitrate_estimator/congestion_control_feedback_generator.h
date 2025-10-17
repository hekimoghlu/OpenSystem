/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_CONGESTION_CONTROL_FEEDBACK_GENERATOR_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_CONGESTION_CONTROL_FEEDBACK_GENERATOR_H_

#include <cstdint>
#include <map>
#include <optional>

#include "api/environment/environment.h"
#include "api/sequence_checker.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"
#include "modules/remote_bitrate_estimator/congestion_control_feedback_tracker.h"
#include "modules/remote_bitrate_estimator/rtp_transport_feedback_generator.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/experiments/field_trial_parser.h"

namespace webrtc {

// The class is responsible for generating RTCP feedback packets based on
// incoming media packets. Feedback format will comply with RFC 8888.
// https://datatracker.ietf.org/doc/rfc8888/

// Feedback should not use more than 5% of the configured send bandwidth
// estimate. Min and max duration between feedback is configurable using field
// trials, but per default, min is 25ms and max is 250ms.
// If possible, given the other constraints, feedback will be sent when a packet
// with marker bit is received in order to provide feedback as soon as possible
// after receiving a complete video frame. If no packet with marker bit is
// received, feedback can be delayed up to 25ms after the first packet since the
// last sent feedback. On good networks, this means that a sender may receive
// feedback for every sent frame.
class CongestionControlFeedbackGenerator
    : public RtpTransportFeedbackGenerator {
 public:
  CongestionControlFeedbackGenerator(
      const Environment& env,
      RtpTransportFeedbackGenerator::RtcpSender feedback_sender);
  ~CongestionControlFeedbackGenerator() = default;

  void OnReceivedPacket(const RtpPacketReceived& packet) override;

  void OnSendBandwidthEstimateChanged(DataRate estimate) override;

  TimeDelta Process(Timestamp now) override;

  void SetTransportOverhead(DataSize overhead_per_packet) override;

 private:
  Timestamp NextFeedbackTime() const RTC_RUN_ON(sequence_checker_);

  void SendFeedback(Timestamp now) RTC_RUN_ON(sequence_checker_);

  void CalculateNextPossibleSendTime(DataSize feedback_size, Timestamp now)
      RTC_RUN_ON(sequence_checker_);

  const Environment env_;
  SequenceChecker sequence_checker_;
  const RtcpSender rtcp_sender_;

  FieldTrialParameter<TimeDelta> min_time_between_feedback_;
  FieldTrialParameter<TimeDelta> max_time_to_wait_for_packet_with_marker_;
  FieldTrialParameter<TimeDelta> max_time_between_feedback_;

  DataRate max_feedback_rate_ = DataRate::KilobitsPerSec(1000);
  DataSize packet_overhead_ = DataSize::Zero();
  DataSize send_rate_debt_ = DataSize::Zero();

  std::map</*ssrc=*/uint32_t, CongestionControlFeedbackTracker>
      feedback_trackers_;

  // std::vector<PacketInfo> packets_;
  Timestamp last_feedback_sent_time_ = Timestamp::Zero();
  std::optional<Timestamp> first_arrival_time_since_feedback_;
  bool marker_bit_seen_ = false;
  Timestamp next_possible_feedback_send_time_ = Timestamp::Zero();
};

}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_CONGESTION_CONTROL_FEEDBACK_GENERATOR_H_
