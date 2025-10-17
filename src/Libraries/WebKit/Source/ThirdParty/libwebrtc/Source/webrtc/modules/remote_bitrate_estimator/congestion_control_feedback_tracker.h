/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_CONGESTION_CONTROL_FEEDBACK_TRACKER_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_CONGESTION_CONTROL_FEEDBACK_TRACKER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "api/units/timestamp.h"
#include "modules/rtp_rtcp/source/rtcp_packet/congestion_control_feedback.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/network/ecn_marking.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"
namespace webrtc {

// CongestionControlFeedbackTracker is reponsible for creating and keeping track
// of feedback sent for a specific SSRC when feedback is sent according to
// https://datatracker.ietf.org/doc/rfc8888/
class CongestionControlFeedbackTracker {
 public:
  CongestionControlFeedbackTracker() = default;

  void ReceivedPacket(const RtpPacketReceived& packet);

  // Adds received packets to `packet_feedback`
  // RTP sequence numbers are continous from the last created feedback unless
  // reordering has occured between feedback packets. If so, the sequence
  // number range may overlap with previousely sent feedback.
  void AddPacketsToFeedback(
      Timestamp feedback_time,
      std::vector<rtcp::CongestionControlFeedback::PacketInfo>&
          packet_feedback);

 private:
  struct PacketInfo {
    uint32_t ssrc;
    int64_t unwrapped_sequence_number = 0;
    Timestamp arrival_time;
    rtc::EcnMarking ecn = rtc::EcnMarking::kNotEct;
  };

  std::optional<int64_t> last_sequence_number_in_feedback_;
  SeqNumUnwrapper<uint16_t> unwrapper_;

  std::vector<PacketInfo> packets_;
};

}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_CONGESTION_CONTROL_FEEDBACK_TRACKER_H_
