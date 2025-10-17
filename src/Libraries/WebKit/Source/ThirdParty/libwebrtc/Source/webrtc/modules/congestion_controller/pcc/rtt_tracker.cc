/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "modules/congestion_controller/pcc/rtt_tracker.h"

#include <algorithm>

namespace webrtc {
namespace pcc {

RttTracker::RttTracker(TimeDelta initial_rtt, double alpha)
    : rtt_estimate_(initial_rtt), alpha_(alpha) {}

void RttTracker::OnPacketsFeedback(
    const std::vector<PacketResult>& packet_feedbacks,
    Timestamp feedback_received_time) {
  TimeDelta packet_rtt = TimeDelta::MinusInfinity();
  for (const PacketResult& packet_result : packet_feedbacks) {
    if (!packet_result.IsReceived())
      continue;
    packet_rtt = std::max<TimeDelta>(
        packet_rtt,
        feedback_received_time - packet_result.sent_packet.send_time);
  }
  if (packet_rtt.IsFinite())
    rtt_estimate_ = (1 - alpha_) * rtt_estimate_ + alpha_ * packet_rtt;
}

TimeDelta RttTracker::GetRtt() const {
  return rtt_estimate_;
}

}  // namespace pcc
}  // namespace webrtc
