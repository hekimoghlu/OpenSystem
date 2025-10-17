/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#ifndef MODULES_CONGESTION_CONTROLLER_PCC_RTT_TRACKER_H_
#define MODULES_CONGESTION_CONTROLLER_PCC_RTT_TRACKER_H_

#include <vector>

#include "api/transport/network_types.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"

namespace webrtc {
namespace pcc {

class RttTracker {
 public:
  RttTracker(TimeDelta initial_rtt, double alpha);
  // Updates RTT estimate.
  void OnPacketsFeedback(const std::vector<PacketResult>& packet_feedbacks,
                         Timestamp feedback_received_time);
  TimeDelta GetRtt() const;

 private:
  TimeDelta rtt_estimate_;
  double alpha_;
};

}  // namespace pcc
}  // namespace webrtc

#endif  // MODULES_CONGESTION_CONTROLLER_PCC_RTT_TRACKER_H_
