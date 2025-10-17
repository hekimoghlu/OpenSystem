/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
// This class estimates the incoming available bandwidth.

#ifndef MODULES_REMOTE_BITRATE_ESTIMATOR_INCLUDE_REMOTE_BITRATE_ESTIMATOR_H_
#define MODULES_REMOTE_BITRATE_ESTIMATOR_INCLUDE_REMOTE_BITRATE_ESTIMATOR_H_

#include <cstdint>
#include <vector>

#include "api/units/data_rate.h"
#include "api/units/time_delta.h"
#include "modules/include/module_common_types.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"

namespace webrtc {

class Clock;

// RemoteBitrateObserver is used to signal changes in bitrate estimates for
// the incoming streams.
class RemoteBitrateObserver {
 public:
  // Called when a receive channel group has a new bitrate estimate for the
  // incoming streams.
  virtual void OnReceiveBitrateChanged(const std::vector<uint32_t>& ssrcs,
                                       uint32_t bitrate) = 0;

  virtual ~RemoteBitrateObserver() {}
};

class RemoteBitrateEstimator : public CallStatsObserver {
 public:
  ~RemoteBitrateEstimator() override {}

  // Called for each incoming packet. Updates the incoming payload bitrate
  // estimate and the over-use detector. If an over-use is detected the
  // remote bitrate estimate will be updated.
  virtual void IncomingPacket(const RtpPacketReceived& rtp_packet) = 0;

  // Removes all data for `ssrc`.
  virtual void RemoveStream(uint32_t ssrc) = 0;

  // Returns latest estimate or DataRate::Zero() if estimation is unavailable.
  virtual DataRate LatestEstimate() const = 0;

  virtual TimeDelta Process() = 0;

 protected:
  static constexpr TimeDelta kProcessInterval = TimeDelta::Millis(500);
  static constexpr TimeDelta kStreamTimeOut = TimeDelta::Seconds(2);
};

}  // namespace webrtc

#endif  // MODULES_REMOTE_BITRATE_ESTIMATOR_INCLUDE_REMOTE_BITRATE_ESTIMATOR_H_
