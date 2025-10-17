/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
#ifndef MODULES_PACING_RTP_PACKET_PACER_H_
#define MODULES_PACING_RTP_PACKET_PACER_H_


#include <optional>
#include <vector>

#include "api/transport/network_types.h"
#include "api/units/data_rate.h"
#include "api/units/data_size.h"
#include "api/units/time_delta.h"
#include "api/units/timestamp.h"

namespace webrtc {

class RtpPacketPacer {
 public:
  virtual ~RtpPacketPacer() = default;

  virtual void CreateProbeClusters(
      std::vector<ProbeClusterConfig> probe_cluster_configs) = 0;

  // Temporarily pause all sending.
  virtual void Pause() = 0;

  // Resume sending packets.
  virtual void Resume() = 0;

  virtual void SetCongested(bool congested) = 0;

  // Sets the pacing rates. Must be called once before packets can be sent.
  virtual void SetPacingRates(DataRate pacing_rate, DataRate padding_rate) = 0;

  // Time since the oldest packet currently in the queue was added.
  virtual TimeDelta OldestPacketWaitTime() const = 0;

  // Sum of payload + padding bytes of all packets currently in the pacer queue.
  virtual DataSize QueueSizeData() const = 0;

  // Returns the time when the first packet was sent.
  virtual std::optional<Timestamp> FirstSentPacketTime() const = 0;

  // Returns the expected number of milliseconds it will take to send the
  // current packets in the queue, given the current size and bitrate, ignoring
  // priority.
  virtual TimeDelta ExpectedQueueTime() const = 0;

  // Set the average upper bound on pacer queuing delay. The pacer may send at
  // a higher rate than what was configured via SetPacingRates() in order to
  // keep ExpectedQueueTimeMs() below `limit_ms` on average.
  virtual void SetQueueTimeLimit(TimeDelta limit) = 0;

  // Currently audio traffic is not accounted by pacer and passed through.
  // With the introduction of audio BWE audio traffic will be accounted for
  // the pacer budget calculation. The audio traffic still will be injected
  // at high priority.
  virtual void SetAccountForAudioPackets(bool account_for_audio) = 0;
  virtual void SetIncludeOverhead() = 0;
  virtual void SetTransportOverhead(DataSize overhead_per_packet) = 0;
};

}  // namespace webrtc
#endif  // MODULES_PACING_RTP_PACKET_PACER_H_
