/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_PACKET_LOSS_STATS_H_
#define MODULES_RTP_RTCP_SOURCE_PACKET_LOSS_STATS_H_

#include <stdint.h>

#include <set>

namespace webrtc {

// Keeps track of statistics of packet loss including whether losses are a
// single packet or multiple packets in a row.
class PacketLossStats {
 public:
  PacketLossStats();
  ~PacketLossStats();

  // Adds a lost packet to the stats by sequence number.
  void AddLostPacket(uint16_t sequence_number);

  // Queries the number of packets that were lost by themselves, no neighboring
  // packets were lost.
  int GetSingleLossCount() const;

  // Queries the number of times that multiple packets with sequential numbers
  // were lost. This is the number of events with more than one packet lost,
  // regardless of the size of the event;
  int GetMultipleLossEventCount() const;

  // Queries the number of packets lost in multiple packet loss events. Combined
  // with the event count, this can be used to determine the average event size.
  int GetMultipleLossPacketCount() const;

 private:
  std::set<uint16_t> lost_packets_buffer_;
  std::set<uint16_t> lost_packets_wrapped_buffer_;
  int single_loss_historic_count_;
  int multiple_loss_historic_event_count_;
  int multiple_loss_historic_packet_count_;

  void ComputeLossCounts(int* out_single_loss_count,
                         int* out_multiple_loss_event_count,
                         int* out_multiple_loss_packet_count) const;
  void PruneBuffer();
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_PACKET_LOSS_STATS_H_
