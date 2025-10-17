/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#ifndef MODULES_RTP_RTCP_INCLUDE_RTCP_STATISTICS_H_
#define MODULES_RTP_RTCP_INCLUDE_RTCP_STATISTICS_H_

#include <stdint.h>

#include "absl/strings/string_view.h"

namespace webrtc {

// Statistics for RTCP packet types.
struct RtcpPacketTypeCounter {
  RtcpPacketTypeCounter()
      : nack_packets(0),
        fir_packets(0),
        pli_packets(0),
        nack_requests(0),
        unique_nack_requests(0) {}

  void Add(const RtcpPacketTypeCounter& other) {
    nack_packets += other.nack_packets;
    fir_packets += other.fir_packets;
    pli_packets += other.pli_packets;
    nack_requests += other.nack_requests;
    unique_nack_requests += other.unique_nack_requests;
  }

  void Subtract(const RtcpPacketTypeCounter& other) {
    nack_packets -= other.nack_packets;
    fir_packets -= other.fir_packets;
    pli_packets -= other.pli_packets;
    nack_requests -= other.nack_requests;
    unique_nack_requests -= other.unique_nack_requests;
  }

  int UniqueNackRequestsInPercent() const {
    if (nack_requests == 0) {
      return 0;
    }
    return static_cast<int>((unique_nack_requests * 100.0f / nack_requests) +
                            0.5f);
  }

  uint32_t nack_packets;          // Number of RTCP NACK packets.
  uint32_t fir_packets;           // Number of RTCP FIR packets.
  uint32_t pli_packets;           // Number of RTCP PLI packets.
  uint32_t nack_requests;         // Number of NACKed RTP packets.
  uint32_t unique_nack_requests;  // Number of unique NACKed RTP packets.
};

class RtcpPacketTypeCounterObserver {
 public:
  virtual ~RtcpPacketTypeCounterObserver() {}
  virtual void RtcpPacketTypesCounterUpdated(
      uint32_t ssrc,
      const RtcpPacketTypeCounter& packet_counter) = 0;
};

// Invoked for each cname passed in RTCP SDES blocks.
class RtcpCnameCallback {
 public:
  virtual ~RtcpCnameCallback() = default;

  virtual void OnCname(uint32_t ssrc, absl::string_view cname) = 0;
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_INCLUDE_RTCP_STATISTICS_H_
