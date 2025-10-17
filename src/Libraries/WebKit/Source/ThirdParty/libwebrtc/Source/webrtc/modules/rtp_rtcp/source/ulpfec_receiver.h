/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_ULPFEC_RECEIVER_H_
#define MODULES_RTP_RTCP_SOURCE_ULPFEC_RECEIVER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "api/sequence_checker.h"
#include "modules/rtp_rtcp/include/recovered_packet_receiver.h"
#include "modules/rtp_rtcp/include/rtp_header_extension_map.h"
#include "modules/rtp_rtcp/source/forward_error_correction.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "rtc_base/system/no_unique_address.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

struct FecPacketCounter {
  FecPacketCounter() = default;
  size_t num_packets = 0;  // Number of received packets.
  size_t num_bytes = 0;
  size_t num_fec_packets = 0;  // Number of received FEC packets.
  size_t num_recovered_packets =
      0;  // Number of recovered media packets using FEC.
  // Time when first packet is received.
  Timestamp first_packet_time = Timestamp::MinusInfinity();
};

class UlpfecReceiver {
 public:
  UlpfecReceiver(uint32_t ssrc,
                 int ulpfec_payload_type,
                 RecoveredPacketReceiver* callback,
                 Clock* clock);
  ~UlpfecReceiver();

  int ulpfec_payload_type() const { return ulpfec_payload_type_; }

  bool AddReceivedRedPacket(const RtpPacketReceived& rtp_packet);

  void ProcessReceivedFec();

  FecPacketCounter GetPacketCounter() const;

 private:
  const uint32_t ssrc_;
  const int ulpfec_payload_type_;
  Clock* const clock_;

  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_;
  RecoveredPacketReceiver* const recovered_packet_callback_;
  const std::unique_ptr<ForwardErrorCorrection> fec_;
  // TODO(nisse): The AddReceivedRedPacket method adds one or two packets to
  // this list at a time, after which it is emptied by ProcessReceivedFec. It
  // will make things simpler to merge AddReceivedRedPacket and
  // ProcessReceivedFec into a single method, and we can then delete this list.
  std::vector<std::unique_ptr<ForwardErrorCorrection::ReceivedPacket>>
      received_packets_ RTC_GUARDED_BY(&sequence_checker_);
  ForwardErrorCorrection::RecoveredPacketList recovered_packets_
      RTC_GUARDED_BY(&sequence_checker_);
  FecPacketCounter packet_counter_ RTC_GUARDED_BY(&sequence_checker_);
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_ULPFEC_RECEIVER_H_
