/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#ifndef MODULES_RTP_RTCP_INCLUDE_FLEXFEC_RECEIVER_H_
#define MODULES_RTP_RTCP_INCLUDE_FLEXFEC_RECEIVER_H_

#include <stdint.h>

#include <memory>

#include "api/sequence_checker.h"
#include "api/units/timestamp.h"
#include "modules/rtp_rtcp/include/recovered_packet_receiver.h"
#include "modules/rtp_rtcp/source/forward_error_correction.h"
#include "modules/rtp_rtcp/source/rtp_packet_received.h"
#include "modules/rtp_rtcp/source/ulpfec_receiver.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class Clock;

class FlexfecReceiver {
 public:
  FlexfecReceiver(uint32_t ssrc,
                  uint32_t protected_media_ssrc,
                  RecoveredPacketReceiver* recovered_packet_receiver);
  FlexfecReceiver(Clock* clock,
                  uint32_t ssrc,
                  uint32_t protected_media_ssrc,
                  RecoveredPacketReceiver* recovered_packet_receiver);
  ~FlexfecReceiver();

  // Inserts a received packet (can be either media or FlexFEC) into the
  // internal buffer, and sends the received packets to the erasure code.
  // All newly recovered packets are sent back through the callback.
  void OnRtpPacket(const RtpPacketReceived& packet);

  // Returns a counter describing the added and recovered packets.
  FecPacketCounter GetPacketCounter() const;

  // Protected to aid testing.
 protected:
  std::unique_ptr<ForwardErrorCorrection::ReceivedPacket> AddReceivedPacket(
      const RtpPacketReceived& packet);
  void ProcessReceivedPacket(
      const ForwardErrorCorrection::ReceivedPacket& received_packet);

 private:
  // Config.
  const uint32_t ssrc_;
  const uint32_t protected_media_ssrc_;

  // Erasure code interfacing and callback.
  std::unique_ptr<ForwardErrorCorrection> erasure_code_
      RTC_GUARDED_BY(sequence_checker_);
  ForwardErrorCorrection::RecoveredPacketList recovered_packets_
      RTC_GUARDED_BY(sequence_checker_);
  RecoveredPacketReceiver* const recovered_packet_receiver_;

  // Logging and stats.
  Clock* const clock_;
  Timestamp last_recovered_packet_ RTC_GUARDED_BY(sequence_checker_) =
      Timestamp::MinusInfinity();
  FecPacketCounter packet_counter_ RTC_GUARDED_BY(sequence_checker_);

  RTC_NO_UNIQUE_ADDRESS SequenceChecker sequence_checker_;
};

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_INCLUDE_FLEXFEC_RECEIVER_H_
