/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#ifndef CALL_RTX_RECEIVE_STREAM_H_
#define CALL_RTX_RECEIVE_STREAM_H_

#include <cstdint>
#include <map>

#include "api/sequence_checker.h"
#include "call/rtp_packet_sink_interface.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread_annotations.h"

namespace webrtc {

class ReceiveStatistics;

// This class is responsible for RTX decapsulation. The resulting media packets
// are passed on to a sink representing the associated media stream.
class RtxReceiveStream : public RtpPacketSinkInterface {
 public:
  RtxReceiveStream(RtpPacketSinkInterface* media_sink,
                   std::map<int, int> associated_payload_types,
                   uint32_t media_ssrc,
                   // TODO(nisse): Delete this argument, and
                   // corresponding member variable, by moving the
                   // responsibility for rtcp feedback to
                   // RtpStreamReceiverController.
                   ReceiveStatistics* rtp_receive_statistics = nullptr);
  ~RtxReceiveStream() override;

  // Update payload types post construction. Must be called from the same
  // calling context as `OnRtpPacket` is called on.
  void SetAssociatedPayloadTypes(std::map<int, int> associated_payload_types);

  // RtpPacketSinkInterface.
  void OnRtpPacket(const RtpPacketReceived& packet) override;

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker packet_checker_;
  RtpPacketSinkInterface* const media_sink_;
  // Map from rtx payload type -> media payload type.
  std::map<int, int> associated_payload_types_ RTC_GUARDED_BY(&packet_checker_);
  // TODO(nisse): Ultimately, the media receive stream shouldn't care about the
  // ssrc, and we should delete this.
  const uint32_t media_ssrc_;
  ReceiveStatistics* const rtp_receive_statistics_;
};

}  // namespace webrtc

#endif  // CALL_RTX_RECEIVE_STREAM_H_
