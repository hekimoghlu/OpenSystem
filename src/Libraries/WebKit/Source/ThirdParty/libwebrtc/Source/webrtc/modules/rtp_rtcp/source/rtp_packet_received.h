/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_PACKET_RECEIVED_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_PACKET_RECEIVED_H_

#include <stdint.h>

#include <utility>

#include "api/array_view.h"
#include "api/ref_counted_base.h"
#include "api/rtp_headers.h"
#include "api/scoped_refptr.h"
#include "api/units/timestamp.h"
#include "modules/rtp_rtcp/source/rtp_packet.h"
#include "rtc_base/network/ecn_marking.h"

namespace webrtc {
// Class to hold rtp packet with metadata for receiver side.
// The metadata is not parsed from the rtp packet, but may be derived from the
// data that is parsed from the rtp packet.
class RtpPacketReceived : public RtpPacket {
 public:
  RtpPacketReceived();
  explicit RtpPacketReceived(
      const ExtensionManager* extensions,
      webrtc::Timestamp arrival_time = webrtc::Timestamp::MinusInfinity());
  RtpPacketReceived(const RtpPacketReceived& packet);
  RtpPacketReceived(RtpPacketReceived&& packet);

  RtpPacketReceived& operator=(const RtpPacketReceived& packet);
  RtpPacketReceived& operator=(RtpPacketReceived&& packet);

  ~RtpPacketReceived();

  // TODO(bugs.webrtc.org/15054): Remove this function when all code is updated
  // to use RtpPacket directly.
  void GetHeader(RTPHeader* header) const;

  // Time in local time base as close as it can to packet arrived on the
  // network.
  webrtc::Timestamp arrival_time() const { return arrival_time_; }
  void set_arrival_time(webrtc::Timestamp time) { arrival_time_ = time; }

  // Explicit Congestion Notification (ECN), RFC-3168, Section 5.
  // Used by L4S: https://www.rfc-editor.org/rfc/rfc9331.html
  rtc::EcnMarking ecn() const { return ecn_; }
  void set_ecn(rtc::EcnMarking ecn) { ecn_ = ecn; }

  // Flag if packet was recovered via RTX or FEC.
  bool recovered() const { return recovered_; }
  void set_recovered(bool value) { recovered_ = value; }

  int payload_type_frequency() const { return payload_type_frequency_; }
  void set_payload_type_frequency(int value) {
    payload_type_frequency_ = value;
  }

  // An application can attach arbitrary data to an RTP packet using
  // `additional_data`. The additional data does not affect WebRTC processing.
  rtc::scoped_refptr<rtc::RefCountedBase> additional_data() const {
    return additional_data_;
  }
  void set_additional_data(rtc::scoped_refptr<rtc::RefCountedBase> data) {
    additional_data_ = std::move(data);
  }

 private:
  webrtc::Timestamp arrival_time_ = Timestamp::MinusInfinity();
  rtc::EcnMarking ecn_ = rtc::EcnMarking::kNotEct;
  int payload_type_frequency_ = 0;
  bool recovered_ = false;
  rtc::scoped_refptr<rtc::RefCountedBase> additional_data_;
};

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_SOURCE_RTP_PACKET_RECEIVED_H_
