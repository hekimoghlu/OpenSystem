/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
#ifndef RTC_BASE_NETWORK_RECEIVED_PACKET_H_
#define RTC_BASE_NETWORK_RECEIVED_PACKET_H_

#include <cstdint>
#include <optional>

#include "api/array_view.h"
#include "api/units/timestamp.h"
#include "rtc_base/network/ecn_marking.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {

// ReceivedPacket repressent a received IP packet.
// It contains a payload and metadata.
// ReceivedPacket itself does not put constraints on what payload contains. For
// example it may contains STUN, SCTP, SRTP, RTP, RTCP.... etc.
class RTC_EXPORT ReceivedPacket {
 public:
  enum DecryptionInfo {
    kNotDecrypted,   // Payload has not yet been decrypted or encryption is not
                     // used.
    kDtlsDecrypted,  // Payload has been Dtls decrypted
    kSrtpEncrypted   // Payload is SRTP encrypted.
  };

  // Caller must keep memory pointed to by payload and address valid for the
  // lifetime of this ReceivedPacket.
  ReceivedPacket(rtc::ArrayView<const uint8_t> payload,
                 const SocketAddress& source_address,
                 std::optional<webrtc::Timestamp> arrival_time = std::nullopt,
                 EcnMarking ecn = EcnMarking::kNotEct,
                 DecryptionInfo decryption = kNotDecrypted);

  ReceivedPacket CopyAndSet(DecryptionInfo decryption_info) const;

  // Address/port of the packet sender.
  const SocketAddress& source_address() const { return source_address_; }
  rtc::ArrayView<const uint8_t> payload() const { return payload_; }

  // Timestamp when this packet was received. Not available on all socket
  // implementations.
  std::optional<webrtc::Timestamp> arrival_time() const {
    return arrival_time_;
  }

  // L4S Explicit Congestion Notification.
  EcnMarking ecn() const { return ecn_; }

  const DecryptionInfo& decryption_info() const { return decryption_info_; }

  static ReceivedPacket CreateFromLegacy(
      const char* data,
      size_t size,
      int64_t packet_time_us,
      const rtc::SocketAddress& addr = rtc::SocketAddress()) {
    return CreateFromLegacy(reinterpret_cast<const uint8_t*>(data), size,
                            packet_time_us, addr);
  }

  static ReceivedPacket CreateFromLegacy(
      const uint8_t* data,
      size_t size,
      int64_t packet_time_us,
      const rtc::SocketAddress& = rtc::SocketAddress());

 private:
  rtc::ArrayView<const uint8_t> payload_;
  std::optional<webrtc::Timestamp> arrival_time_;
  const SocketAddress& source_address_;
  EcnMarking ecn_;
  DecryptionInfo decryption_info_;
};

}  // namespace rtc
#endif  // RTC_BASE_NETWORK_RECEIVED_PACKET_H_
