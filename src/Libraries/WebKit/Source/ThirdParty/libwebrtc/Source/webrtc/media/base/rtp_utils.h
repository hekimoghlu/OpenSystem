/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#ifndef MEDIA_BASE_RTP_UTILS_H_
#define MEDIA_BASE_RTP_UTILS_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "api/array_view.h"
#include "rtc_base/byte_order.h"
#include "rtc_base/system/rtc_export.h"

namespace rtc {
struct PacketTimeUpdateParams;
}  // namespace rtc

namespace cricket {

const size_t kMinRtpPacketLen = 12;
const size_t kMaxRtpPacketLen = 2048;
const size_t kMinRtcpPacketLen = 4;

enum RtcpTypes {
  kRtcpTypeSR = 200,     // Sender report payload type.
  kRtcpTypeRR = 201,     // Receiver report payload type.
  kRtcpTypeSDES = 202,   // SDES payload type.
  kRtcpTypeBye = 203,    // BYE payload type.
  kRtcpTypeApp = 204,    // APP payload type.
  kRtcpTypeRTPFB = 205,  // Transport layer Feedback message payload type.
  kRtcpTypePSFB = 206,   // Payload-specific Feedback message payload type.
};

enum class RtpPacketType {
  kRtp,
  kRtcp,
  kUnknown,
};

bool GetRtcpType(const void* data, size_t len, int* value);
bool GetRtcpSsrc(const void* data, size_t len, uint32_t* value);

// Checks the packet header to determine if it can be an RTP or RTCP packet.
RtpPacketType InferRtpPacketType(rtc::ArrayView<const uint8_t> packet);
// True if |payload type| is 0-127.
bool IsValidRtpPayloadType(int payload_type);

// True if `size` is appropriate for the indicated packet type.
bool IsValidRtpPacketSize(RtpPacketType packet_type, size_t size);

// Returns "RTCP", "RTP" or "Unknown" according to `packet_type`.
absl::string_view RtpPacketTypeToString(RtpPacketType packet_type);

// Verifies that a packet has a valid RTP header.
bool RTC_EXPORT ValidateRtpHeader(const uint8_t* rtp,
                                  size_t length,
                                  size_t* header_length);

// Helper method which updates the absolute send time extension if present.
bool UpdateRtpAbsSendTimeExtension(uint8_t* rtp,
                                   size_t length,
                                   int extension_id,
                                   uint64_t time_us);

// Applies specified `options` to the packet. It updates the absolute send time
// extension header if it is present present then updates HMAC.
bool RTC_EXPORT
ApplyPacketOptions(uint8_t* data,
                   size_t length,
                   const rtc::PacketTimeUpdateParams& packet_time_params,
                   uint64_t time_us);

}  // namespace cricket

#endif  // MEDIA_BASE_RTP_UTILS_H_
