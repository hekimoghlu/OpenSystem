/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#ifndef MODULES_RTP_RTCP_SOURCE_RTP_UTIL_H_
#define MODULES_RTP_RTCP_SOURCE_RTP_UTIL_H_

#include <cstdint>

#include "api/array_view.h"

namespace webrtc {

bool IsRtcpPacket(rtc::ArrayView<const uint8_t> packet);
bool IsRtpPacket(rtc::ArrayView<const uint8_t> packet);

// Returns base rtp header fields of the rtp packet.
// Behaviour is undefined when `!IsRtpPacket(rtp_packet)`.
int ParseRtpPayloadType(rtc::ArrayView<const uint8_t> rtp_packet);
uint16_t ParseRtpSequenceNumber(rtc::ArrayView<const uint8_t> rtp_packet);
uint32_t ParseRtpSsrc(rtc::ArrayView<const uint8_t> rtp_packet);

}  // namespace webrtc

#endif  // MODULES_RTP_RTCP_SOURCE_RTP_UTIL_H_
