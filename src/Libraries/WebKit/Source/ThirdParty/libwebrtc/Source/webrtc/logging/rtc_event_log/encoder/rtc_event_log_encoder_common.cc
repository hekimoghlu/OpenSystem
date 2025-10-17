/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "logging/rtc_event_log/encoder/rtc_event_log_encoder_common.h"

#include <cstdint>

#include "rtc_base/checks.h"

namespace webrtc {
namespace {
// We use 0x3fff because that gives decent precision (compared to the underlying
// measurement producing the packet loss fraction) on the one hand, while
// allowing us to use no more than 2 bytes in varint form on the other hand.
// (We might also fixed-size encode using at most 14 bits.)
constexpr uint32_t kPacketLossFractionRange = (1 << 14) - 1;  // 0x3fff
constexpr float kPacketLossFractionRangeFloat =
    static_cast<float>(kPacketLossFractionRange);
}  // namespace

uint32_t ConvertPacketLossFractionToProtoFormat(float packet_loss_fraction) {
  RTC_DCHECK_GE(packet_loss_fraction, 0);
  RTC_DCHECK_LE(packet_loss_fraction, 1);
  return static_cast<uint32_t>(packet_loss_fraction * kPacketLossFractionRange);
}

bool ParsePacketLossFractionFromProtoFormat(uint32_t proto_packet_loss_fraction,
                                            float* output) {
  if (proto_packet_loss_fraction >= kPacketLossFractionRange) {
    return false;
  }
  *output = proto_packet_loss_fraction / kPacketLossFractionRangeFloat;
  return true;
}
}  // namespace webrtc
