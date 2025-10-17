/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#include "modules/rtp_rtcp/source/rtp_packetizer_av1_test_helper.h"

#include <stdint.h>

#include <initializer_list>
#include <vector>

namespace webrtc {

Av1Obu::Av1Obu(uint8_t obu_type) : header_(obu_type | kAv1ObuSizePresentBit) {}

Av1Obu& Av1Obu::WithExtension(uint8_t extension) {
  extension_ = extension;
  header_ |= kAv1ObuExtensionPresentBit;
  return *this;
}
Av1Obu& Av1Obu::WithoutSize() {
  header_ &= ~kAv1ObuSizePresentBit;
  return *this;
}
Av1Obu& Av1Obu::WithPayload(std::vector<uint8_t> payload) {
  payload_ = std::move(payload);
  return *this;
}

std::vector<uint8_t> BuildAv1Frame(std::initializer_list<Av1Obu> obus) {
  std::vector<uint8_t> raw;
  for (const Av1Obu& obu : obus) {
    raw.push_back(obu.header_);
    if (obu.header_ & kAv1ObuExtensionPresentBit) {
      raw.push_back(obu.extension_);
    }
    if (obu.header_ & kAv1ObuSizePresentBit) {
      // write size in leb128 format.
      size_t payload_size = obu.payload_.size();
      while (payload_size >= 0x80) {
        raw.push_back(0x80 | (payload_size & 0x7F));
        payload_size >>= 7;
      }
      raw.push_back(payload_size);
    }
    raw.insert(raw.end(), obu.payload_.begin(), obu.payload_.end());
  }
  return raw;
}

}  // namespace webrtc
