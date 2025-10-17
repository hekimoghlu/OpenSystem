/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "modules/rtp_rtcp/source/leb128.h"

#include <cstdint>

namespace webrtc {

int Leb128Size(uint64_t value) {
  int size = 0;
  while (value >= 0x80) {
    ++size;
    value >>= 7;
  }
  return size + 1;
}

uint64_t ReadLeb128(const uint8_t*& read_at, const uint8_t* end) {
  uint64_t value = 0;
  int fill_bits = 0;
  while (read_at != end && fill_bits < 64 - 7) {
    uint8_t leb128_byte = *read_at;
    value |= uint64_t{leb128_byte & 0x7Fu} << fill_bits;
    ++read_at;
    fill_bits += 7;
    if ((leb128_byte & 0x80) == 0) {
      return value;
    }
  }
  // Read 9 bytes and didn't find the terminator byte. Check if 10th byte
  // is that terminator, however to fit result into uint64_t it may carry only
  // single bit.
  if (read_at != end && *read_at <= 1) {
    value |= uint64_t{*read_at} << fill_bits;
    ++read_at;
    return value;
  }
  // Failed to find terminator leb128 byte.
  read_at = nullptr;
  return 0;
}

int WriteLeb128(uint64_t value, uint8_t* buffer) {
  int size = 0;
  while (value >= 0x80) {
    buffer[size] = 0x80 | (value & 0x7F);
    ++size;
    value >>= 7;
  }
  buffer[size] = value;
  ++size;
  return size;
}

}  // namespace webrtc
