/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
#include "rtc_base/crc32.h"

#include "rtc_base/arraysize.h"

namespace rtc {

// This implementation is based on the sample implementation in RFC 1952.

// CRC32 polynomial, in reversed form.
// See RFC 1952, or http://en.wikipedia.org/wiki/Cyclic_redundancy_check
static const uint32_t kCrc32Polynomial = 0xEDB88320;

static uint32_t* LoadCrc32Table() {
  static uint32_t kCrc32Table[256];
  for (uint32_t i = 0; i < arraysize(kCrc32Table); ++i) {
    uint32_t c = i;
    for (size_t j = 0; j < 8; ++j) {
      if (c & 1) {
        c = kCrc32Polynomial ^ (c >> 1);
      } else {
        c >>= 1;
      }
    }
    kCrc32Table[i] = c;
  }
  return kCrc32Table;
}

uint32_t UpdateCrc32(uint32_t start, const void* buf, size_t len) {
  static uint32_t* kCrc32Table = LoadCrc32Table();

  uint32_t c = start ^ 0xFFFFFFFF;
  const uint8_t* u = static_cast<const uint8_t*>(buf);
  for (size_t i = 0; i < len; ++i) {
    c = kCrc32Table[(c ^ u[i]) & 0xFF] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFF;
}

}  // namespace rtc
