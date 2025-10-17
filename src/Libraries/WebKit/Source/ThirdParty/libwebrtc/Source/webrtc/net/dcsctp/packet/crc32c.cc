/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#include "net/dcsctp/packet/crc32c.h"

#include <cstdint>

#include "third_party/crc32c/src/include/crc32c/crc32c.h"

namespace dcsctp {

uint32_t GenerateCrc32C(rtc::ArrayView<const uint8_t> data) {
  uint32_t crc32c = crc32c_value(data.data(), data.size());

  // Byte swapping for little endian byte order:
  uint8_t byte0 = crc32c;
  uint8_t byte1 = crc32c >> 8;
  uint8_t byte2 = crc32c >> 16;
  uint8_t byte3 = crc32c >> 24;
  crc32c = ((byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3);
  return crc32c;
}
}  // namespace dcsctp
