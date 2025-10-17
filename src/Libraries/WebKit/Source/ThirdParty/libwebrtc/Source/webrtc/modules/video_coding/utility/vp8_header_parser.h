/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_VP8_HEADER_PARSER_H_
#define MODULES_VIDEO_CODING_UTILITY_VP8_HEADER_PARSER_H_

#include <stdint.h>
#include <stdio.h>

namespace webrtc {

namespace vp8 {

typedef struct VP8BitReader VP8BitReader;
struct VP8BitReader {
  // Boolean decoder.
  uint32_t value_;  // Current value (2 bytes).
  uint32_t range_;  // Current range (always in [128..255] interval).
  int bits_;        // Number of bits shifted out of value, at most 7.
  // Read buffer.
  const uint8_t* buf_;      // Next byte to be read.
  const uint8_t* buf_end_;  // End of read buffer.
};

// Gets the QP, QP range: [0, 127].
// Returns true on success, false otherwise.
bool GetQp(const uint8_t* buf, size_t length, int* qp);

}  // namespace vp8

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_UTILITY_VP8_HEADER_PARSER_H_
