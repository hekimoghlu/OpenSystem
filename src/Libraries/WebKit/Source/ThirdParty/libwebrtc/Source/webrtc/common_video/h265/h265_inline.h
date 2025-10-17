/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
// This header file includes the inline functions in H265 parser.

#ifndef COMMON_VIDEO_H265_H265_INLINE_H_
#define COMMON_VIDEO_H265_H265_INLINE_H_

#include <stdint.h>

#include "rtc_base/compile_assert_c.h"

extern const int8_t kWebRtcVideo_CountLeadingZeros32_Table[64];

static __inline int WebRtcVideo_CountLeadingZeros32_NotBuiltin(uint32_t n) {
  // Normalize n by rounding up to the nearest number that is a sequence of 0
  // bits followed by a sequence of 1 bits. This number has the same number of
  // leading zeros as the original n. There are exactly 33 such values.
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;

  // Multiply the modified n with a constant selected (by exhaustive search)
  // such that each of the 33 possible values of n give a product whose 6 most
  // significant bits are unique. Then look up the answer in the table.
  return kWebRtcVideo_CountLeadingZeros32_Table[(n * 0x8c0b2891) >> 26];
}

// Returns the number of leading zero bits in the argument.
static __inline int WebRtcVideo_CountLeadingZeros32(uint32_t n) {
#ifdef __GNUC__
  RTC_COMPILE_ASSERT(sizeof(unsigned int) == sizeof(uint32_t));
  return n == 0 ? 32 : __builtin_clz(n);
#else
  return WebRtcVideo_CountLeadingZeros32_NotBuiltin(n);
#endif
}
#endif  // COMMON_VIDEO_H265_H265_INLINE_H_
