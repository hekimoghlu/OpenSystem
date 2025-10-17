/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#ifndef AUDIO_CONVERSION_H_
#define AUDIO_CONVERSION_H_

#include <stddef.h>
#include <stdint.h>

namespace webrtc {

// Convert fixed point number with 8 bit fractional part, to floating point.
inline float Q8ToFloat(uint32_t v) {
  return static_cast<float>(v) / (1 << 8);
}

// Convert fixed point number with 14 bit fractional part, to floating point.
inline float Q14ToFloat(uint32_t v) {
  return static_cast<float>(v) / (1 << 14);
}
}  // namespace webrtc

#endif  // AUDIO_CONVERSION_H_
