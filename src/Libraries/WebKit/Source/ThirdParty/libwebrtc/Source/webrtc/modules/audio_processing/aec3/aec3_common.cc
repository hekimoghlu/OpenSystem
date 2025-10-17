/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#include "modules/audio_processing/aec3/aec3_common.h"

#include <stdint.h>

#include "rtc_base/checks.h"
#include "rtc_base/system/arch.h"
#include "system_wrappers/include/cpu_features_wrapper.h"

namespace webrtc {

Aec3Optimization DetectOptimization() {
#if defined(WEBRTC_ARCH_X86_FAMILY)
  if (GetCPUInfo(kAVX2) != 0) {
    return Aec3Optimization::kAvx2;
  } else if (GetCPUInfo(kSSE2) != 0) {
    return Aec3Optimization::kSse2;
  }
#endif

#if defined(WEBRTC_HAS_NEON)
  return Aec3Optimization::kNeon;
#else
  return Aec3Optimization::kNone;
#endif
}

float FastApproxLog2f(const float in) {
  RTC_DCHECK_GT(in, .0f);
  // Read and interpret float as uint32_t and then cast to float.
  // This is done to extract the exponent (bits 30 - 23).
  // "Right shift" of the exponent is then performed by multiplying
  // with the constant (1/2^23). Finally, we subtract a constant to
  // remove the bias (https://en.wikipedia.org/wiki/Exponent_bias).
  union {
    float dummy;
    uint32_t a;
  } x = {in};
  float out = x.a;
  out *= 1.1920929e-7f;  // 1/2^23
  out -= 126.942695f;    // Remove bias.
  return out;
}

float Log2TodB(const float in_log2) {
  return 3.0102999566398121 * in_log2;
}

}  // namespace webrtc
