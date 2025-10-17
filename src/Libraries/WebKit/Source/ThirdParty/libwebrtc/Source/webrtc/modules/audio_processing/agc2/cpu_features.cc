/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#include "modules/audio_processing/agc2/cpu_features.h"

#include "rtc_base/strings/string_builder.h"
#include "rtc_base/system/arch.h"
#include "system_wrappers/include/cpu_features_wrapper.h"

namespace webrtc {

std::string AvailableCpuFeatures::ToString() const {
  char buf[64];
  rtc::SimpleStringBuilder builder(buf);
  bool first = true;
  if (sse2) {
    builder << (first ? "SSE2" : "_SSE2");
    first = false;
  }
  if (avx2) {
    builder << (first ? "AVX2" : "_AVX2");
    first = false;
  }
  if (neon) {
    builder << (first ? "NEON" : "_NEON");
    first = false;
  }
  if (first) {
    return "none";
  }
  return builder.str();
}

// Detects available CPU features.
AvailableCpuFeatures GetAvailableCpuFeatures() {
#if defined(WEBRTC_ARCH_X86_FAMILY)
  return {/*sse2=*/GetCPUInfo(kSSE2) != 0,
          /*avx2=*/GetCPUInfo(kAVX2) != 0,
          /*neon=*/false};
#elif defined(WEBRTC_HAS_NEON)
  return {/*sse2=*/false,
          /*avx2=*/false,
          /*neon=*/true};
#else
  return {/*sse2=*/false,
          /*avx2=*/false,
          /*neon=*/false};
#endif
}

AvailableCpuFeatures NoAvailableCpuFeatures() {
  return {/*sse2=*/false, /*avx2=*/false, /*neon=*/false};
}

}  // namespace webrtc
