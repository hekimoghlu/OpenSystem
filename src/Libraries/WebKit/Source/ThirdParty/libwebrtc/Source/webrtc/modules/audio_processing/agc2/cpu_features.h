/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AGC2_CPU_FEATURES_H_
#define MODULES_AUDIO_PROCESSING_AGC2_CPU_FEATURES_H_

#include <string>

namespace webrtc {

// Collection of flags indicating which CPU features are available on the
// current platform. True means available.
struct AvailableCpuFeatures {
  AvailableCpuFeatures(bool sse2, bool avx2, bool neon)
      : sse2(sse2), avx2(avx2), neon(neon) {}
  // Intel.
  bool sse2;
  bool avx2;
  // ARM.
  bool neon;
  std::string ToString() const;
};

// Detects what CPU features are available.
AvailableCpuFeatures GetAvailableCpuFeatures();

// Returns the CPU feature flags all set to false.
AvailableCpuFeatures NoAvailableCpuFeatures();

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AGC2_CPU_FEATURES_H_
