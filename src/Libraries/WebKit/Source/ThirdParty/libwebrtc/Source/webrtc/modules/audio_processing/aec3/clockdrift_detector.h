/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_CLOCKDRIFT_DETECTOR_H_
#define MODULES_AUDIO_PROCESSING_AEC3_CLOCKDRIFT_DETECTOR_H_

#if defined(WEBRTC_WEBKIT_BUILD)
#include <cstdlib>
#endif

#include <stddef.h>

#include <array>

namespace webrtc {

class ApmDataDumper;
struct DownsampledRenderBuffer;
struct EchoCanceller3Config;

// Detects clockdrift by analyzing the estimated delay.
class ClockdriftDetector {
 public:
  enum class Level { kNone, kProbable, kVerified, kNumCategories };
  ClockdriftDetector();
  ~ClockdriftDetector();
  void Update(int delay_estimate);
  Level ClockdriftLevel() const { return level_; }

 private:
  std::array<int, 3> delay_history_;
  Level level_;
  size_t stability_counter_;
};
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_CLOCKDRIFT_DETECTOR_H_
