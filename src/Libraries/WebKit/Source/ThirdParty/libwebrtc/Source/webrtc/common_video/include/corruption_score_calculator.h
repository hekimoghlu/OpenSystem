/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#ifndef COMMON_VIDEO_INCLUDE_CORRUPTION_SCORE_CALCULATOR_H_
#define COMMON_VIDEO_INCLUDE_CORRUPTION_SCORE_CALCULATOR_H_

#include <optional>

#include "api/video/video_frame.h"
#include "common_video/frame_instrumentation_data.h"

namespace webrtc {

// Allow classes to have their own implementations of how to calculate a score
// for automatic corruption detection.
class CorruptionScoreCalculator {
 public:
  virtual ~CorruptionScoreCalculator() = default;

  virtual std::optional<double> CalculateCorruptionScore(
      const VideoFrame& frame,
      const FrameInstrumentationData& frame_instrumentation_data) = 0;
};

}  // namespace webrtc

#endif  // COMMON_VIDEO_INCLUDE_CORRUPTION_SCORE_CALCULATOR_H_
