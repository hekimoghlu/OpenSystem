/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#ifndef MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_TEST_HELPERS_H_
#define MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_TEST_HELPERS_H_

#include <stdint.h>

#include <vector>

#include "api/array_view.h"
#include "api/transport/rtp/dependency_descriptor.h"
#include "api/video/video_bitrate_allocation.h"
#include "common_video/generic_frame_descriptor/generic_frame_info.h"
#include "modules/video_coding/chain_diff_calculator.h"
#include "modules/video_coding/frame_dependencies_calculator.h"
#include "modules/video_coding/svc/scalable_video_controller.h"

namespace webrtc {

// Creates bitrate allocation with non-zero bitrate for given number of temporal
// layers for each spatial layer.
VideoBitrateAllocation EnableTemporalLayers(int s0, int s1 = 0, int s2 = 0);

class ScalabilityStructureWrapper {
 public:
  explicit ScalabilityStructureWrapper(ScalableVideoController& structure)
      : structure_controller_(structure) {}

  std::vector<GenericFrameInfo> GenerateFrames(int num_temporal_units) {
    std::vector<GenericFrameInfo> frames;
    GenerateFrames(num_temporal_units, frames);
    return frames;
  }
  void GenerateFrames(int num_temporal_units,
                      std::vector<GenericFrameInfo>& frames);

  // Returns false and ADD_FAILUREs for frames with invalid references.
  // In particular validates no frame frame reference to frame before frames[0].
  // In error messages frames are indexed starting with 0.
  bool FrameReferencesAreValid(
      rtc::ArrayView<const GenericFrameInfo> frames) const;

 private:
  ScalableVideoController& structure_controller_;
  FrameDependenciesCalculator frame_deps_calculator_;
  ChainDiffCalculator chain_diff_calculator_;
  int64_t frame_id_ = 0;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_TEST_HELPERS_H_
