/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_L2T2_KEY_SHIFT_H_
#define MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_L2T2_KEY_SHIFT_H_

#include <vector>

#include "api/transport/rtp/dependency_descriptor.h"
#include "api/video/video_bitrate_allocation.h"
#include "common_video/generic_frame_descriptor/generic_frame_info.h"
#include "modules/video_coding/svc/scalable_video_controller.h"

namespace webrtc {

// S1T1     0   0
//         /   /   /
// S1T0   0---0---0
//        |        ...
// S0T1   |   0   0
//        |  /   /
// S0T0   0-0---0--
// Time-> 0 1 2 3 4
class ScalabilityStructureL2T2KeyShift : public ScalableVideoController {
 public:
  ~ScalabilityStructureL2T2KeyShift() override;

  StreamLayersConfig StreamConfig() const override;
  FrameDependencyStructure DependencyStructure() const override;

  std::vector<LayerFrameConfig> NextFrameConfig(bool restart) override;
  GenericFrameInfo OnEncodeDone(const LayerFrameConfig& config) override;
  void OnRatesUpdated(const VideoBitrateAllocation& bitrates) override;

 private:
  enum FramePattern {
    kKey,
    kDelta0,
    kDelta1,
  };

  static constexpr int kNumSpatialLayers = 2;
  static constexpr int kNumTemporalLayers = 2;

  bool DecodeTargetIsActive(int sid, int tid) const {
    return active_decode_targets_[sid * kNumTemporalLayers + tid];
  }
  void SetDecodeTargetIsActive(int sid, int tid, bool value) {
    active_decode_targets_.set(sid * kNumTemporalLayers + tid, value);
  }

  FramePattern next_pattern_ = kKey;
  std::bitset<32> active_decode_targets_ = 0b1111;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_SVC_SCALABILITY_STRUCTURE_L2T2_KEY_SHIFT_H_
