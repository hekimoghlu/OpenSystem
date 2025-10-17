/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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
#include "modules/video_coding/svc/scalable_video_controller_no_layering.h"

#include <utility>
#include <vector>

#include "api/transport/rtp/dependency_descriptor.h"
#include "rtc_base/checks.h"

namespace webrtc {

ScalableVideoControllerNoLayering::~ScalableVideoControllerNoLayering() =
    default;

ScalableVideoController::StreamLayersConfig
ScalableVideoControllerNoLayering::StreamConfig() const {
  StreamLayersConfig result;
  result.num_spatial_layers = 1;
  result.num_temporal_layers = 1;
  result.uses_reference_scaling = false;
  return result;
}

FrameDependencyStructure
ScalableVideoControllerNoLayering::DependencyStructure() const {
  FrameDependencyStructure structure;
  structure.num_decode_targets = 1;
  structure.num_chains = 1;
  structure.decode_target_protected_by_chain = {0};

  FrameDependencyTemplate key_frame;
  key_frame.decode_target_indications = {DecodeTargetIndication::kSwitch};
  key_frame.chain_diffs = {0};
  structure.templates.push_back(key_frame);

  FrameDependencyTemplate delta_frame;
  delta_frame.decode_target_indications = {DecodeTargetIndication::kSwitch};
  delta_frame.chain_diffs = {1};
  delta_frame.frame_diffs = {1};
  structure.templates.push_back(delta_frame);

  return structure;
}

std::vector<ScalableVideoController::LayerFrameConfig>
ScalableVideoControllerNoLayering::NextFrameConfig(bool restart) {
  if (!enabled_) {
    return {};
  }
  std::vector<LayerFrameConfig> result(1);
  if (restart || start_) {
    result[0].Id(0).Keyframe().Update(0);
  } else {
    result[0].Id(0).ReferenceAndUpdate(0);
  }
  start_ = false;
  return result;
}

GenericFrameInfo ScalableVideoControllerNoLayering::OnEncodeDone(
    const LayerFrameConfig& config) {
  RTC_DCHECK_EQ(config.Id(), 0);
  GenericFrameInfo frame_info;
  frame_info.encoder_buffers = config.Buffers();
  if (config.IsKeyframe()) {
    for (auto& buffer : frame_info.encoder_buffers) {
      buffer.referenced = false;
    }
  }
  frame_info.decode_target_indications = {DecodeTargetIndication::kSwitch};
  frame_info.part_of_chain = {true};
  return frame_info;
}

void ScalableVideoControllerNoLayering::OnRatesUpdated(
    const VideoBitrateAllocation& bitrates) {
  enabled_ = bitrates.GetBitrate(0, 0) > 0;
}

}  // namespace webrtc
