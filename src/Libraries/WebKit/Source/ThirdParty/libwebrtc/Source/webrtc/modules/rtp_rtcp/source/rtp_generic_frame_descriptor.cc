/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor.h"

#include <cstdint>

#include "rtc_base/checks.h"

namespace webrtc {

RtpGenericFrameDescriptor::RtpGenericFrameDescriptor() = default;
RtpGenericFrameDescriptor::RtpGenericFrameDescriptor(
    const RtpGenericFrameDescriptor&) = default;
RtpGenericFrameDescriptor::~RtpGenericFrameDescriptor() = default;

int RtpGenericFrameDescriptor::TemporalLayer() const {
  RTC_DCHECK(FirstPacketInSubFrame());
  return temporal_layer_;
}

void RtpGenericFrameDescriptor::SetTemporalLayer(int temporal_layer) {
  RTC_DCHECK_GE(temporal_layer, 0);
  RTC_DCHECK_LT(temporal_layer, kMaxTemporalLayers);
  temporal_layer_ = temporal_layer;
}

int RtpGenericFrameDescriptor::SpatialLayer() const {
  RTC_DCHECK(FirstPacketInSubFrame());
  int layer = 0;
  uint8_t spatial_layers = spatial_layers_;
  while (spatial_layers_ != 0 && !(spatial_layers & 1)) {
    spatial_layers >>= 1;
    layer++;
  }
  return layer;
}

uint8_t RtpGenericFrameDescriptor::SpatialLayersBitmask() const {
  RTC_DCHECK(FirstPacketInSubFrame());
  return spatial_layers_;
}

void RtpGenericFrameDescriptor::SetSpatialLayersBitmask(
    uint8_t spatial_layers) {
  RTC_DCHECK(FirstPacketInSubFrame());
  spatial_layers_ = spatial_layers;
}

void RtpGenericFrameDescriptor::SetResolution(int width, int height) {
  RTC_DCHECK(FirstPacketInSubFrame());
  RTC_DCHECK_GE(width, 0);
  RTC_DCHECK_LE(width, 0xFFFF);
  RTC_DCHECK_GE(height, 0);
  RTC_DCHECK_LE(height, 0xFFFF);
  width_ = width;
  height_ = height;
}

uint16_t RtpGenericFrameDescriptor::FrameId() const {
  RTC_DCHECK(FirstPacketInSubFrame());
  return frame_id_;
}

void RtpGenericFrameDescriptor::SetFrameId(uint16_t frame_id) {
  RTC_DCHECK(FirstPacketInSubFrame());
  frame_id_ = frame_id;
}

rtc::ArrayView<const uint16_t>
RtpGenericFrameDescriptor::FrameDependenciesDiffs() const {
  RTC_DCHECK(FirstPacketInSubFrame());
  return rtc::MakeArrayView(frame_deps_id_diffs_, num_frame_deps_);
}

bool RtpGenericFrameDescriptor::AddFrameDependencyDiff(uint16_t fdiff) {
  RTC_DCHECK(FirstPacketInSubFrame());
  if (num_frame_deps_ == kMaxNumFrameDependencies)
    return false;
  if (fdiff == 0)
    return false;
  RTC_DCHECK_LT(fdiff, 1 << 14);
  RTC_DCHECK_GT(fdiff, 0);
  frame_deps_id_diffs_[num_frame_deps_] = fdiff;
  num_frame_deps_++;
  return true;
}

}  // namespace webrtc
