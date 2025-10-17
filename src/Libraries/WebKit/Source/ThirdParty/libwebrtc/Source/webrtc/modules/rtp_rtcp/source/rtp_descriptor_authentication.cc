/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
#include "modules/rtp_rtcp/source/rtp_descriptor_authentication.h"

#include <cstdint>
#include <vector>

#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor.h"
#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor_extension.h"
#include "modules/rtp_rtcp/source/rtp_video_header.h"

namespace webrtc {

std::vector<uint8_t> RtpDescriptorAuthentication(
    const RTPVideoHeader& rtp_video_header) {
  if (!rtp_video_header.generic) {
    return {};
  }
  const RTPVideoHeader::GenericDescriptorInfo& descriptor =
      *rtp_video_header.generic;
  // Default way of creating additional data for an encrypted frame.
  if (descriptor.spatial_index < 0 || descriptor.temporal_index < 0 ||
      descriptor.spatial_index >=
          RtpGenericFrameDescriptor::kMaxSpatialLayers ||
      descriptor.temporal_index >=
          RtpGenericFrameDescriptor::kMaxTemporalLayers ||
      descriptor.dependencies.size() >
          RtpGenericFrameDescriptor::kMaxNumFrameDependencies) {
    return {};
  }
  RtpGenericFrameDescriptor frame_descriptor;
  frame_descriptor.SetFirstPacketInSubFrame(true);
  frame_descriptor.SetLastPacketInSubFrame(false);
  frame_descriptor.SetTemporalLayer(descriptor.temporal_index);
  frame_descriptor.SetSpatialLayersBitmask(1 << descriptor.spatial_index);
  frame_descriptor.SetFrameId(descriptor.frame_id & 0xFFFF);
  for (int64_t dependency : descriptor.dependencies) {
    frame_descriptor.AddFrameDependencyDiff(descriptor.frame_id - dependency);
  }
  if (descriptor.dependencies.empty()) {
    frame_descriptor.SetResolution(rtp_video_header.width,
                                   rtp_video_header.height);
  }
  std::vector<uint8_t> result(
      RtpGenericFrameDescriptorExtension00::ValueSize(frame_descriptor));
  RtpGenericFrameDescriptorExtension00::Write(result, frame_descriptor);
  return result;
}

}  // namespace webrtc
