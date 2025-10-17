/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "modules/video_coding/rtp_generic_ref_finder.h"

#include <utility>

#include "api/video/video_codec_constants.h"
#include "rtc_base/logging.h"

namespace webrtc {

RtpFrameReferenceFinder::ReturnVector RtpGenericFrameRefFinder::ManageFrame(
    std::unique_ptr<RtpFrameObject> frame,
    const RTPVideoHeader::GenericDescriptorInfo& descriptor) {
  RtpFrameReferenceFinder::ReturnVector res;
  if (descriptor.spatial_index >= kMaxSpatialLayers) {
    RTC_LOG(LS_WARNING) << "Spatial index " << descriptor.spatial_index
                        << " is unsupported.";
    return res;
  }

  // Frame IDs are unwrapped in the RtpVideoStreamReceiver, no need to unwrap
  // them here.
  frame->SetId(descriptor.frame_id);
  frame->SetSpatialIndex(descriptor.spatial_index);
  if (descriptor.temporal_index != kNoTemporalIdx)
    frame->SetTemporalIndex(descriptor.temporal_index);

  if (EncodedFrame::kMaxFrameReferences < descriptor.dependencies.size()) {
    RTC_LOG(LS_WARNING) << "Too many dependencies in generic descriptor.";
    return res;
  }

  frame->num_references = descriptor.dependencies.size();
  for (size_t i = 0; i < descriptor.dependencies.size(); ++i) {
    frame->references[i] = descriptor.dependencies[i];
  }

  res.push_back(std::move(frame));
  return res;
}

}  // namespace webrtc
