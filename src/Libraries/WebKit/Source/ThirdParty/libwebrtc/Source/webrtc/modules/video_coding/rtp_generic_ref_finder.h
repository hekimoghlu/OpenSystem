/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#ifndef MODULES_VIDEO_CODING_RTP_GENERIC_REF_FINDER_H_
#define MODULES_VIDEO_CODING_RTP_GENERIC_REF_FINDER_H_

#include <memory>

#include "modules/rtp_rtcp/source/frame_object.h"
#include "modules/video_coding/rtp_frame_reference_finder.h"

namespace webrtc {

class RtpGenericFrameRefFinder {
 public:
  RtpGenericFrameRefFinder() = default;

  RtpFrameReferenceFinder::ReturnVector ManageFrame(
      std::unique_ptr<RtpFrameObject> frame,
      const RTPVideoHeader::GenericDescriptorInfo& descriptor);
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_RTP_GENERIC_REF_FINDER_H_
