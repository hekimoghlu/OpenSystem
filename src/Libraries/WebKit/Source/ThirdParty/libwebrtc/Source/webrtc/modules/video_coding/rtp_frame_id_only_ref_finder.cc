/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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
#include "modules/video_coding/rtp_frame_id_only_ref_finder.h"

#include <utility>

#include "rtc_base/logging.h"

namespace webrtc {

RtpFrameReferenceFinder::ReturnVector RtpFrameIdOnlyRefFinder::ManageFrame(
    std::unique_ptr<RtpFrameObject> frame,
    int frame_id) {
  frame->SetSpatialIndex(0);
  frame->SetId(unwrapper_.Unwrap(frame_id & (kFrameIdLength - 1)));
  frame->num_references =
      frame->frame_type() == VideoFrameType::kVideoFrameKey ? 0 : 1;
  frame->references[0] = frame->Id() - 1;

  RtpFrameReferenceFinder::ReturnVector res;
  res.push_back(std::move(frame));
  return res;
}

}  // namespace webrtc
