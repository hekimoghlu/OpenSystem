/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#ifndef MODULES_VIDEO_CODING_RTP_FRAME_ID_ONLY_REF_FINDER_H_
#define MODULES_VIDEO_CODING_RTP_FRAME_ID_ONLY_REF_FINDER_H_

#include <memory>

#include "absl/container/inlined_vector.h"
#include "modules/rtp_rtcp/source/frame_object.h"
#include "modules/video_coding/rtp_frame_reference_finder.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {

class RtpFrameIdOnlyRefFinder {
 public:
  RtpFrameIdOnlyRefFinder() = default;

  RtpFrameReferenceFinder::ReturnVector ManageFrame(
      std::unique_ptr<RtpFrameObject> frame,
      int frame_id);

 private:
  static constexpr int kFrameIdLength = 1 << 15;
  SeqNumUnwrapper<uint16_t, kFrameIdLength> unwrapper_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_RTP_FRAME_ID_ONLY_REF_FINDER_H_
