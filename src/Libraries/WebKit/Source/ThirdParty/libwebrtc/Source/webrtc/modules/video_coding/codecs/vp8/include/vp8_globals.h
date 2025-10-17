/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
// This file contains codec dependent definitions that are needed in
// order to compile the WebRTC codebase, even if this codec is not used.

#ifndef MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_VP8_GLOBALS_H_
#define MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_VP8_GLOBALS_H_

#include "modules/video_coding/codecs/interface/common_constants.h"

namespace webrtc {

struct RTPVideoHeaderVP8 {
  void InitRTPVideoHeaderVP8() {
    nonReference = false;
    pictureId = kNoPictureId;
    tl0PicIdx = kNoTl0PicIdx;
    temporalIdx = kNoTemporalIdx;
    layerSync = false;
    keyIdx = kNoKeyIdx;
    partitionId = 0;
    beginningOfPartition = false;
  }

  friend bool operator==(const RTPVideoHeaderVP8& lhs,
                         const RTPVideoHeaderVP8& rhs) {
    return lhs.nonReference == rhs.nonReference &&
           lhs.pictureId == rhs.pictureId && lhs.tl0PicIdx == rhs.tl0PicIdx &&
           lhs.temporalIdx == rhs.temporalIdx &&
           lhs.layerSync == rhs.layerSync && lhs.keyIdx == rhs.keyIdx &&
           lhs.partitionId == rhs.partitionId &&
           lhs.beginningOfPartition == rhs.beginningOfPartition;
  }

  friend bool operator!=(const RTPVideoHeaderVP8& lhs,
                         const RTPVideoHeaderVP8& rhs) {
    return !(lhs == rhs);
  }

  bool nonReference;          // Frame is discardable.
  int16_t pictureId;          // Picture ID index, 15 bits;
                              // kNoPictureId if PictureID does not exist.
  int16_t tl0PicIdx;          // TL0PIC_IDX, 8 bits;
                              // kNoTl0PicIdx means no value provided.
  uint8_t temporalIdx;        // Temporal layer index, or kNoTemporalIdx.
  bool layerSync;             // This frame is a layer sync frame.
                              // Disabled if temporalIdx == kNoTemporalIdx.
  int keyIdx;                 // 5 bits; kNoKeyIdx means not used.
  int partitionId;            // VP8 partition ID
  bool beginningOfPartition;  // True if this packet is the first
                              // in a VP8 partition. Otherwise false
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_CODECS_VP8_INCLUDE_VP8_GLOBALS_H_
