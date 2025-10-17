/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#ifndef MODULES_RTP_RTCP_INCLUDE_RTP_CVO_H_
#define MODULES_RTP_RTCP_INCLUDE_RTP_CVO_H_

#include "api/video/video_rotation.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Please refer to http://www.etsi.org/deliver/etsi_ts/126100_126199/126114/
// 12.07.00_60/ts_126114v120700p.pdf Section 7.4.5. The rotation of a frame is
// the clockwise angle the frames must be rotated in order to display the frames
// correctly if the display is rotated in its natural orientation.
inline uint8_t ConvertVideoRotationToCVOByte(VideoRotation rotation) {
  switch (rotation) {
    case kVideoRotation_0:
      return 0;
    case kVideoRotation_90:
      return 1;
    case kVideoRotation_180:
      return 2;
    case kVideoRotation_270:
      return 3;
  }
  RTC_DCHECK_NOTREACHED();
  return 0;
}

inline VideoRotation ConvertCVOByteToVideoRotation(uint8_t cvo_byte) {
  // CVO byte: |0 0 0 0 C F R R|.
  const uint8_t rotation_bits = cvo_byte & 0x3;
  switch (rotation_bits) {
    case 0:
      return kVideoRotation_0;
    case 1:
      return kVideoRotation_90;
    case 2:
      return kVideoRotation_180;
    case 3:
      return kVideoRotation_270;
    default:
      RTC_DCHECK_NOTREACHED();
      return kVideoRotation_0;
  }
}

}  // namespace webrtc
#endif  // MODULES_RTP_RTCP_INCLUDE_RTP_CVO_H_
