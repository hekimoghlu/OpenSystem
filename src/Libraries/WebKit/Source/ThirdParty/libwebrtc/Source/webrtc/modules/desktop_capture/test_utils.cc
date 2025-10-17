/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#include "modules/desktop_capture/test_utils.h"

#include <stdint.h>
#include <string.h>

#include "modules/desktop_capture/desktop_geometry.h"
#include "rtc_base/checks.h"

namespace webrtc {

void ClearDesktopFrame(DesktopFrame* frame) {
  RTC_DCHECK(frame);
  uint8_t* data = frame->data();
  for (int i = 0; i < frame->size().height(); i++) {
    memset(data, 0, frame->size().width() * DesktopFrame::kBytesPerPixel);
    data += frame->stride();
  }
}

bool DesktopFrameDataEquals(const DesktopFrame& left,
                            const DesktopFrame& right) {
  if (!left.size().equals(right.size())) {
    return false;
  }

  const uint8_t* left_array = left.data();
  const uint8_t* right_array = right.data();
  for (int i = 0; i < left.size().height(); i++) {
    if (memcmp(left_array, right_array,
               DesktopFrame::kBytesPerPixel * left.size().width()) != 0) {
      return false;
    }
    left_array += left.stride();
    right_array += right.stride();
  }

  return true;
}

}  // namespace webrtc
