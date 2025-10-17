/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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
#include "modules/desktop_capture/mouse_cursor.h"

#include "modules/desktop_capture/desktop_frame.h"
#include "rtc_base/checks.h"

namespace webrtc {

MouseCursor::MouseCursor() {}

MouseCursor::MouseCursor(DesktopFrame* image, const DesktopVector& hotspot)
    : image_(image), hotspot_(hotspot) {
  RTC_DCHECK(0 <= hotspot_.x() && hotspot_.x() <= image_->size().width());
  RTC_DCHECK(0 <= hotspot_.y() && hotspot_.y() <= image_->size().height());
}

MouseCursor::~MouseCursor() {}

// static
MouseCursor* MouseCursor::CopyOf(const MouseCursor& cursor) {
  return cursor.image()
             ? new MouseCursor(BasicDesktopFrame::CopyOf(*cursor.image()),
                               cursor.hotspot())
             : new MouseCursor();
}

}  // namespace webrtc
