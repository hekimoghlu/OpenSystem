/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_MOUSE_CURSOR_H_
#define MODULES_DESKTOP_CAPTURE_MOUSE_CURSOR_H_

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"
#include "modules/desktop_capture/desktop_geometry.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT MouseCursor {
 public:
  MouseCursor();

  // Takes ownership of `image`. `hotspot` must be within `image` boundaries.
  MouseCursor(DesktopFrame* image, const DesktopVector& hotspot);

  ~MouseCursor();

  MouseCursor(const MouseCursor&) = delete;
  MouseCursor& operator=(const MouseCursor&) = delete;

  static MouseCursor* CopyOf(const MouseCursor& cursor);

  void set_image(DesktopFrame* image) { image_.reset(image); }
  const DesktopFrame* image() const { return image_.get(); }

  void set_hotspot(const DesktopVector& hotspot) { hotspot_ = hotspot; }
  const DesktopVector& hotspot() const { return hotspot_; }

 private:
  std::unique_ptr<DesktopFrame> image_;
  DesktopVector hotspot_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MOUSE_CURSOR_H_
