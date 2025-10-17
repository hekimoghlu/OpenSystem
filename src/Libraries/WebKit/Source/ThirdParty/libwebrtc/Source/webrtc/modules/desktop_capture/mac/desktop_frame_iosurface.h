/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_IOSURFACE_H_
#define MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_IOSURFACE_H_

#include <CoreGraphics/CoreGraphics.h>
#include <IOSurface/IOSurface.h>

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"
#include "sdk/objc/helpers/scoped_cftyperef.h"

namespace webrtc {

class DesktopFrameIOSurface final : public DesktopFrame {
 public:
  // Lock an IOSurfaceRef containing a snapshot of a display. Return NULL if
  // failed to lock.
  static std::unique_ptr<DesktopFrameIOSurface> Wrap(
      rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface);

  ~DesktopFrameIOSurface() override;

  DesktopFrameIOSurface(const DesktopFrameIOSurface&) = delete;
  DesktopFrameIOSurface& operator=(const DesktopFrameIOSurface&) = delete;

 private:
  // This constructor expects `io_surface` to hold a non-null IOSurfaceRef.
  explicit DesktopFrameIOSurface(rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface);

  const rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_IOSURFACE_H_
