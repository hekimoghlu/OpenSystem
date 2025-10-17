/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_CGIMAGE_H_
#define MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_CGIMAGE_H_

#include <CoreGraphics/CoreGraphics.h>

#include <memory>

#include "modules/desktop_capture/desktop_frame.h"
#include "sdk/objc/helpers/scoped_cftyperef.h"

namespace webrtc {

class RTC_EXPORT DesktopFrameCGImage final : public DesktopFrame {
 public:
  // Create an image containing a snapshot of the display at the time this is
  // being called.
  static std::unique_ptr<DesktopFrameCGImage> CreateForDisplay(
      CGDirectDisplayID display_id);

  // Create an image containing a snaphot of the given window at the time this
  // is being called. This also works when the window is overlapped or in
  // another workspace.
  static std::unique_ptr<DesktopFrameCGImage> CreateForWindow(
      CGWindowID window_id);

  static std::unique_ptr<DesktopFrameCGImage> CreateFromCGImage(
      rtc::ScopedCFTypeRef<CGImageRef> cg_image);

  ~DesktopFrameCGImage() override;

  DesktopFrameCGImage(const DesktopFrameCGImage&) = delete;
  DesktopFrameCGImage& operator=(const DesktopFrameCGImage&) = delete;

 private:
  // This constructor expects `cg_image` to hold a non-null CGImageRef.
  DesktopFrameCGImage(DesktopSize size,
                      int stride,
                      uint8_t* data,
                      rtc::ScopedCFTypeRef<CGImageRef> cg_image,
                      rtc::ScopedCFTypeRef<CFDataRef> cg_data);

  const rtc::ScopedCFTypeRef<CGImageRef> cg_image_;
  const rtc::ScopedCFTypeRef<CFDataRef> cg_data_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_CGIMAGE_H_
