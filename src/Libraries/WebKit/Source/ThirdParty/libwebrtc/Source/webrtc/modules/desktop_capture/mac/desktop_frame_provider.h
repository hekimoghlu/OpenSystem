/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#ifndef MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_PROVIDER_H_
#define MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_PROVIDER_H_

#include <CoreGraphics/CoreGraphics.h>
#include <IOSurface/IOSurface.h>

#include <map>
#include <memory>

#include "api/sequence_checker.h"
#include "modules/desktop_capture/shared_desktop_frame.h"
#include "sdk/objc/helpers/scoped_cftyperef.h"

namespace webrtc {

class DesktopFrameProvider {
 public:
  explicit DesktopFrameProvider(bool allow_iosurface);
  ~DesktopFrameProvider();

  DesktopFrameProvider(const DesktopFrameProvider&) = delete;
  DesktopFrameProvider& operator=(const DesktopFrameProvider&) = delete;

  // The caller takes ownership of the returned desktop frame. Otherwise
  // returns null if `display_id` is invalid or not ready. Note that this
  // function does not remove the frame from the internal container. Caller
  // has to call the Release function.
  std::unique_ptr<DesktopFrame> TakeLatestFrameForDisplay(
      CGDirectDisplayID display_id);

  // OS sends the latest IOSurfaceRef through
  // CGDisplayStreamFrameAvailableHandler callback; we store it here.
  void InvalidateIOSurface(CGDirectDisplayID display_id,
                           rtc::ScopedCFTypeRef<IOSurfaceRef> io_surface);

  // Expected to be called before stopping the CGDisplayStreamRef streams.
  void Release();

  bool allow_iosurface() const { return allow_iosurface_; }

 private:
  SequenceChecker thread_checker_;
  const bool allow_iosurface_;

  // Most recent IOSurface that contains a capture of matching display.
  std::map<CGDirectDisplayID, std::unique_ptr<SharedDesktopFrame>> io_surfaces_;
};

}  // namespace webrtc

#endif  // MODULES_DESKTOP_CAPTURE_MAC_DESKTOP_FRAME_PROVIDER_H_
